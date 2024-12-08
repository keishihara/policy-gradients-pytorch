"""
The Advantage Actor Critic (A2C) method

This method improves upon the Vanilla Policy Gradient by using both a policy
and a value function. The value function serves as a baseline to reduce the
variance of policy gradient updates. Unlike REINFORCE, where updates are done
at the end of an episode, A2C performs updates more frequently and can leverage
parallel environments for faster convergence.

Key Points
----------
- The baseline used to reduce variance is now state-dependent, i.e., the value
  function predicts the expected discounted reward-to-go for a given state.
- A2C works with both the advantage function (A = Q - V) and policy gradients
  to update the model.
- The policy is updated by maximizing the expected reward while minimizing
  the policy loss (log probability weighted by the advantage) and adding
  entropy regularization to encourage exploration.
- The value network is trained using Mean Squared Error (MSE) between the
  predicted value and the discounted reward-to-go.

"""

import time
from argparse import ArgumentParser, RawTextHelpFormatter

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
import torch.optim as optim
from gymnasium import Env
from numpy import ndarray
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from tools import tools
from tools.agent import (
    Actions,
    DiscreteActionAgent,
    Observations,
    Preprocessor,
    preprocessor_atari,
)
from tools.rollout import (
    RolloutSimulatorFirstLast,
    TransitionFirstLast,
)
from tools.wrappers import wrap_dqn

logger = tools.setup_logger("a2c")


class AtariA2C(nn.Module):
    def __init__(self, input_shape: tuple, n_actions: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.policy = nn.Sequential(nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, n_actions))

        self.value = nn.Sequential(nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, 1))

    def _get_conv_out(self, shape: tuple) -> int:
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.policy(conv_out), self.value(conv_out)


def unpack_batch(
    batch: list[TransitionFirstLast],
    model: nn.Module,
    gamma: float = 0.99,
    reward_steps: int = 4,
    preprocessor: Preprocessor = preprocessor_atari,
    device: torch.device = "cpu",
) -> tuple[Tensor, ...]:
    observations = []
    actions = []
    rewards = []
    not_done_idx = []
    next_obs = []
    transition: TransitionFirstLast
    for idx, transition in enumerate(batch):
        observations.append(np.asarray(transition.obs))
        actions.append(int(transition.action))
        rewards.append(transition.reward)
        if transition.next_obs is not None:
            not_done_idx.append(idx)
            next_obs.append(np.asarray(transition.next_obs))

    observations = preprocessor(observations).to(device)
    actions = torch.tensor(actions).long().to(device)

    rewards = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        next_obs: Tensor = preprocessor(next_obs).to(device)
        next_vals: Tensor = model(next_obs)[1]
        next_vals: ndarray = next_vals.detach().cpu().numpy()[:, 0]

        # Calculate value target using Bellman Equation of value function
        rewards[not_done_idx] += gamma**reward_steps * next_vals

    reference_values = torch.tensor(rewards).float().to(device)
    return observations, actions, reference_values


def make_env(env_id: str, **kwargs) -> Env:
    return wrap_dqn(gym.make(env_id, **kwargs))


def main(args) -> None:
    # Load hyperparams and setup for logging
    config = tools.read_hyperparameters("a2c", args.env_id, args.hparams)
    tools.setup_logdir(args, f"a2c_{args.env_id}{args.infix}", subdirs=["video"])
    config.to_json(args.log_dir / "config.json", indent=4)
    writer = SummaryWriter(args.log_dir)
    tools.add_file_handler(logger, args.log_dir / "console.log")

    tools.set_seed_everywhere(config.seed, deterministic=config.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Instantiate the environment
    envs: list[Env] = [make_env(args.env_id) for _ in range(config.n_envs)]
    [envs[i].action_space.seed(config.seed + i) for i in range(config.n_envs)]

    input_shape = envs[0].observation_space.shape  # (4, 84, 84)
    n_actions = envs[0].action_space.n

    # Build actor and critic models
    model = AtariA2C(input_shape=input_shape, n_actions=n_actions)
    model.apply(tools.orthgonal_initialization)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-3)

    agent = DiscreteActionAgent(
        lambda x: model(x)[0],
        preprocessor=preprocessor_atari,
        device=device,
    )

    def eval_agent(obs: Observations) -> Actions:
        return agent(obs, deterministic=True)

    rollout = RolloutSimulatorFirstLast(envs, agent, n_steps=config.reward_steps, gamma=config.gamma)
    logger.info(rollout)

    reward_tracker = tools.RunningMeanTrackerForSingleMetric(
        "rollout/reward",
        writer,
        bound=config.mean_reward_bound,
        window_size=config.reward_window_size,
    )
    batched_writer = tools.BatchedSummaryWriter(writer, batch_size=10)

    episode = 0
    gradient_step = 0
    running_mean_entropy = 0
    best_score = -1
    time_enter = time.time()
    batch = []

    transition: TransitionFirstLast
    for global_step, transition in enumerate(rollout):
        batch.append(transition)

        # =========================== #
        #   End-of-episode handling   #
        # =========================== #

        info = rollout.pop_total_rewards()
        if info.is_terminal:
            total_reward = info.total_rewards[0]

            if reward_tracker.add(total_reward, global_step):
                # Training completed
                logger.info(
                    f"Task solved in {global_step:,} steps, {episode:,} episodes.\n"
                    f"Running mean reward: {reward_tracker.get():.3f}\n"
                    f"Time taken: {tools.seconds_to_hms(time.time() - time_enter)}."
                )
                suffix = reward_tracker.as_suffix()
                dst = f"final_ckpt_{global_step=}_{episode=}_{suffix}.pt"
                tools.save_state_dict(model, args.ckpts_dir / dst)
                break

            writer.add_scalar("rollout/latest_reward", total_reward, global_step)
            if episode % config.console_log_freq == 0:
                logger.info(f"{global_step}: done {episode} games, mean reward {reward_tracker.get():.3f}")
            episode += 1

        # ================== #
        #   Evalation step   #
        # ================== #

        if (
            global_step > 0  # Skip evaluation at the very beginning of training
            and global_step % config.eval_every_n_steps == 0
        ):
            model.eval()
            res = tools.evaluate_policy(
                eval_agent,
                make_env=make_env,
                env_kwargs={"env_id": args.env_id, "render_mode": "rgb_array"},
                n_eval_episodes=config.n_eval_episodes,
                seed=config.seed,
            )
            writer.add_scalar("eval/return_mean", res.episode_reward_mean, global_step)
            writer.add_scalar("eval/return_std", res.episode_reward_std, global_step)
            writer.add_scalar("eval/ep_len_mean", res.episode_length_mean, global_step)
            logger.info(
                f"Eval at step {global_step} - total reward: {res.episode_reward_mean:.2f} +/- {res.episode_reward_std:.2f} "
                f"across {config.n_eval_episodes} episodes.\n"
                f"Elapsed time {tools.seconds_to_hms(time.time() - time_enter)}"
            )

            suffix = reward_tracker.as_suffix()
            if res.episode_reward_mean >= best_score:
                logger.info(f"Best evaluation score updated from {best_score:.2f} to {res.episode_reward_mean:.2f}.")
                dst = f"{global_step=}_{episode=}_reward={res.best_episode_reward:.2f}_{suffix}_best_episode.{{ext}}"
                tools.save_video(res.best_episode_frames, args.video_dir / dst.format(ext="mp4"))
                tools.save_state_dict(model, args.ckpts_dir / dst.format(ext="pt"))
                best_score = res.episode_reward_mean
            else:
                dst = f"{global_step=}_{episode=}_reward={res.best_episode_reward:.2f}_{suffix}.{{ext}}"
                tools.save_video(res.best_episode_frames, args.video_dir / dst.format(ext="mp4"))

            model.train()

        if len(batch) < config.batch_size:
            continue

        # ================ #
        #  Training step   #
        # ================ #

        obs, acts, vals_ref = unpack_batch(
            batch,
            model,
            gamma=config.gamma,
            reward_steps=config.reward_steps,
            preprocessor=preprocessor_atari,
            device=device,
        )
        batch.clear()

        optimizer.zero_grad()
        # Get model predictions of policy logits and values
        logits, values = model(obs)

        # Calculate loss for value network
        loss_value = F.mse_loss(values.squeeze(-1), vals_ref)

        # Calculate policy gradients (loss for policy network)
        log_prob = F.log_softmax(logits, dim=1)
        advantage = vals_ref - values.squeeze(-1).detach()
        log_prob_act = advantage * log_prob[range(config.batch_size), acts]
        loss_policy = -1 * log_prob_act.mean()

        # Calculate entropy loss
        prob = F.softmax(logits, dim=1)
        entropy = -(prob * log_prob).sum(dim=1).mean()
        loss_entropy = -1 * config.entropy_beta * entropy

        # Backpropagate the policy loss only. retain_graph=True ensures the computation graph is
        # not freed, allowing further backward passes to accumulate gradients on the same graph.
        loss_policy.backward(retain_graph=True)

        # Gather actual policy gradient values across the policy network
        grads = np.concatenate(
            [p.grad.detach().cpu().numpy().flatten() for p in model.parameters() if p.grad is not None]
        )
        grad_max = np.max(np.abs(grads))
        grad_l2 = np.sqrt(np.mean(np.square(grads)))

        # Backpropagate entropy and value losses
        loss = loss_entropy + loss_value
        loss.backward()
        U.clip_grad_norm_(model.parameters(), config.grad_l2_clip)
        optimizer.step()

        # Get full loss
        loss += loss_policy

        # Notify when some metrics exceeded some threshold
        if loss_policy > 1.0:
            logger.warning(f"loss_policy became larger than 1.0: {loss_policy=:.4f}")
        if grad_max > 1.0:
            logger.warning(f"grad_max became larger than 1.0: {grad_max=:.4f}")

        # ================ #
        #     Metrics      #
        # ================ #

        # Entropy
        running_mean_entropy = tools.smoothen(running_mean_entropy, entropy.item())

        # KL-div btwn previous policy and current policy
        with torch.inference_mode():
            new_logits: Tensor = model(obs)[0]
        new_prob = F.softmax(new_logits, dim=1)
        kl_div = ((new_prob / prob).log() * new_prob).sum(dim=1).mean().item()

        writer.add_scalar("time/episode", episode, global_step)
        writer.add_scalar("time/gradient_step", gradient_step, global_step)
        writer.add_scalar("time/elapsed_time", time.time() - time_enter, global_step)

        batched_writer.add("train/batch_advantage", advantage, global_step)
        batched_writer.add("train/batch_values", values, global_step)
        batched_writer.add("train/batch_rewards", vals_ref, global_step)
        batched_writer.add("train/loss_entropy", loss_entropy, global_step)
        batched_writer.add("train/loss_policy", loss_policy, global_step)
        batched_writer.add("train/loss_value", loss_value, global_step)
        batched_writer.add("train/loss", loss, global_step)
        batched_writer.add("train/grad_l2", grad_l2, global_step)
        batched_writer.add("train/grad_max", grad_max, global_step)
        batched_writer.add("train/grad_std", np.std(grads), global_step)
        batched_writer.add("metrics/running_mean_entropy", running_mean_entropy, global_step)
        batched_writer.add("metrics/kl_div", kl_div, global_step)

        gradient_step += 1


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument("--env-id", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--cuda", action="store_true", help="Enable gpu computation.")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--infix", type=str, default="")
    parser.add_argument(
        "--hparams",
        type=str,
        nargs="+",
        action=tools.StoreDict,
        help="Overwrite hyperparameter (Usage: --hyperparams learning_rate:0.01 train_freq:10)",
    )
    args = parser.parse_args()

    if args.env_id not in tools.list_registered_envs_by_category("atari"):
        raise ValueError("This script is only supported for Atari environments.")

    main(args)
