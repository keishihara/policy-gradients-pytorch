"""
The Vanilla Policy Gradient method

NOTE
----
This script is tested on `PongNoFrameskip-v4`, but other Atari games are also supported.
Note that VPG algorithm does not perform well on Atari games with pixel-level inputs.

"""

import time
from argparse import ArgumentParser, RawTextHelpFormatter
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
import torch.optim as optim
from gymnasium import Env
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from tools import tools
from tools.agent import DiscreteActionAgent, preprocessor_atari
from tools.rollout import (
    RolloutSimulatorFirstLast,
    TransitionFirstLast,
)
from tools.wrappers import wrap_dqn

logger = tools.setup_logger("vpg_on_pong")


class MeanBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.deque = deque(maxlen=capacity)
        self.sum = 0.0

    def add(self, val: object) -> None:
        if len(self.deque) == self.capacity:
            self.sum -= self.deque[0]
        self.deque.append(val)
        self.sum += val

    def mean(self) -> float:
        if not self.deque:
            return 0.0
        return self.sum / len(self.deque)


class AtariPGN(nn.Module):
    def __init__(self, input_shape: tuple[int], n_actions: int) -> None:
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
        self.head = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def _get_conv_out(self, shape: tuple[int]) -> int:
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x: Tensor) -> Tensor:
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.head(conv_out)


def make_env(env_id: str, **kwargs) -> Env:
    return wrap_dqn(gym.make(env_id, **kwargs))


def main(args) -> None:
    # Load hyperparams and setup for logging
    config = tools.read_hyperparameters("vpg", args.env_id, args.hparams)
    tools.setup_logdir(args, f"vpg_{args.env_id}", subdirs=["video"])
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

    policy = AtariPGN(input_shape=input_shape, n_actions=n_actions)
    policy.apply(tools.orthgonal_initialization)
    policy.to(device)
    agent = DiscreteActionAgent(policy, preprocessor=preprocessor_atari, device=device)
    optimizer = optim.Adam(policy.parameters(), lr=config.learning_rate, eps=1e-3)

    rollout = RolloutSimulatorFirstLast(envs, agent, n_steps=config.reward_steps, gamma=config.gamma)
    reward_tracker = tools.RunningMeanTrackerForSingleMetric(
        "rollout/reward",
        writer,
        bound=config.mean_reward_bound,
        window_size=config.reward_window_size,
    )

    batch_obs = []
    batch_acts = []
    batch_phi = []
    baseline_buffer = MeanBuffer(config.baseline_steps)

    episode = 0
    gradient_step = 0
    running_mean_entropy = 0
    best_score = float("-inf")

    finished_rendering_this_epoch = False
    time_enter = time.time()

    # Collect experience by acting in the env with "current policy" (on-policy)
    for global_step, transition in enumerate(rollout):
        transition: TransitionFirstLast

        if args.render and not finished_rendering_this_epoch:
            envs[0].render()

        baseline_buffer.add(transition.reward)
        baseline = baseline_buffer.mean()
        writer.add_scalar("train/baseline", baseline, global_step)

        batch_obs.append(transition.obs)
        batch_acts.append(transition.action)
        batch_phi.append(transition.reward - baseline)

        # =========================== #
        #   End-of-episode handling   #
        # =========================== #

        info = rollout.pop_total_rewards()
        if info.is_terminal:
            total_reward = info.total_rewards[0]

            if reward_tracker.add(total_reward, global_step):
                # Training completed
                logger.info(
                    f"Task solved in {global_step+1:,} steps, {episode+1:,} episodes.\n"
                    f"Running mean return: {reward_tracker.get():.3f}\n"
                    f"Time taken: {tools.seconds_to_hms(time.time() - time_enter)}."
                )
                suffix = reward_tracker.as_suffix()
                dst = f"final_ckpt_{global_step=}_{episode=}_{suffix}.pt"
                tools.save_state_dict(policy, args.ckpts_dir / dst)
                break

            writer.add_scalar("rollout/lastest_reward", total_reward, global_step)
            if episode % config.console_log_freq == 0:
                logger.info(
                    f"{global_step}: done {episode} games, mean reward: {reward_tracker.get():.2f}, "
                    f"baseline: {baseline:.2f}"
                )

            finished_rendering_this_epoch = True
            episode += 1

        # ================== #
        #   Evalation step   #
        # ================== #

        if (
            global_step > 0  # Skip evaluation at the very beginning of training
            and global_step % config.eval_every_n_steps == 0
        ):
            policy.eval()
            res = tools.evaluate_policy(
                agent.deterministic_policy,
                make_env=make_env,
                env_kwargs={"env_id": args.env_id, "render_mode": "rgb_array"},
                n_eval_episodes=config.n_eval_episodes,
                seed=config.seed,
            )
            writer.add_scalar("eval/return_mean", res.episode_reward_mean, global_step)
            writer.add_scalar("eval/return_std", res.episode_reward_std, global_step)
            writer.add_scalar("eval/ep_len_mean", res.episode_length_mean, global_step)
            logger.info(
                f"Eval at step {global_step+1} - total reward: {res.episode_reward_mean:.2f} +/- {res.episode_reward_std:.2f} "
                f"across {config.n_eval_episodes} episodes.\n"
                f"Elapsed time {tools.seconds_to_hms(time.time() - time_enter)}"
            )

            suffix = reward_tracker.as_suffix()
            if res.episode_reward_mean >= best_score:
                logger.info(f"Best evaluation score updated from {best_score:.2f} to {res.episode_reward_mean:.2f}.")
                dst = f"{global_step=}_{episode=}_reward={res.best_episode_reward:.2f}_{suffix}_best_episode.{{ext}}"
                tools.save_video(res.best_episode_frames, args.video_dir / dst.format(ext="mp4"))
                tools.save_state_dict(policy, args.ckpts_dir / dst.format(ext="pt"))
                best_score = res.episode_reward_mean
            else:
                dst = f"{global_step=}_{episode=}_reward={res.best_episode_reward:.2f}_{suffix}.{{ext}}"
                tools.save_video(res.best_episode_frames, args.video_dir / dst.format(ext="mp4"))

            policy.train()

        if len(batch_obs) < config.batch_size:  # It was episodes_to_train in REINFORCE
            continue

        # ================ #
        #  Training step   #
        # ================ #

        batch_obs = preprocessor_atari(batch_obs).to(device)
        batch_acts = torch.tensor(np.int64(batch_acts)).long().to(device)
        batch_phi = torch.tensor(np.float32(batch_phi)).float().to(device)

        # Take a single gradient step on policy network
        optimizer.zero_grad()
        logits = policy(batch_obs)
        log_prob = F.log_softmax(logits, dim=1)
        log_prob_act = batch_phi * log_prob[range(len(batch_phi)), batch_acts]
        loss_policy = -log_prob_act.mean()

        prob = F.softmax(logits, dim=1)
        entropy = -(prob * log_prob).sum(dim=1).mean()
        loss_entropy = -config.entropy_beta * entropy
        loss = loss_policy + loss_entropy

        loss.backward()
        U.clip_grad_norm_(policy.parameters(), config.grad_l2_clip)
        optimizer.step()

        gradient_step += 1

        # ================ #
        #     Metrics      #
        # ================ #

        # Entropy
        prob = F.softmax(logits, dim=1)
        entropy = -(prob * log_prob).sum(dim=1).mean()
        running_mean_entropy = tools.smoothen(running_mean_entropy, entropy.item())

        # KL-div btwn previous policy and current policy
        new_logits: Tensor = policy(batch_obs)
        new_prob = F.softmax(new_logits, dim=1)
        kl_div = ((new_prob / prob).log() * new_prob).sum(dim=1).mean()

        grad_max = 0.0
        grad_means = 0.0
        grad_count = 0
        for p in policy.parameters():
            grad_max = max(grad_max, p.grad.abs().max().item())
            grad_means += (p.grad**2).mean().sqrt().item()
            grad_count += 1
        grad_l2 = grad_means / grad_count

        writer.add_scalar("time/global_step", global_step, global_step)
        writer.add_scalar("time/episode", episode, global_step)
        writer.add_scalar("time/gradient_step", gradient_step, global_step)
        writer.add_scalar("time/time_elapsed", time.time() - time_enter, global_step)
        writer.add_scalar("train/batch_size", len(batch_obs), global_step)
        writer.add_scalar("train/loss", loss.item(), global_step)
        writer.add_scalar("train/loss_policy", loss_policy.item(), global_step)
        writer.add_scalar("train/loss_entropy", loss_entropy.item(), global_step)
        writer.add_scalar("train/grad_l2", grad_l2, global_step)
        writer.add_scalar("train/grad_max", grad_max, global_step)
        writer.add_scalar("train/batch_phi", batch_phi.mean().item(), global_step)
        writer.add_scalar("train/batch_phi_std", batch_phi.std().item(), global_step)
        writer.add_scalar("metrics/running_mean_entropy", running_mean_entropy, global_step)
        writer.add_scalar("metrics/kl_div", kl_div.item(), global_step)

        batch_obs = []
        batch_acts = []
        batch_phi = []
        finished_rendering_this_epoch = False


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument("--env-id", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--cuda", action="store_true", help="Enable gpu computation.")
    parser.add_argument("--render", action="store_true")
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
