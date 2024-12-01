"""
The Vanilla Policy Gradient method

Key Points
----------
Improvements over the REINFORCE method include:
- Training can start without requiring full episodes, allowing for updates
  at each step (i.e., it can be on-policy with step-wise updates).
- High variance in gradients is reduced by subtracting a baseline (typically
  the state value) from the total reward.
- Exploration is improved by incorporating an entropy loss term to encourage
  more diverse action sampling.
- A more accurate estimation of future rewards is achieved by unrolling n-steps
  forward and using reward-to-go instead of full episode returns.


*This script is for training on `PongNoFrameskip-v4`, but you can also train on other Atari games.
"""

import shutil
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
from numpy import ndarray
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from tools import tools
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
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.head(conv_out)


class DiscretePolicyAtariAgent:
    def __init__(self, model, device="cpu") -> None:
        self.model = model
        self.device = device

    def __call__(self, obs, deterministic: bool = False):
        if deterministic:
            return self.deterministic_policy(obs)
        else:
            return self.stochastic_policy(obs)

    def preprocessing(self, obs: ndarray | list) -> Tensor:
        if isinstance(obs, list):
            obs = np.array(obs)
        if obs.ndim == 3:
            obs = obs[None, ...]  # add batch dim
        obs = obs / 255
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        return obs

    @torch.inference_mode()
    def stochastic_policy(self, obs: ndarray | list) -> int:
        obs = self.preprocessing(obs)
        logits = self.model(obs)  # (B, A)
        probs = F.softmax(logits, dim=1)  # (B, A)
        probs = probs.detach()
        if probs.shape[0] == 1:
            prob = probs.squeeze(0)
            # Sample action for a single observation
            action = torch.multinomial(prob, num_samples=1).item()
            return action  # int
        # Sample actions for each batch in `obs`
        actions = torch.multinomial(probs, num_samples=1)
        return actions.squeeze(1).cpu().numpy()  # (B,)

    @torch.inference_mode()
    def deterministic_policy(self, obs: ndarray | list) -> int | ndarray:
        obs = self.preprocessing(obs)
        logits = self.model(obs)  # (B, A)
        probs = F.softmax(logits, dim=1)  # (B, A)
        acts = probs.argmax(dim=1)  # (B,)
        acts = acts.detach().cpu().numpy()  # Convert actions to numpy
        if acts.shape[0] == 1:
            return acts.item()  # Return a single action for batch size 1
        return acts  # Return actions as a numpy array for batch size > 1


def is_task_solved(config, running_mean_reward: float, **kwargs) -> bool:
    return running_mean_reward >= config.mean_reward_bound


def make_env(env_id: str, **kwargs) -> Env:
    return wrap_dqn(gym.make(env_id, **kwargs))


def main(args) -> None:
    config = tools.read_hyperparameters("vpg", args.env_id, args.hparams)
    tools.set_seed_everywhere(config.seed, deterministic=config.deterministic)
    tools.setup_logdir(args, f"vpg_{args.env_id}", subdirs=["video"])
    config.to_json(args.log_dir / "config.json", indent=4)

    writer = SummaryWriter(args.log_dir)
    tools.add_file_handler(logger, args.log_dir / "console.log")
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs: list[Env]
    envs = [make_env(args.env_id) for _ in range(config.n_envs)]
    [envs[i].action_space.seed(config.seed + i) for i in range(config.n_envs)]

    input_shape = envs[0].observation_space.shape  # (4, 84, 84)
    n_actions = envs[0].action_space.n

    policy = AtariPGN(input_shape=input_shape, n_actions=n_actions)
    policy.apply(tools.orthgonal_initialization)
    policy.to(device)
    agent = DiscretePolicyAtariAgent(policy, device)
    optimizer = optim.Adam(policy.parameters(), lr=config.learning_rate, eps=1e-3)

    rollout = RolloutSimulatorFirstLast(envs, agent, n_steps=config.reward_steps, gamma=config.gamma)

    batch_obs = []  # for observations
    batch_acts = []  # for actions
    batch_phi = []
    total_rewards = []
    baseline_buffer = MeanBuffer(config.baseline_steps)

    episode = 0
    batch_episodes = 0
    gradient_step = 0
    running_mean_reward = 0
    running_mean_entropy = 0
    best_score = -1

    finished_rendering_this_epoch = False
    time_enter = time.time()

    # Collect experience by acting in the env with "current policy" (on-policy)
    for step, transition in enumerate(rollout):
        transition: TransitionFirstLast

        if args.render and not finished_rendering_this_epoch:
            envs[0].render()

        baseline_buffer.add(transition.reward)
        baseline = baseline_buffer.mean()
        writer.add_scalar("train/baseline", baseline, step)

        batch_obs.append(transition.obs)
        batch_acts.append(transition.action)
        batch_phi.append(transition.reward - baseline)

        # =========================== #
        #   End-of-episode handling   #
        # =========================== #

        info = rollout.pop_total_rewards()
        if info.is_terminal:
            ep_ret = info.total_rewards[0]
            total_rewards.append(ep_ret)
            running_mean_reward = np.mean(total_rewards[-config.reward_window_size :])
            writer.add_scalar("rollout/lastest_ep_reward", total_rewards[-1], step)
            writer.add_scalar("rollout/running_mean_reward", running_mean_reward, step)

            if episode % config.console_log_freq == 0:
                logger.info(
                    f"Step: {step}, Episode: {episode} - latest episode reward: {ep_ret:.2f}, "
                    f"running mean reward: {running_mean_reward:.2f}, baseline: {baseline:.2f}"
                )

            finished_rendering_this_epoch = True
            batch_episodes += 1
            episode += 1

        # ================== #
        #   Evalation step   #
        # ================== #

        if step > 0 and step % config.eval_every_n_steps == 0 or is_task_solved(**locals()):
            policy.eval()
            ret_mean, ret_std, ep_len_mean, frames = tools.evaluate_policy(
                agent.deterministic_policy,
                make_env=make_env,
                env_kwargs={"env_id": args.env_id, "render_mode": "rgb_array"},
                n_eval_episodes=config.n_eval_episodes,
                return_frames=True,
                seed=config.seed,
            )
            writer.add_scalar("eval/return_mean", ret_mean, step)
            writer.add_scalar("eval/return_std", ret_std, step)
            writer.add_scalar("eval/ep_len_mean", ep_len_mean, step)
            filename = f"eval_replay_{step=}_{episode=}_reward={ret_mean}.gif"
            tools.save_video(frames[::2], args.video_dir / filename)
            logger.info(
                f"Eval at step {step+1} - total reward: {ret_mean:.2f} +/- {ret_std:.2f} "
                f"across {config.n_eval_episodes} episodes."
            )

            if ret_mean >= best_score:
                dst = f"new_record_{step=}_{episode=}_reward={ret_mean:.1f}.gif"
                shutil.copy(args.video_dir / filename, args.video_dir / dst)

                filename = f"ckpt_{step=}_{episode=}_reward={ret_mean:.1f}.pt"
                tools.save_state_dict(policy, args.ckpts_dir / filename)
                best_score = ret_mean

            # これがここにあるのはおかしいかもしれない。学習が完了するタイミングは、評価のタイミングではなく、rolloutが基準？
            if is_task_solved(**locals()):
                msg = "\n" + (
                    f"Task solved in {step+1:,} steps, {episode+1:,} episodes.\n"
                    f"Running mean return: {running_mean_reward:.2f}\n"
                    f"Time taken: {tools.seconds_to_hms(time.time() - time_enter)}.\n"
                )
                logger.info(msg)
                break

            policy.train()

        if len(batch_obs) < config.batch_size:  # It was episodes_to_train in REINFORCE
            continue

        # ================ #
        #  Training step   #
        # ================ #

        batch_obs = torch.as_tensor(np.float32(batch_obs)).float().to(device)
        batch_acts = torch.as_tensor(np.int64(batch_acts)).long().to(device)
        batch_phi = torch.as_tensor(np.float32(batch_phi)).float().to(device)

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

        writer.add_scalar("time/global_step", step, step)
        writer.add_scalar("time/episode", episode, step)
        writer.add_scalar("time/gradient_step", gradient_step, step)
        writer.add_scalar("time/time_elapsed", time.time() - time_enter, step)
        writer.add_scalar("train/batch_size", len(batch_obs), step)
        writer.add_scalar("train/loss", loss.item(), step)
        writer.add_scalar("train/loss_policy", loss_policy.item(), step)
        writer.add_scalar("train/loss_entropy", loss_entropy.item(), step)
        writer.add_scalar("train/grad_l2", grad_l2, step)
        writer.add_scalar("train/grad_max", grad_max, step)
        writer.add_scalar("train/batch_phi", batch_phi.mean().item(), step)
        writer.add_scalar("train/batch_phi_std", batch_phi.std().item(), step)
        writer.add_scalar("metrics/running_mean_entropy", running_mean_entropy, step)
        writer.add_scalar("metrics/kl_div", kl_div.item(), step)

        batch_obs = []
        batch_acts = []
        batch_phi = []
        batch_episodes = 0
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

    main(args)
