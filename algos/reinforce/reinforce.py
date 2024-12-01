"""
The REINFORCE method

To estimate the state value, this implementation uses the discounted
reward-to-go, which is the cumulative sum of rewards from subsequent
steps starting from thecurrent state.


Algorithm
---------
1. Initialize the network with random weights.
2. Play N full episodes, saving (s, a, r, s') transitions (i.e., trajectories).
3. For every step t of every episdoe k, calculate the discounted total reward
   for subsequent steps:
   Q_(k,t) = sum_(i=t)^T (gamma^i * r_i) (discounted rewards-to-go)
4. Calculate the loss function for all transitions:
   L = - sum_(k,t) Q_(k,t) * log pi(s_(k,t), a_(k,t))
5. Perform an SGD update of the weights, minimizing the loss.
6. Repeat from step 2 until convergence.

NOTE
----
In this code, policy updates are performed on an episodic basis, which means
the batch size is not always consistent in some games; for a simple task like
CartPole, this would lead to faster convergence but is not a common practice.

"""

import time
from argparse import ArgumentParser, RawTextHelpFormatter

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium import Env
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from tools import tools
from tools.agent import DiscreteActionAgent, default_preprocessor

logger = tools.setup_logger("reinforce")


class PolicyModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_actions: int,
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


def calc_q_values(rewards: list[float], gamma: float = 0.99) -> list[float]:
    """Calculate discounted cumulative sum of rewards."""
    discounted_rewards = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= gamma
        sum_r += r
        discounted_rewards.append(sum_r)
    return list(reversed(discounted_rewards))


def main(args) -> None:
    config = tools.read_hyperparameters("reinforce", args.env_id, args.hparams)
    tools.set_seed_everywhere(config.seed, deterministic=config.deterministic)
    tools.setup_logdir(args, f"reinforce_{args.env_id}", subdirs=["video"])
    config.to_json(args.log_dir / "config.json", indent=4)

    writer = SummaryWriter(args.log_dir)
    tools.add_file_handler(logger, args.log_dir / "console.log")
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env: Env = gym.make(id=args.env_id, render_mode="rgb_array")
    env.action_space.seed(config.seed)

    input_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = PolicyModel(input_size=input_size, n_actions=n_actions)
    policy.to(device)
    agent = DiscreteActionAgent(policy, device=device)
    optimizer = optim.Adam(policy.parameters(), config.learning_rate)

    reward_tracker = tools.RunningMeanTrackerForSingleMetric(
        "rollout/reward",
        writer,
        bound=config.mean_reward_bound,
        window_size=config.reward_window_size,
    )

    batch_obs = []
    batch_acts = []
    batch_qvals = []

    step = 0
    episode = 0
    batch_episodes = 0
    gradient_step = 0
    running_mean_entropy = 0
    best_score = -1

    obs, _ = env.reset(seed=config.seed)
    terminated, truncated = False, False
    ep_rewards = []
    finished_rendering_this_epoch = False
    time_enter = time.time()

    # Collect transitions by acting in the env with "current policy" (on-policy)
    while step < config.max_steps:
        if args.render and not finished_rendering_this_epoch:
            env.render()

        action = agent(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        batch_obs.append(obs)
        batch_acts.append(action)
        ep_rewards.append(reward)

        obs = next_obs
        timeout = len(ep_rewards) == config.episode_limit_steps
        timeout |= truncated

        # =========================== #
        #   End-of-episode handling   #
        # =========================== #

        if terminated or truncated:
            # You would want to add timeout handling if needed, eg, making reward = 0
            # to punish the learning policy for long lasting episodes

            # Add Q values (reward-to-go) of this episode
            batch_qvals.extend(calc_q_values(ep_rewards, config.gamma))

            total_reward = sum(ep_rewards)
            if reward_tracker.add(total_reward, step):
                msg = "\n" + (
                    f"Task solved in {step+1:,} steps, {episode+1:,} episodes.\n"
                    f"Running mean return: {reward_tracker.get():.3f}\n"
                    f"Time taken: {tools.seconds_to_hms(time.time() - time_enter)}.\n"
                )
                logger.info(msg)
                suffix = reward_tracker.as_suffix()
                dst = f"final_ckpt_{step=}_{episode=}_{suffix}.pt"
                tools.save_state_dict(policy, args.ckpts_dir / dst)
                break

            writer.add_scalar("rollout/lastest_reward", total_reward, step)
            if episode % config.console_log_freq == 0:
                logger.info(f"{step}: done {episode} games, mean reward {reward_tracker.get():.3f} ")

            obs, _ = env.reset()
            terminated, truncated, ep_rewards = False, False, []
            finished_rendering_this_epoch = True
            batch_episodes += 1
            episode += 1

        step += 1

        # ============= #
        #   Evaluation  #
        # ============= #

        if (
            step > 0  # Skip evaluation at the very beginning of training
            and step % config.eval_every_n_steps == 0
        ):
            policy.eval()
            reward, rew_std, ep_len_mean, frames = tools.evaluate_policy(
                agent.deterministic_policy,
                env=gym.make(args.env_id, render_mode="rgb_array"),
                n_eval_episodes=config.n_eval_episodes,
                return_frames=True,
            )
            writer.add_scalar("eval/return_mean", reward, step)
            writer.add_scalar("eval/return_std", rew_std, step)
            writer.add_scalar("eval/ep_len_mean", ep_len_mean, step)
            logger.info(
                f"Eval at step {step} - total reward: {reward:.3f} +/- {rew_std:.3f} "
                f"across {config.n_eval_episodes} episodes.\n"
                f"Elapsed time {tools.seconds_to_hms(time.time() - time_enter)}\n"
            )
            suffix = reward_tracker.as_suffix()
            if reward >= best_score:
                logger.info(f"Best evaluation score updated from {best_score:.3f} to {reward:.3f}.")
                # Save video
                dst = f"best_model_{step=}_{episode=}_{reward=:.3f}_{suffix}.{{ext}}"
                tools.save_video(frames[::2], args.video_dir / dst.format(ext="gif"))
                # Save checkpoint
                tools.save_state_dict(policy, args.ckpts_dir / dst.format(ext="pt"))
                best_score = reward
            else:
                # Save video
                dst = f"eval_replay_{step=}_{episode=}_{reward=:.3f}_{suffix}.{{ext}}"
                tools.save_video(frames[::2], args.video_dir / dst.format(ext="gif"))

            policy.train()

        if batch_episodes < config.episodes_to_train:
            continue

        # ================ #
        #  Training step   #
        # ================ #

        batch_obs = default_preprocessor(batch_obs).to(device)
        batch_acts = torch.tensor(np.int64(batch_acts)).long().to(device)
        batch_qvals = torch.tensor(np.float32(batch_qvals)).float().to(device)

        # Take a single gradient step on policy network
        optimizer.zero_grad()
        logits = policy(batch_obs)
        log_probs = F.log_softmax(logits, dim=1)
        log_probs_act = batch_qvals * log_probs[range(len(batch_qvals)), batch_acts]
        policy_gradient = log_probs_act.mean()
        loss = -1 * policy_gradient
        loss.backward()
        optimizer.step()

        gradient_step += 1

        # ================ #
        #     Metrics      #
        # ================ #

        # Entropy
        prob = F.softmax(logits, dim=1)
        entropy = -(prob * log_probs).sum(dim=1).mean()
        running_mean_entropy = tools.smoothen(running_mean_entropy, entropy.item())

        # KL-div btwn previous policy and current policy
        with torch.inference_mode():
            new_logits: Tensor = policy(batch_obs)
        new_prob = F.softmax(new_logits, dim=1)
        kl_div = ((new_prob / prob).log() * new_prob).sum(dim=1).mean().item()

        writer.add_scalar("time/episode", episode, step)
        writer.add_scalar("time/gradient_step", gradient_step, step)
        writer.add_scalar("time/time_elapsed", time.time() - time_enter, step)
        writer.add_scalar("train/batch_size", len(batch_obs), step)
        writer.add_scalar("train/loss", loss.item(), step)
        writer.add_scalar("metrics/running_mean_entropy", running_mean_entropy, step)
        writer.add_scalar("metrics/kl_div", kl_div, step)

        batch_obs = []
        batch_acts = []
        batch_qvals = []
        batch_episodes = 0
        finished_rendering_this_epoch = False


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--cuda", action="store_true", help="Enable gpu computation.")
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--hparams",
        type=str,
        nargs="+",
        action=tools.StoreDict,
        help="Overwrite hyperparameter (Usage: --hparams learning_rate:0.01 train_freq:10)",
    )
    args = parser.parse_args()

    main(args)