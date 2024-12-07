"""Play a single episode with a trained model."""

from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from pathlib import Path

import gymnasium as gym
import torch
import torch.nn as nn

from tools import tools
from tools.agent import DiscreteActionAgent


def main(args: Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.log_dir = Path(args.log_dir).resolve()
    algo_name, env_id = args.log_dir.name.split("_")[:2]
    if args.checkpoint is None:
        ckpt_path = sorted((args.log_dir / "ckpts").glob("*_best_episode.pt"))[-1]
    else:
        ckpt_path = Path(args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")

    env = gym.make(env_id, render_mode="rgb_array")

    if args.seed is not None:
        print(f"Setting seed: {args.seed}")
        tools.set_seed_everywhere(args.seed)
        env.action_space.seed(args.seed)
        env.observation_space.seed(args.seed)
        torch.use_deterministic_algorithms(True)

    module_path, class_name = args.model_definition.rsplit(".", 1)
    module = tools.load_module_from_py_file(module_path)
    model_cls = getattr(module, class_name)
    model: nn.Module = model_cls(input_size=env.observation_space.shape[0], n_actions=env.action_space.n)

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.to(device)
    model.eval()

    agent = DiscreteActionAgent(model, device=device)

    frames = []
    obs, _ = env.reset()
    while True:
        action = agent.deterministic_policy(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            obs, _ = env.reset()
            break

    folder = args.log_dir / "play"
    folder.mkdir(exist_ok=True)
    tools.save_video(frames, folder / f"{ckpt_path.stem}.mp4", fps=30)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument("--log-dir", type=str, required=True, help="Path to the log directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--model-definition",
        type=str,
        required=True,
        help="Path to the model definition file: module_path.ClassName",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint file. If None, the best checkpoint will be used.",
    )
    args = parser.parse_args()

    main(args)
