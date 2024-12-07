import argparse
import datetime
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import scipy
import torch
import torch.nn as nn
import yaml
from box import Box
from gymnasium import Env
from joblib import Parallel, delayed
from rich.logging import RichHandler
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm as std_tqdm

tqdm = partial(std_tqdm, dynamic_ncols=True)

HYPERPARAMS_DIR = Path(__file__).parents[1] / "hparams"
LOG_DIR = Path(__file__).parents[1] / "logs"


def save_video(
    frames: list[np.ndarray] | np.ndarray,
    filename: str,
    fps: int = 60,
    align_macro_block_size: bool = True,
) -> None:
    """Save video as gif or mp4 as filename suggests

    Args:
        frames (array-like): frames to save
        filename (str): filename of a video clip. Allowed extentions are: `mp4` or `gif`
        fps (int, optional): frame rate. Defaults to 60.
        align_macro_block_size (bool, optional): Whether to align the macro block size of the frames. Defaults to True.
    Note:
        fps can be 60 at the most when saving as .gif otherwise weirdly slow gif.
        However, mp4 accepts wider range. Also, gif is typically 100x larger in
        size than mp4.

    Example:
    >>> noisy_frames = np.random.randint(0, 255, (32, 400, 608, 3), dtype=np.uint8)
    >>> save_video(noisy_frames, './logs/demo.gif')

    """

    filename = Path(filename).resolve()
    dirname = filename.parent
    dirname.mkdir(parents=True, exist_ok=True)

    if align_macro_block_size:
        h, w = frames[0].shape[:2]
        nh, nw = (h + 15) // 16 * 16, (w + 15) // 16 * 16
        frames = [Image.fromarray(frame).resize((nw, nh)) for frame in frames]

    frames = np.uint8(frames)

    if filename.suffix == ".gif":
        kwargs = {"duration": 1 / fps}
        imageio.mimsave(filename, frames, **kwargs)
    elif filename.suffix == ".mp4":
        kwargs = {"fps": fps}
        imageio.mimsave(filename, frames, **kwargs)
    else:
        raise ValueError(f"Not supported file type: {filename}")

    print(f"A video saved at: {str(filename)}")


# copied from: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/utils.py#L340
class StoreDict(argparse.Action):
    """
    Custom argparse action for storing dict.
    In: args1:0.0 args2:"dict(a=1)"
    Out: {'args1': 0.0, arg2: dict(a=1)}
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDict, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        arg_dict = {}
        for arguments in values:
            key = arguments.split(":")[0]
            value = ":".join(arguments.split(":")[1:])
            # Evaluate the string as python code
            arg_dict[key] = eval(value)
        setattr(namespace, self.dest, arg_dict)


def read_hyperparameters(
    algo_name: str = "reinforce",
    env_id: str = "CartPole-v3",
    custom_hyperparams: dict = None,
    folder: str | Path = HYPERPARAMS_DIR,
) -> Box:
    """
    Load hyperparameters from yaml file, and update it with given custom
    hyperparameters as needed.
    """

    folder = Path(folder)

    if (folder / f"{algo_name}.yaml").is_file():
        filename = folder / f"{algo_name}.yaml"
    elif (folder / f"{algo_name}.yml").is_file():
        filename = folder / f"{algo_name}.yml"
    else:
        raise FileNotFoundError(f"{folder}/{algo_name}.{{yml/yaml}}")

    def recursive_update(frm, to) -> None:
        for k, v in to.items():
            if isinstance(v, dict) and k in frm:
                recursive_update(frm[k], v)
            else:
                frm[k] = v

    hyperparams = {}

    with open(filename) as f:
        hyperparams_dict = yaml.safe_load(f)
        if env_id not in list(hyperparams_dict.keys()):
            raise ValueError(f"Hyperparameters not found for {algo_name}-{env_id}")

        recursive_update(hyperparams, hyperparams_dict.get("defaults", {}))
        recursive_update(hyperparams, hyperparams_dict[env_id])

    if custom_hyperparams is not None:
        recursive_update(hyperparams, custom_hyperparams)

    # Evaluate the item if whoes key starts with `eval:`
    for k, v in hyperparams.items():
        if isinstance(v, str) and v.startswith("eval:"):
            hyperparams[k] = eval(v.replace("eval:", ""))

    return Box(hyperparams)


def setup_logdir(
    args: argparse.Namespace,
    prefix: str = "run",
    log_root_dir: str = LOG_DIR,
    subdirs: list[str] | None = None,
    create_ckpts_dir: bool = True,
) -> None:
    """
    Set up the logging directory structure and assign the paths to the
    corresponding attributes in the `script_args` parameter. Also sets the
    generated `run_name` to `script_args.run_name`.

    The directory structure created will be as follows:

    Args:
        args (argparse.Namespace)
        prefix (str)
        makedirs (bool): Whether to create the directories. Default is based on
            `is_main_process()`.
        subdirs (Iterable[str] | None): List of subdirectories to create under
            the logging directory. Default is None.
    """

    timestamp = time.strftime("%y%m%dT%H%M%S%z")
    run_name = f"{prefix}_{timestamp}"
    args.run_name = run_name
    args.log_dir = Path(log_root_dir) / run_name
    args.log_dir.mkdir(exist_ok=True, parents=True)

    subdirs = list(subdirs) if subdirs is not None else []
    if create_ckpts_dir and "ckpts" not in subdirs:
        subdirs.append("ckpts")

    for subdir in subdirs:
        subdir = subdir.replace("/", "_")  # make sure do not make nested folders
        # map subdir to the namespace like: foo ---> args.foo_dir
        setattr(args, subdir + "_dir", args.log_dir / subdir)
        getattr(args, f"{subdir}_dir").mkdir(exist_ok=True)


# https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/core.py#L29
def discounted_inverse_cumsum(
    rewards: list | np.ndarray,
    gamma: float,
    dtype="float32",
) -> np.ndarray:
    """
    Magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    # return scipy.signal.lfilter([1], [1, float(-gamma)], rewards[::-1], axis=0)[
    #     ::-1
    # ].astype(dtype)

    return scipy.signal.lfilter(
        [1],
        [1, float(-gamma)],
        rewards[::-1],
        axis=0,
    )[::-1].astype(dtype)


def rewards_to_go(
    rews: list | np.ndarray,
    dtype: str | np.dtype = "float32",
) -> np.ndarray:
    """Computes rewards-to-go

    Args:
        rews (Union[List, np.ndarray]): Array-like of rewards
        dtype (str, optional): numpy dtype. Defaults to 'float32'.

    Returns:
        np.ndarray: rewards-to-go
    """
    n = len(rews)
    rtgs = np.zeros_like(rews, dtype=dtype)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i + 1] if i + 1 < n else 0)
    return rtgs


def standardize(x: list | np.ndarray) -> np.ndarray:
    """Computes z-score normalization of input array

    Args:
        x (Union[List, np.ndarray]): Array-like

    Returns:
        np.ndarray: Standardized array
    """
    # This func does not support `axis`
    x = np.asarray(x)
    return (x - np.mean(x)) / np.std(x)


# Adapted from: https://github.com/NM512/dreamerv3-torch/blob/main/tools.py#L950
def set_seed_everywhere(seed: int, deterministic: bool = False) -> None:
    # NOTE: This might be deprecated
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


def seconds_to_hms(seconds: float) -> str:
    td = datetime.timedelta(seconds=seconds)
    return str(td)  # format with str() to HH:MM:SS


def setup_logger(name: str = __name__, level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    fmt = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    rich_handler = RichHandler(rich_tracebacks=True)
    rich_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(rich_handler)

    # Prevent duplicated log messages by stopping propagation to the root logger
    logger.propagate = False

    # Handler for global exception
    def log_unhandled_exceptions(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = log_unhandled_exceptions

    return logger


def add_file_handler(logger: logging.Logger, filename: str) -> None:
    """Add a file handler to an existing logger, ensuring no duplicate handlers are added."""
    if any(isinstance(h, logging.FileHandler) and h.baseFilename == str(filename) for h in logger.handlers):
        # File handler already exists.
        return

    file_handler = logging.FileHandler(filename, encoding="utf-8")
    fmt = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    formatter = logging.Formatter(fmt)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


@dataclass
class EvaluationResults:
    episode_rewards: list[float]
    episode_lengths: list[float]
    episode_reward_mean: float
    episode_reward_std: float
    episode_length_mean: float
    frames: list[list]
    best_episode_reward: float
    best_episode_frames: list
    best_episode_idx: int


def evaluate_policy(
    policy: Callable,
    env: Env = None,
    make_env_fn: Callable = None,
    n_eval_episodes: int = 5,
    seed: int | None = None,
    timeout: int = 5000,
    verbose: bool = True,
    env_kwargs: dict | None = None,
) -> EvaluationResults:
    """
    Evaluates performance of given agent by letting it play a few episodes of
    the game and then calculates the average reward it gets.

    Args:
        policy (Callable): A function that takes observation as an input and
            outputs an action
        make_env_fn (Callable): A function that gives the environment to execute
            the policy on. This must accept `env_id` param.
        env (gym.Env): gym.Env object to run agent on. one of make_env_fn and env
            must be passed in.
        seed (int): Seed for env.
        timeout (int): Episode timeout in game steps
        verbose (bool): If True, progress bar will be shown. Defaults to True.
        env_kwargs (dict): Keyward arguments to be sent to make_env.

    Returns:
        tuple[float, ...]: Average reward across episodes, average episode
            length, and best frames if specified

    Example:
    # Example 1
    >>> eval_results = evaluate_policy(policy, env=gym.make('CartPole-v0'), n_eval_episodes=10)
    >>> print(eval_results.episode_reward_mean)
    >>> print(eval_results.episode_reward_std)

    # Example 2
    >>> eval_results = evaluate_policy(policy, make_env_fn=gym.make, env_kwargs=dict(id='CartPole-v0', render_mode='rgb_array'))
    """

    assert callable(policy), "`policy` must be callable"

    if env_kwargs is None:
        env_kwargs = {}

    if env is None and make_env_fn is None:
        raise ValueError("Either env or make_env_fn param must be passed in.")

    if make_env_fn is not None:
        assert callable(make_env_fn), "`make_env_fn` must be callable"
        env = make_env_fn(**env_kwargs)

    if seed is not None:
        env.action_space.seed(seed)

    episode_rewards = []
    episode_lengths = []
    all_episode_frames = []
    current_best_reward = float("-inf")
    best_episode_idx = None

    for episode_idx in tqdm(range(n_eval_episodes), disable=not verbose, desc="Evaluation"):
        episode_frames = []
        obs, _ = env.reset()
        terminated, truncated = False, False
        episode_reward = 0.0
        step = 0

        while True:
            action = policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            step += 1

            episode_frames.append(env.render())

            if terminated or truncated:
                break

            if step >= timeout:
                # handle as episode is done
                truncated = True

        # end-of-episode handling
        if episode_reward > current_best_reward:
            current_best_reward = episode_reward
            best_episode_idx = episode_idx
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        all_episode_frames.append(episode_frames)

    env.close()

    episode_reward_mean = np.mean(episode_rewards)
    episode_reward_std = np.std(episode_rewards)
    episode_length_mean = np.mean(episode_lengths)

    return EvaluationResults(
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        episode_reward_mean=episode_reward_mean,
        episode_reward_std=episode_reward_std,
        episode_length_mean=episode_length_mean,
        frames=all_episode_frames,
        best_episode_reward=current_best_reward,
        best_episode_frames=all_episode_frames[best_episode_idx],
        best_episode_idx=best_episode_idx,
    )


def evaluate_policy_parallel(
    make_env: Callable,
    policy: Callable,
    n_eval_episodes: int,
    n_eval_envs: int = 1,
    seed: int = 0,
    all_render: bool = True,
    return_frames: bool = False,
    return_in_dict: bool = False,
) -> dict[str, np.floating[Any]] | list[np.floating[Any]]:
    n_envs = n_eval_envs
    episode_targets = [(n_eval_episodes + i) // n_envs for i in range(n_envs)]

    render = [False] * n_envs
    render[-1] = return_frames

    _out = Parallel(n_jobs=n_envs)(
        [
            delayed(evaluate_policy)(
                make_env,
                policy,
                seed=seed + n,
                n_eval_episodes=episode_targets[n],
                return_frames=render[n],  # becomes True only at very last iteration if return_frames is True
                return_raw_metrics=True,
                render=all_render,
            )
            for n in range(n_envs)
        ]
    )

    ep_rews = [ep_rew for o in _out for ep_rew in o["ep_rews"]]
    ep_lens = [ep_len for o in _out for ep_len in o["ep_lens"]]
    out = dict(
        ep_rew_mean=np.mean(ep_rews),
        ep_rew_std=np.std(ep_rews),
        ep_len_mean=np.mean(ep_lens),
    )
    if return_frames:
        out["frames"] = _out[-1]["frames"]
    if return_in_dict:
        return out
    return list(out.values())


def save_state_dict(model: nn.Module, path: Path) -> None:
    """Save model's state dict at a path."""
    device = next(model.parameters()).device
    model.cpu()
    torch.save(model.state_dict(), path)
    model.to(device)


def smoothen(old: float | None, val: float, alpha: float = 0.95) -> float:
    """Smoothen new value incorporating into one older value."""
    if old is None:
        return val
    return alpha * old + (1 - alpha) * val


def save_config(config, filename: str) -> None:
    config_json = convert_json(config)
    kwargs = dict(
        separators=(",", ":"),
        indent=4,
        sort_keys=True,
        default=lambda o: "<not serializable>",
    )
    output = json.dumps(config_json, **kwargs)
    with open(filename, "w") as out:
        out.write(output)


# copied from : https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/utils/serialization_utils.py
def convert_json(obj: object) -> object:  # noqa: PLR0911
    """Convert obj to a version which can be serialized with JSON."""
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) for k, v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj, "__name__") and "lambda" not in obj.__name__:
            return convert_json(obj.__name__)

        elif hasattr(obj, "__dict__") and obj.__dict__:
            obj_dict = {convert_json(k): convert_json(v) for k, v in obj.__dict__.items()}
            return {str(obj): obj_dict}

        return str(obj)


def is_json_serializable(v: object) -> bool:
    try:
        json.dumps(v)
        return True
    except Exception:
        return False


def orthgonal_initialization(module: nn.Module, gain: int = 1) -> None:
    """
    Orthgonal initialization

    Reference: https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.orthogonal_
    """
    if isinstance(module, nn.Linear | nn.Conv2d):
        torch.nn.init.orthogonal_(module.weight, gain=gain)
        if hasattr(module, "bias"):
            module.bias.data.fill_(0.0)


class RunningMeanTrackerForSingleMetric:
    """
    A class for tracking running mean value over a sliding window.

    Args:
        writer (SummaryWriter): An instance of TensorBoard's SummaryWriter to log the data.
        window_size (int): The number of recent values used to calculate the running mean. Default is 100.
        bound (float): A threshold value for filtering the running mean. If `upper_is_better` is True, the running mean is only returned if it is greater than or equal to this bound. If False, the running mean is returned if it is less than or equal to this bound. Default is infinity.
        upper_is_better (bool): A flag to determine if higher running mean values are considered better. Default is True.
    """

    def __init__(
        self,
        tag: str,
        writer: SummaryWriter,
        bound: float = float("inf"),
        window_size: int = 100,
        upper_is_better: bool = True,
        write_immediate_value: bool = True,
    ) -> None:
        self.tag = tag
        self.writer = writer
        self.bound = bound
        self.window_size = window_size
        self.upper_is_better = upper_is_better
        self.write_immediate_value = write_immediate_value
        self.xs = deque(maxlen=window_size)

    def as_suffix(self, precision: int = 3, latest: bool = False, tag: str | None = None) -> str:
        tag = self.tag.split("/")[-1] if tag is None else tag
        return f"{tag}_{self.window_size}={self.get(latest):.{precision}f}"

    def add(self, x: float, global_step: int) -> None:
        """
        Adds a new value to the tracker and logs both the running mean over the
        defined window size and the raw value.

        Args:
            tag (str): The tag associated with the scalar values for logging.
            x (float): The new scalar value to be added to the tracker.
            global_step (int): Global step number for logging the value.

        Returns:
            float or None: The running mean value if it meets the bound condition, otherwise None.
        """
        self.xs.append(x)
        running_mean = np.mean(self.xs)
        self.writer.add_scalar(f"{self.tag}_{self.window_size}", running_mean, global_step)
        if self.write_immediate_value:
            self.writer.add_scalar(self.tag, x, global_step)
        if self.upper_is_better:
            return running_mean if running_mean >= self.bound else None
        return running_mean if running_mean <= self.bound else None

    def get(self, latest: bool = False) -> float:
        if latest:
            return self.xs[-1]
        return np.mean(self.xs).item()


class BatchedSummaryWriter:
    """
    A class for logging batched scalar values to TensorBoard at specified intervals.

    This class accumulates scalar values and logs their average to TensorBoard
    after a fixed number of values (batch size) have been collected.

    Args:
        writer (SummaryWriter): An instance of TensorBoard's SummaryWriter to log the data.
        batch_size (int): The number of scalar values to accumulate before logging the average.
    """

    def __init__(self, writer: SummaryWriter, batch_size: int) -> None:
        self.writer = writer
        self.batch_size = batch_size
        self.batches = defaultdict(list)

    @staticmethod
    def _to_float(x: float | list | np.ndarray | Tensor) -> float:
        if torch.is_tensor(x):
            return x.detach().cpu().float().mean().item()
        elif isinstance(x, np.ndarray | list):
            return np.mean(x).item()
        else:
            return float(x)

    def add(self, tag: str, x, global_step: int) -> None:
        """
        Adds a scalar value to the specified tag and logs the average after the batch size is reached.

        Args:
            tag (str): The tag associated with the scalar value.
            value (float): The scalar value to be logged.
            global_step (int): Global step number for logging the value.
        """
        data = self.batches[tag]
        data.append(self._to_float(x))
        if len(data) >= self.batch_size:
            self.writer.add_scalar(tag, np.mean(data), global_step)
            data.clear()
