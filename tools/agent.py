from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import torch
import torch.nn.functional as F
from numpy import ndarray
from torch import Tensor

Observations: TypeAlias = list[ndarray] | ndarray
Actions: TypeAlias = int | ndarray
Model: TypeAlias = Callable[[Observations], Tensor]
Preprocessor: TypeAlias = Callable[[Observations], Tensor]


def default_preprocessor(obs: Observations) -> Tensor:
    """Convert list of observations into the form suitable for model"""
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs[None, ...]
    return torch.as_tensor(obs).float()


def preprocessor_atari(obs: Observations) -> Tensor:
    """Return a normalized float tensor"""
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim == 3:
        obs = obs[None, ...]
    return torch.as_tensor(obs).float() / 255


class BaseAgent(ABC):
    @abstractmethod
    def __call__(
        self,
        obs: Observations,
        deterministic: bool = False,
    ) -> Actions: ...


class DiscreteActionAgent(BaseAgent):
    def __init__(
        self,
        model: Model,
        preprocessor: Preprocessor = default_preprocessor,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.preprocessor = preprocessor
        assert isinstance(preprocessor, Callable)

    def __call__(self, obs: Observations, deterministic: bool = False) -> Actions:
        if deterministic:
            return self.deterministic_policy(obs)
        return self.stochastic_policy(obs)

    @torch.inference_mode()
    def stochastic_policy(self, obs: Observations) -> Actions:
        obs = self.preprocessor(obs)
        assert torch.is_tensor(obs), f"obs must be a tensor object, got {type(obs)}"
        obs = obs.to(self.device)

        logits = self.model(obs)

        probs = F.softmax(logits, dim=1)
        actions = torch.multinomial(probs, num_samples=1)  # (B, 1), ndim=2
        actions = actions.squeeze(dim=1)  # (B,), ndim=1
        actions = actions.cpu().numpy()  # no need to call .detach()
        return actions.item() if len(actions) == actions.ndim == 1 else actions

    @torch.inference_mode()
    def deterministic_policy(self, obs: Observations) -> Actions:
        obs = self.preprocessor(obs)
        assert torch.is_tensor(obs), f"obs must be a tensor object, got {type(obs)}"
        obs = obs.to(self.device)

        logits = self.model(obs)

        probs = F.softmax(logits, dim=1)  # (B, A), ndim=2
        actions = probs.argmax(dim=1)  # (B,), ndim=1
        actions = actions.cpu().numpy()  # no need to call .detach()
        return actions.item() if len(actions) == actions.ndim == 1 else actions
