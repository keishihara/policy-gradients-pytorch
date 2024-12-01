from collections import deque
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Generator, TypeAlias

from box import Box
from gymnasium import Env
from numpy import ndarray

Observation: TypeAlias = ndarray
Action: TypeAlias = int


@dataclass(frozen=True)
class Transition:
    obs: Observation
    action: Action
    reward: int
    terminated: bool
    truncated: bool

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@dataclass(frozen=True)
class TransitionFirstLast(Transition):
    next_obs: Observation

    @property
    def done(self) -> bool:
        return self.next_obs is None


# adapted from: https://github.com/Shmuma/ptan/blob/master/ptan/experience.py#L27
class RolloutSimulator:
    Item = tuple[Transition, ...]

    def __init__(
        self,
        env: Env | Collection[Env],
        agent: Callable,
        n_steps: int = 1,
        steps_delta: int = 1,
        env_seed: int | None = None,
    ) -> None:
        assert n_steps >= 1
        assert steps_delta >= 1
        if isinstance(env, (list, tuple)):
            self.pool = env
            # Do the check for the multiple copies passed
            ids = set(id(e) for e in env)
            if len(ids) < len(env):
                raise ValueError(
                    "You passed single environment instance multiple times"
                )
        else:
            self.pool = [env]

        self.agent = agent
        self.n_steps = n_steps
        self.steps_delta = steps_delta
        self.total_rewards = []
        self.total_steps = []
        self.env_seed = env_seed

    def __repr__(self):
        return f"RolloutSimulator(env={self.pool[0]}, pool_size={len(self.pool)}, n_steps={self.n_steps}, steps_delta={self.steps_delta}, env_seed={self.env_seed})"

    def __iter__(self) -> Generator[Item, None, None]:
        states, histories, cur_rewards, cur_steps = [], [], [], []
        for i, env in enumerate(self.pool):
            if self.env_seed is not None:
                obs, _ = env.reset(seed=self.env_seed + i)
            else:
                obs, _ = env.reset()
            states.append(obs)
            histories.append(deque(maxlen=self.n_steps))
            cur_rewards.append(0.0)
            cur_steps.append(0)

        iter_idx = 0
        while True:
            actions = self.agent(states)
            if isinstance(actions, (int, float)):
                actions = [actions]

            for idx, env in enumerate(self.pool):
                state = states[idx]
                action = actions[idx]
                history: deque = histories[idx]
                next_state, r, terminated, truncated, _ = env.step(action)
                cur_rewards[idx] += r
                cur_steps[idx] += 1
                history.append(Transition(state, action, r, terminated, truncated))
                if len(history) == self.n_steps and iter_idx % self.steps_delta == 0:
                    yield tuple(history)
                states[idx] = next_state
                if terminated or truncated:
                    # Generate tail of history
                    if 0 < len(history) < self.n_steps:
                        yield tuple(history)
                    while len(history) > 1:
                        history.popleft()
                        yield tuple(history)
                    self.total_rewards.append(cur_rewards[idx])
                    self.total_steps.append(cur_steps[idx])
                    cur_rewards[idx] = 0.0
                    cur_steps[idx] = 0
                    if self.env_seed is not None:
                        states[idx], _ = env.reset(seed=self.env_seed + idx)
                    else:
                        states[idx], _ = env.reset()
                    history.clear()
            iter_idx += 1

    def pop_total_rewards(self) -> Box:
        r = self.total_rewards
        if r:
            self.total_rewards = []
            self.total_steps = []
        return Box(is_terminal=len(r) > 0, total_rewards=r)


class RolloutSimulatorFirstLast(RolloutSimulator):
    def __init__(
        self,
        env: Env | Collection[Env],
        agent: Callable,
        gamma: float,
        n_steps: int = 1,
        steps_delta: int = 1,
        env_seed: int | None = None,
    ) -> None:
        super(RolloutSimulatorFirstLast, self).__init__(
            env,
            agent,
            n_steps=n_steps + 1,  # NOTE: This seems to be important
            steps_delta=steps_delta,
            env_seed=env_seed,
        )
        self.gamma = gamma
        self.steps = n_steps

    def __repr__(self):
        return (
            f"RolloutSimulatorFirstLast(env={self.pool[0]}, pool_size={len(self.pool)}, "
            f"gamma={self.gamma}, steps={self.steps} n_steps={self.n_steps}, "
            f"steps_delta={self.steps_delta}, env_seed={self.env_seed})"
        )

    def __iter__(self) -> Generator[TransitionFirstLast, None, None]:
        for trans in super().__iter__():
            if trans[-1].done and len(trans) <= self.steps:
                last_state = None
                elems = trans
            else:
                last_state = trans[-1].obs
                elems = trans[:-1]
            total_reward = 0.0
            for e in reversed(elems):
                total_reward *= self.gamma
                total_reward += e.reward
            yield TransitionFirstLast(
                trans[0].obs,
                trans[0].action,
                total_reward,
                trans[0].terminated,
                trans[0].truncated,
                last_state,
            )
