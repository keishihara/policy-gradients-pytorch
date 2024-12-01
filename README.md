# Policy Gradients in PyTorch

A simple collection of policy gradient algorithm implementations in PyTorch. This repository is designed for anyone looking to get hands-on experience with basic RL algorithms.

Experiments are conducted on the following environments:

- [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
- [PongNoFrameskip-v4](https://ale.farama.org/environments/pong/)

If you'd like to try other Gym environments, you'll need to define the corresponding hyperparameters in `hparams/` and adjust the scripts to handle the new observation and action spaces, and the network architecture.

## Getting Started

The code is developed and tested with Python 3.11 and CUDA 12.1.
Make sure you have them installed, and then follow the steps below to setup.

```bash
git clone https://github.com/keishihara/policy-gradients-pytorch.git
cd policy-gradients-pytorch

# Install dependencies via `uv`
uv sync

# Or install via `pip`
pip install -e .
```

## Policy Gradient Algorithms

This repo currently contains the following classic policy gradient algorithms. All hyperparameters are stored in `hparams/`, and logs are automatically saved to `logs/`.

### 1. REINFORCE

The simplest policy gradient algorithm, which optimizes the policy by following the gradient of the expected cumulative reward.

```bash
# CartPole-v1
python algos/reinforce/reinforce.py --cuda
```

### 2. Vanilla Policy Gradient (VPG)

An upgraded version of REINFORCE that incorporates a baseline to reduce variance in the gradient estimate.

```bash
# Train a policy on CartPole-v1
python algos/vpg/vpg_on_cc.py --cuda
```

A training script for `PongNoFrameskip-v4` is also provided, although it may not converge.

```bash
# Train a policy on PongNoFrameskip-v4
python algos/vpg/vpg_on_atari.py --cuda
```

### 3. Advantage Actor-Critic (A2C)

A more advanced algorithm that introduces a critic network to estimate the value function (state-dependent baseline), enabling it to solve Atari games with pixel observations.

```bash
# Train a policy on PongNoFrameskip-v4
python algos/a2c/a2c.py --cuda
```

## Visulization

Track training dynamics and performance metrics using TensorBoard:

```bash
tensorboard --logdir logs
```
