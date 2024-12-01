from gymnasium import register

# Equivalent settings to the legacy "CartPole-v0", which is deprecated in gymnasium.
register(
    id="CartPole-v3",
    entry_point="gymnasium.envs.classic_control:CartPoleEnv",
    max_episode_steps=200,
    reward_threshold=195.0,
)
