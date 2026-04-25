import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# generate a random 4x4 map
env = gym.make("FrozenLake-v1", render_mode="human",
    desc=generate_random_map(size=4),
    is_slippery=True,
    success_rate=0.8,
    reward_schedule=(10, -10, 0)
)