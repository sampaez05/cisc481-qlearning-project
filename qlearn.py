import random

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# generate default 4x4 map
default_env = gym.make("FrozenLake-v1", render_mode="None",
    map_name="4x4",
    is_slippery=True,
    success_rate=0.8,
    reward_schedule=(10, -10, 0)
    )

# generate a random 4x4 slippery map with p=3/4 and 10,-10,0 rewards
stochastoc_4x4_env = gym.make("FrozenLake-v1", render_mode="None",
    desc=generate_random_map(size=4),
    is_slippery=True,
    success_rate=0.75, # should this be .8 or .75?
    reward_schedule=(10, -10, 0)
)

# generate a random 8x8 slippery map with p=3/4 and 10,-10,0 rewards
stochastoc_8x8_env = gym.make("FrozenLake-v1", render_mode="None",
    desc=generate_random_map(size=8),
    is_slippery=True,
    success_rate=0.75, # should this be .8 or .75?
    reward_schedule=(10, -10, 0)
)

# generate a random 4x4 non-slippery map with p=3/4 and 10,-10,0 rewards
deterministic_4x4_env = gym.make("FrozenLake-v1", render_mode="None",
    desc=generate_random_map(size=4),
    is_slippery=False,
    success_rate=0.75, # should this be .8 or .75?
    reward_schedule=(10, -10, 0)
)

# generate a random 8x8 non-slippery map with p=3/4 and 10,-10,0 rewards
deterministic_8x8_env = gym.make("FrozenLake-v1", render_mode="None",
    desc=generate_random_map(size=8),
    is_slippery=False,
    success_rate=0.75, # should this be .8 or .75?
    reward_schedule=(10, -10, 0)
)

LEARNING_RATE = 0.8
DISCOUNT_FACTOR = 0.95
EXPLORATION_PROB = 0.4 # epsilon value
EXPLOITATION_PROB = 0.6 # aka 1-epsilon 
EPOCHS = 3000

# learn through iterations (Q-Learning)
def learn(map):
    n_rows = map.unwrapped.nrow
    n_cols = map.unwrapped.ncol
    n_states = map.observation_space.n
    n_actions = map.action_space.n

    # make Q-table and initialize all values to 0
    Q = [[0 for _ in range(n_actions)] for _ in range(n_states)]
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    stable_count = 0
    STABLE_REQUIRED = 10
    global_max_change = 0
    max_change = 0
    successes = 0
    for episode in range (EPOCHS):
        episode_max_change = 0
        episode_reward = 0
        state,_ = map.reset()
        global_max_change = max(global_max_change, max_change)
        THRESHOLD = 1e-4
        done = False
        while not done:
            # epsilon greedy
            if random.random() < epsilon:
                action = random.randint(0,n_actions-1)
            else:
                #action = Q[state].index(max(Q[state]))
                best_actions = [i for i, q in enumerate(Q[state]) if q == max(Q[state])]
                action = random.choice(best_actions)
            next_state, reward, terminated, truncated, info = map.step(action)

            done = terminated or truncated
            
            old_value = Q[state][action]
            # Q-learning update
            target = reward if terminated else reward + DISCOUNT_FACTOR * max(Q[next_state])
            
            episode_reward += reward
            Q[state][action] += LEARNING_RATE * (target - Q[state][action])
            change = abs(Q[state][action] - old_value)

            state = next_state

            max_change = max(max_change, change)
        if terminated and reward > 0:
            successes += 1

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if max_change < THRESHOLD:
            stable_count += 1
        else:
            stable_count = 0
        if episode > 500 and global_max_change < THRESHOLD:
            print(f"Converged at episode {episode}")   
            break
    print("Success rate:", successes / EPOCHS)
    print("the Q table at the end of learning is: ", Q, "\n")
    return Q

# default 4x4 slippery map
print("default map")
learn(default_env)

# randomly generated 4x4 slippery map
print("stochastoc_4x4_env map")
learn(stochastoc_4x4_env)

# randomly generated 8x8 slippery map
print("stochastoc_8x8_env map")
learn(stochastoc_8x8_env)

# randomly generated 4x4 non-slippery map
print("deterministic_4x4_env map")
learn(deterministic_4x4_env)

# randomly generated 8x8 non-slippery map
print("deterministic_8x8_env map")
learn(deterministic_8x8_env)

