import random

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# generate default 4x4 map
default_env = gym.make("FrozenLake-v1", render_mode="None",
    map_name="4x4",
    is_slippery=True,
    success_rate=0.75,
    reward_schedule=(10, -10, 0)
    )

# generate a random 4x4 slippery map with p=3/4 and 10,-10,0 rewards
stochastoc_4x4_env = gym.make("FrozenLake-v1", render_mode="None",
    desc=generate_random_map(size=4),
    is_slippery=True,
    success_rate=0.75, 
    reward_schedule=(10, -10, 0)
)

# generate a random 8x8 slippery map with p=3/4 and 10,-10,0 rewards
stochastoc_8x8_env = gym.make("FrozenLake-v1", render_mode="None",
    desc=generate_random_map(size=8),
    is_slippery=True,
    success_rate=0.75, 
    reward_schedule=(10, -10, 0)
)

# generate a random 4x4 non-slippery map with p=3/4 and 10,-10,0 rewards
deterministic_4x4_env = gym.make("FrozenLake-v1", render_mode="None",
    desc=generate_random_map(size=4),
    is_slippery=False,
    success_rate=0.75, 
    reward_schedule=(10, -10, 0)
)

# generate a random 8x8 non-slippery map with p=3/4 and 10,-10,0 rewards
deterministic_8x8_env = gym.make("FrozenLake-v1", render_mode="None",
    desc=generate_random_map(size=8),
    is_slippery=False,
    success_rate=0.75, 
    reward_schedule=(10, -10, 0)
)

LEARNING_RATE = 0.6 #alpha
DISCOUNT_FACTOR = 0.95
EPOCHS = 5000


# learn through iterations (Q-Learning)
def learn(map):
    n_states = map.observation_space.n
    n_actions = map.action_space.n

    # make Q-table and initialize all values to 0
    Q = [[0 for _ in range(n_actions)] for _ in range(n_states)]
    epsilon = 1 #exploration rate 
    epsilon_min = 0.1
    epsilon_decay = 0.999
    stable_count = 0
    STABLE_REQUIRED = 10
    successes = 0
    converged = False
    for episode in range (EPOCHS):
        episode_max_change = 0
        state,_ = map.reset()
        THRESHOLD = 1e-3
        done = False
        while not done:
            # epsilon greedy
            if random.random() < epsilon:
                action = random.randint(0,n_actions-1)
            else:
                best_actions = [i for i, q in enumerate(Q[state]) if q == max(Q[state])]
                action = random.choice(best_actions)
            next_state, reward, terminated, truncated, info = map.step(action)

            done = terminated or truncated
            
            old_value = Q[state][action]
            # Q-learning update
            target = reward if terminated else reward + DISCOUNT_FACTOR * max(Q[next_state])
            
            Q[state][action] += LEARNING_RATE * (target - Q[state][action])

            change = abs(Q[state][action] - old_value)
            episode_max_change = max(episode_max_change, change)

            state = next_state
            
            if terminated and reward > 0:
                successes += 1

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        if episode_max_change < THRESHOLD:
            stable_count += 1
        else:
            stable_count = 0
    
        if stable_count >= STABLE_REQUIRED and episode>500 and successes>300:
            converged = True
            print(f"Converged at episode {episode}")
            break
        
    if converged==False:
        print(f"Did not converge within {EPOCHS} episodes")
    print("Success rate:", successes / (episode + 1))
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


'''
# Tests for part b of the assignment:
# randomly generated 4x4 slippery map
print("stochastoc_4x4_env map")
learn(stochastoc_4x4_env)

# randomly generated 8x8 slippery map
print("stochastoc_8x8_env map")
learn(stochastoc_8x8_env)

# randomly generated 4x4 slippery map
print("stochastoc_4x4_env map")
learn(stochastoc_4x4_env)

# randomly generated 8x8 slippery map
print("stochastoc_8x8_env map")
learn(stochastoc_8x8_env)

# randomly generated 4x4 slippery map
print("stochastoc_4x4_env map")
learn(stochastoc_4x4_env)

# randomly generated 8x8 slippery map
print("stochastoc_8x8_env map")
learn(stochastoc_8x8_env)

# randomly generated 4x4 slippery map
print("stochastoc_4x4_env map")
learn(stochastoc_4x4_env)

# randomly generated 8x8 slippery map
print("stochastoc_8x8_env map")
learn(stochastoc_8x8_env)

# randomly generated 4x4 slippery map
print("stochastoc_4x4_env map")
learn(stochastoc_4x4_env)

# randomly generated 8x8 slippery map
print("stochastoc_8x8_env map")
learn(stochastoc_8x8_env)

# randomly generated 4x4 slippery map
print("stochastoc_4x4_env map")
learn(stochastoc_4x4_env)

# randomly generated 8x8 slippery map
print("stochastoc_8x8_env map")
learn(stochastoc_8x8_env)
'''

'''
# Tests for part b of the assignment:
# randomly generated 4x4 non-slippery map
print("deterministic_4x4_env map")
learn(deterministic_4x4_env)

# randomly generated 8x8 non-slippery map
print("deterministic_8x8_env map")
learn(deterministic_8x8_env)

# randomly generated 4x4 non-slippery map
print("deterministic_4x4_env map")
learn(deterministic_4x4_env)

# randomly generated 8x8 non-slippery map
print("deterministic_8x8_env map")
learn(deterministic_8x8_env)

# randomly generated 4x4 non-slippery map
print("deterministic_4x4_env map")
learn(deterministic_4x4_env)

# randomly generated 8x8 non-slippery map
print("deterministic_8x8_env map")
learn(deterministic_8x8_env)

# randomly generated 4x4 non-slippery map
print("deterministic_4x4_env map")
learn(deterministic_4x4_env)

# randomly generated 8x8 non-slippery map
print("deterministic_8x8_env map")
learn(deterministic_8x8_env)

# randomly generated 4x4 non-slippery map
print("deterministic_4x4_env map")
learn(deterministic_4x4_env)

# randomly generated 8x8 non-slippery map
print("deterministic_8x8_env map")
learn(deterministic_8x8_env)
'''
