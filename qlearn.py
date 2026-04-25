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
EPOCHS = 1000


'''
def movement(action, state, ncols):
    """
    Parameters: 
        action : int representing which way to move
        state : int representing the current location by doing current_row * ncols + current_col, counting from 0)

    Return:
        Resulting state?
    """
    current_row = state // ncols # integer division dividing the state int by the number of columns
    current_col = state % ncols # state int modulo number of columns
    # move left when not on left edge 
    if action == 0 and current_col > 0:
        current_col = current_col - 1
    # move down when not on bottom edge
    elif action == 1 and current_row < 3:
        current_row = current_row + 1
    # move right when not on right edge
    elif action == 2 and current_col < 3:
        current_col = current_col + 1
    # move up when not on upper edge
    elif action == 3 and current_row > 0:
        current_row = current_row - 1
    # invalid input
    else:
        print("Invalid input for movement")
    print("Your new state location is: ", current_row*ncols + current_col)
    return (current_row*ncols + current_col)

# test that movement returns
movement(1,0,4)
'''

# learn through iterations (Q-Learning)
def learn(map):
    n_rows = map.unwrapped.nrow
    n_cols = map.unwrapped.ncol
    n_states = map.observation_space.n
    n_actions = map.action_space.n

    # make Q-table and initialize all values to 0
    Q = [[0 for _ in range(n_actions)] for _ in range(n_states)]
    print("the Q table at the start of learning is: ", Q)

    for episode in range (EPOCHS):
        state,_ = map.reset()
        max_change = 0
        THRESHOLD = 1e-4
        stable_count = 0
        STABLE_REQUIRED = 10
        done = False
        while not done:
            # epsilon greedy
            if random.random() < EXPLORATION_PROB:
                action = random.randint(0,n_actions-1)
            else:
                action = Q[state].index(max(Q[state]))
            #next_state = movement(action,current_state,n_cols)
            next_state, reward, terminated, truncated, info = map.step(action)

            done = terminated or truncated
            
            old_value = Q[state][action]
            # Q-learning update
            Q[state][action] = Q[state][action] + LEARNING_RATE * (
                reward + DISCOUNT_FACTOR * max(Q[next_state]) - Q[state][action]
            )
            change = abs(Q[state][action] - old_value)

            state = next_state

            max_change = max(max_change, change)
        if max_change < THRESHOLD:
            stable_count += 1
        else:
            stable_count = 0
        if stable_count >= STABLE_REQUIRED:
            break

    print(f"Converged at episode {episode}")      
    print("the Q table at the end of learning is: ", Q)
    return Q

# default 4x4 slippery map
learn(default_env)

# randomly generated 4x4 slippery map
learn(stochastoc_4x4_env)

# randomly generated 8x8 slippery map
#learn(stochastoc_8x8_env)

# randomly generated 4x4 non-slippery map
learn(deterministic_4x4_env)

# randomly generated 8x8 non-slippery map
#learn(deterministic_8x8_env)

