import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# generate default 4x4 map
default_env = gym.make("FrozenLake-v1", render_mode="human",
    map_name="4x4",
    is_slippery=True,
    success_rate=0.8,
    reward_schedule=(10, -10, 0)
    )

# generate a random 4x4 slippery map with p=3/4 and 10,-10,0 rewards
stochastoc_4x4_env = gym.make("FrozenLake-v1", render_mode="human",
    desc=generate_random_map(size=4),
    is_slippery=True,
    success_rate=0.75, # should this be .8 or .75?
    reward_schedule=(10, -10, 0)
)

print("stochastoc_4x4_env is: ", stochastoc_4x4_env)

LEARNING_RATE = 0.8
DISCOUNT_FACTOR = 0.95
EXPLORATION_PROB = 0.2 # epsilon value
EXPLOITATION_PROB = 0.8 # aka 1-epsilon 
EPOCHS = 1000

def movement(action, state):
    """
    Parameters: 
        action : int representing which way to move
        state : int representing the current location by doing current_row * ncols + current_col, counting from 0)

    Return:
        Resulting state?
    """
    current_row = state // 4 # integer division dividing the state int by the number of columns
    current_col = state % 4 # state int modulo number of columns
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
        current_row = current_row + 1
    # invalid input
    else:
        print("Invalid input for movement")

