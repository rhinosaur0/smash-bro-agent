# rewards.py previously known as environment.py
# requires hyperparam.py
# Reward functions for the WarehouseBrawl environment

from enum import Enum
from typing import List, Tuple

import numpy as np
from hyperparam import RewardMode, DamageRewardMode
from hyperparam import DEFAULT_PARAMS as PARAMS
from collections import defaultdict
import functools
import os

# replace it with your own env
# from WarehouseEnv import WarehouseBrawl

# env = WarehouseBrawl()

# # Environment variables available
# # Agents
# env.objects["player"]
# env.objects["opponent"]

# # Agent Position
# env.objects["player"].body.position.x   # X position during frame
# env.objects["player"].body.position.y   # Y position during frame

# env.objects["player"].body.position.x_change  # Change in x direction position between frames
# env.objects["player"].body.position.y_change  # Change in y direction position between frames

# env.objects["player"].body.velocity.x # X velocity of agent
# env.objects["player"].body.velocity.y # Y velocity of agent

# # Agent Charachteristics
# env.objects["player"].DamageTakenTotal      # Integer value of total damage taken
# env.objects["player"].DamageTakenThisStock  # Integer value of damage taken this stock life
# env.objects["player"].DamageTakenThisFrame  # Integer value
# env.objects["player"].WeaponHeldThisFrame   # True or False

# # Time
# env.time_elapsed  # Time that has elapsed since start of game
# env.current_frame # Current frame number

# # Platforms
# env.objects['ground']
# env.objects['platform1']
# env.objects['platform2']

# Signals
# knockout_signal = Signal()
# knockout_signal.connect(knockout_reward)
# knockout_signal.emit(agent="player") # Triggered when an agent is knocked out

# win_signal = Signal()
# win_signal.connect(win_reward)
# win_signal.emit(agent="player") # Triggered when the player wins

# no idea

# File to store aggregate rewards


LOG_FILE = "/content/logs/reward_log.txt"
reward_totals: dict = defaultdict(float)


def set_log_file_path(path: str):
    global LOG_FILE
    LOG_FILE = path



# usage: in process @ reward_manager() call log_rewards() at the end


def track_reward(reward_func):
    """Decorator to track and log rewards."""

    @functools.wraps(reward_func)
    def wrapper(*args, **kwargs):
        reward = reward_func(*args, **kwargs)
        reward_totals[reward_func.__name__] += reward
        return reward

    return wrapper


@track_reward
def damage_dealt_reward(
        env: "WarehouseBrawl",
        damage_reward_scale: float = PARAMS.DAMAGE_REWARD_SCALE,
        mode: DamageRewardMode = PARAMS.DAMAGE_REWARD_MODE
) -> float:
    """
    Computes the reward based on damage dealt to the opponent.

    Modes:
    - ASYMMETRIC_OFFENSIVE (0): Reward is based only on damage dealt to the opponent
    - SYMMETRIC (1): Reward is based on both dealing damage to the opponent and avoiding damage
    - ASYMMETRIC_DEFENSIVE (2): Reward is based only on avoiding damage

    Args:
        env (WarehouseBrawl): The game environment
        damage_reward_scale (float): The scale of the reward
        mode (DamageRewardMode): Reward mode, one of DamageRewardMode

    Returns:
        float: The computed reward.
    """
    opponent: "Player" = env.objects["opponent"]
    damage_dealt = opponent.damage_taken_this_frame

    if mode == DamageRewardMode.ASYMMETRIC_OFFENSIVE:
        reward = damage_dealt
    elif mode == DamageRewardMode.SYMMETRIC:
        reward = damage_dealt
    elif mode == DamageRewardMode.ASYMMETRIC_DEFENSIVE:
        reward = 0.0
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return reward * damage_reward_scale


@track_reward
def damage_taken_reward(
        env: "WarehouseBrawl",
        damage_reward_scale: float = PARAMS.DAMAGE_REWARD_SCALE,
        mode: DamageRewardMode = PARAMS.DAMAGE_REWARD_MODE,
) -> float:
    """
    Computes the reward based on damage taken by the player.

    Modes:
    - ASYMMETRIC_OFFENSIVE (0): Reward is based only on damage dealt to the opponent
    - SYMMETRIC (1): Reward is based on both dealing damage to the opponent and avoiding damage
    - ASYMMETRIC_DEFENSIVE (2): Reward is based only on avoiding damage

    Args:
        env (WarehouseBrawl): The game environment
        damage_reward_scale (float): The scale of the reward
        mode (DamageRewardMode): Reward mode, one of DamageRewardMode

    Returns:
        float: The computed reward.
    """
    player: "Player" = env.objects["player"]
    damage_taken = player.damage_taken_this_frame

    if mode == DamageRewardMode.ASYMMETRIC_OFFENSIVE:
        reward = 0.0
    elif mode == DamageRewardMode.SYMMETRIC:
        reward = -damage_taken
    elif mode == DamageRewardMode.ASYMMETRIC_DEFENSIVE:
        reward = -damage_taken
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return reward * damage_reward_scale


@track_reward
def danger_zone_reward(
        env: "WarehouseBrawl",
        zone_penalty: int = PARAMS.ZONE_PENALTY,
        zone_height: float = PARAMS.ZONE_HEIGHT,
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: "Player" = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    reward = zone_penalty if player.body.position.y >= zone_height else 0.0

    return reward


@track_reward
def move_to_opponent_reward(
        env: "WarehouseBrawl",
        reward_scale: float = PARAMS.MOVE_TO_OPPONENT_SCALE,
) -> float:
    """
    Computes the reward based on whether the agent is moving toward the opponent.
    The reward is calculated by taking the dot product of the agent's normalized velocity
    with the normalized direction vector toward the opponent.

    Args:
        env (WarehouseBrawl): The game environment

    Returns:
        float: The computed reward
    """
    # Getting agent and opponent from the enviornment
    player: "Player" = env.objects["player"]
    opponent: "Player" = env.objects["opponent"]

    # Extracting player velocity and position from environment
    player_position_dif = np.array([player.body.velocity.x * env.dt, player.body.velocity.y * env.dt])

    direction_to_opponent = np.array([opponent.body.position.x - player.body.position.x,
                                      opponent.body.position.y - player.body.position.y])

    # Prevent division by zero or extremely small values
    direc_to_opp_norm = np.linalg.norm(direction_to_opponent)
    player_pos_dif_norm = np.linalg.norm(player_position_dif)

    if direc_to_opp_norm < 1e-6 or player_pos_dif_norm < 1e-6:
        return 0.0

    # Compute the dot product of the normalized vectors to figure out how much
    # current movement (aka velocity) is in alignment with the direction they need to go in
    reward = np.dot(player_position_dif / direc_to_opp_norm, direction_to_opponent / direc_to_opp_norm)

    return reward * reward_scale


@track_reward
def edge_guard_reward(
        env: "WarehouseBrawl",
        success_value: float = PARAMS.EDGE_GUARD_SUCCESS,
        fail_value: float = PARAMS.EDGE_GUARD_FAIL,
        zone_width: float = PARAMS.ZONE_WIDTH,
) -> float:
    """
    Computes the reward given for every time step your agent is edge guarding the opponent.

    Args:
        env (WarehouseBrawl): The game environment
        success_value (float): Reward value for the player hitting first
        fail_value (float): Penalty for the opponent hitting first

    Returns:
        float: The computed reward.
    """
    reward = 0.0

    a: int | float = 2

    # condtion not met
    if (abs(env.objects["player"].body.position.x) < zone_width / 2 - a or  # player in the middle
            abs(env.objects["opponent"].body.position.x) < zone_width / 2 - a or  # opponent in the middle
            env.objects["player"].body.position.x * env.objects["opponent"].body.position.x < 0
    # both players are not on the same edge
    ):
        return reward

    # condition met
    if env.objects["opponent"].damage_taken_this_frame > 0:
        reward = success_value
    elif env.objects["player"].damage_taken_this_frame > 0:
        reward = fail_value

    return reward


@track_reward
def knockout_reward(
        env: "WarehouseBrawl",
        mode: RewardMode = PARAMS.REWARD_MODE,
        knockout_value_opponent: float = PARAMS.KNOCKOUT_VALUE_OPPONENT,
        knockout_value_player: float = PARAMS.KNOCKOUT_VALUE_PLAYER,
) -> float:
    """
    Computes the reward based on who won the match.

    Modes:
    - ASYMMETRIC_OPPONENT (0): Reward is based only on the opponent being knocked out
    - SYMMETRIC (1): Reward is based on both agents being knocked out
    - ASYMMETRIC_PLAYER (2): Reward is based only on your own plauyer being knocked out

    Args:
        env (WarehouseBrawl): The game environment
        mode (RewardMode): Reward mode, one of RewardMode
        knockout_value_opponent (float): Reward value for knocking out opponent
        knockout_value_player (float): Reward penalty for player being knocked out

    Returns:
        float: The computed reward.
    """
    reward = 0.0
    player_state = env.objects["player"].state
    opponent_state = env.objects["opponent"].state

    player_inKO = player_state == env.objects["player"].states["KO"]
    opponent_inKO = opponent_state == env.objects["opponent"].states["KO"]
    agent = "player" if player_inKO else "opponent" if opponent_inKO else None

    if agent is None:
        return reward

    # Mode logic to compute reward
    if mode == RewardMode.ASYMMETRIC_OPPONENT:
        if agent == "opponent":
            reward = knockout_value_opponent  # Reward for opponent being knocked out
    elif mode == RewardMode.SYMMETRIC:
        if agent == "player":
            reward = -knockout_value_player  # Penalty for player getting knocoked out
        elif agent == "opponent":
            reward = knockout_value_opponent  # Reward for opponent being knocked out
    elif mode == RewardMode.ASYMMETRIC_PLAYER:
        if agent == "player":
            reward = -knockout_value_player  # Penalty for player getting knocked out

    return reward / 3 / 30


@track_reward
def win_reward(
        env: "WarehouseBrawl",
        win_value: float = PARAMS.WIN_VALUE,
        lose_value: float = PARAMS.LOSE_VALUE
) -> float:
    """
    Computes the reward based on knockouts.


    Args:
        env (WarehouseBrawl): The game environment
        agent(str): The agent that won
        win_value (float): Reward value for knocking out opponent
        lose_value (float): Reward penalty for player being knocked out

    Returns:
        float: The computed reward.
    """

    player_stats = env.get_stats(0)
    opponent_stats = env.get_stats(1)

    is_opponent_lost = opponent_stats.lives_left == 0
    is_player_lost = player_stats.lives_left == 0

    if is_opponent_lost:
        return win_value
    elif is_player_lost:
        return lose_value
    else:
        return 0


@track_reward
def toward_centre_reward(
    env: "WarehouseBrawl",
    reward_scale: float = PARAMS.TOWARD_CENTRE_SCALE,

) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: "Player" = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    multiplier = -1 if player.body.position.x > 0 else 1
    reward = multiplier * (player.body.position.x - player.prev_x)

    return reward * reward_scale


REWARD_FUNCTION_WEIGHTS: dict[callable, float | int] = {
    damage_dealt_reward: 1000,
    damage_taken_reward: 10,
    danger_zone_reward: 0.006,
    move_to_opponent_reward: 100000,
    edge_guard_reward: 0.001,
    knockout_reward: 0.4,
    toward_centre_reward: 0.005,
    win_reward: 0.1
}
# Ensure the directory exists
def reward_initialize():
    global reward_totals
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    reward_totals = defaultdict(float)
    for func in REWARD_FUNCTION_WEIGHTS.keys():
        reward_totals[func.__name__] = 0.0

def log_rewards():
    global reward_totals
    """Writes the aggregate reward totals to a log file."""
    with open(LOG_FILE, "w") as f:
        for reward_name, total in reward_totals.items():
            f.write(f"{reward_name}: {total}\n")
    reward_initialize()

reward_initialize()


def load_reward(rewTerm: callable, rewardManager: callable) -> "RewardManager":
    reward_functions: dict[str | "RewTerm"] = {}
    for func, weight in REWARD_FUNCTION_WEIGHTS.items():
        reward_functions[func.__name__] = rewTerm(func, weight)
    reward_manager = rewardManager(reward_functions)
    return reward_manager


# usage: load_reward(rewTerm, RewardManager)

def print_statistics(filename: str = LOG_FILE):
    """
    Reads the log file and prints a formatted table with reward statistics.

    Args:
        filename (str): The path to the log file.
    """
    try:
        with open(filename, 'r') as file:
            data = {}
            for line in file:
                key, value = line.strip().split(': ')
                data[key] = float(value)

        # Convert numbers to scientific notation
        df = pd.DataFrame.from_dict(data, orient='index', columns=['Total Reward'])
        df['Total Reward'] = df['Total Reward'].apply(lambda x: f'{x:.2e}')

        print(df.to_markdown())
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


