# rewards.py previously known as environment.py
# requires hyperparam.py
# Reward functions for the WarehouseBrawl environment

# *WARNING: POTENTIAL VULNERABILITY: PATH INJECTION*

import numpy as np
import pandas as pd
from hyperparam import RewardMode, DamageRewardMode
from hyperparam import DEFAULT_PARAMS as PARAMS
from collections import defaultdict
import functools
import os

# global logistical hyperparameters
_LOG_FILE = "/content/logs/reward_log.txt"
_REWARD_VERBOSE = True
_RELOAD_FREQUENCY = 300

# globals
temp = 0
reward_totals: dict = defaultdict(float)


def set_log_file_path(path: str) -> None:
    """
    Sets the log file path.

    :param path: Path to the log file.
    """
    global _LOG_FILE
    _LOG_FILE = path


def set_reward_log_verbose_frequency(value: bool,
                                     frequency: int = 300
) -> None:
    """
    Sets the verbosity and frequency of reward logging.

    :param value: Whether to enable verbose logging.
    :param frequency: The frequency of logging.
    """
    global _REWARD_VERBOSE, _RELOAD_FREQUENCY
    _REWARD_VERBOSE = value
    _RELOAD_FREQUENCY = frequency


def track_reward(reward_func: callable) -> callable:
    """
    Decorator to track rewards.

    :param reward_func: The reward function to wrap.
    :return: The wrapped function.
    """

    @functools.wraps(reward_func)
    def wrapper(*args, **kwargs):
        reward = reward_func(*args, **kwargs)
        reward_totals[reward_func.__name__] += reward
        return reward

    return wrapper


@track_reward
def damage_dealt_reward(env: "WarehouseBrawl",
                        damage_reward_scale: float = PARAMS.DAMAGE_REWARD_SCALE,
                        mode: DamageRewardMode = PARAMS.DAMAGE_REWARD_MODE
) -> float:
    """
    Computes the reward based on damage dealt to the opponent.

    :param env: The game environment.
    :param damage_reward_scale: The scale of the reward.
    :param mode: Reward mode, one of DamageRewardMode.
    :return: The computed reward.
    """
    opponent: "Player" = env.objects["opponent"]
    damage_dealt = opponent.damage_taken_this_frame

    reward = damage_dealt if mode in {DamageRewardMode.ASYMMETRIC_OFFENSIVE, DamageRewardMode.SYMMETRIC} else 0.0
    return reward * damage_reward_scale


@track_reward
def damage_taken_reward(env: "WarehouseBrawl",
                        damage_reward_scale: float = PARAMS.DAMAGE_REWARD_SCALE,
                        mode: DamageRewardMode = PARAMS.DAMAGE_REWARD_MODE
) -> float:
    """
    Computes the reward based on damage taken by the player.

    :param env: The game environment.
    :param damage_reward_scale: The scale of the reward.
    :param mode: Reward mode, one of DamageRewardMode.
    :return: The computed reward.
    """
    player: "Player" = env.objects["player"]
    damage_taken = player.damage_taken_this_frame

    reward = -damage_taken if mode in {DamageRewardMode.SYMMETRIC, DamageRewardMode.ASYMMETRIC_DEFENSIVE} else 0.0
    return reward * damage_reward_scale


@track_reward
def danger_zone_reward(env: "WarehouseBrawl",
                       zone_penalty: int = PARAMS.ZONE_PENALTY,
                       zone_height: float = PARAMS.ZONE_HEIGHT
) -> float:
    """
    Applies a penalty when the player surpasses a certain height threshold.

    :param env: The game environment.
    :param zone_penalty: The penalty applied when the player is in the danger zone.
    :param zone_height: The height threshold defining the danger zone.
    :return: The computed penalty.
    """
    player: "Player" = env.objects["player"]
    return zone_penalty if player.body.position.y >= zone_height else 0.0


@track_reward
def danger_zone_reward(
        env: "WarehouseBrawl",
        zone_penalty: int = PARAMS.ZONE_PENALTY,
        zone_height: float = PARAMS.ZONE_HEIGHT,
) -> float:
    """
    Applies a penalty for every time frame player surpasses a certain height threshold in the environment.

    :param env: The game environment.
    :type env: WarehouseBrawl
    :param zone_penalty: The penalty applied when the player is in the danger zone.
    :type zone_penalty: int
    :param zone_height: The height threshold defining the danger zone.
    :type zone_height: float
    :return: The computed penalty as a tensor.
    :rtype: float
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

    :param env: The game environment.
    :type env: WarehouseBrawl
    :param reward_scale: The scale of the reward.
    :type reward_scale: float
    :return: The computed reward.
    :rtype: float
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

    :param env: The game environment.
    :type env: WarehouseBrawl
    :param success_value: Reward value for the player hitting first.
    :type success_value: float
    :param fail_value: Penalty for the opponent hitting first.
    :type fail_value: float
    :param zone_width: The width of the danger zone.
    :type zone_width: float
    :return: The computed reward.
    :rtype: float
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

    :param env: The game environment.
    :type env: WarehouseBrawl
    :param mode: Reward mode, one of RewardMode.
    :type mode: RewardMode
    :param knockout_value_opponent: Reward value for knocking out opponent.
    :type knockout_value_opponent: float
    :param knockout_value_player: Reward penalty for player being knocked out.
    :type knockout_value_player: float
    :return: The computed reward.
    :rtype: float
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

    :param env: The game environment.
    :type env: WarehouseBrawl
    :param win_value: Reward value for knocking out opponent.
    :type win_value: float
    :param lose_value: Reward penalty for player being knocked out.
    :type lose_value: float
    :return: The computed reward.
    :rtype: float
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
    Computes the reward based on whether the agent is moving toward the centre of the map.

    :param env: The game environment.
    :type env: WarehouseBrawl
    :param reward_scale: The scale of the reward.
    :type reward_scale: float
    :return: The computed reward.
    :rtype: float
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
    danger_zone_reward: 0.6,
    move_to_opponent_reward: 100000,
    edge_guard_reward: 0.01,
    knockout_reward: 0.4,
    toward_centre_reward: 5,
    win_reward: 0.1
}


def reward_initialize() -> None:
    """
    Initializes the reward tracking system.
    """
    global reward_totals
    os.makedirs(os.path.dirname(_LOG_FILE), exist_ok=True)
    reward_totals = defaultdict(float)
    for func in REWARD_FUNCTION_WEIGHTS.keys():
        reward_totals[func.__name__] = 0.0


def log_rewards(do_log_now: bool = False, block_verbose: bool = False) -> None:
    """
    Writes the aggregate reward totals to a log file and prints the statistics.


    :param do_log_now: Whether to log the rewards now.
    :type do_log_now: bool
    :param block_verbose: Whether to block verbose output.
    :type block_verbose: bool
    """
    global reward_totals, temp

    with open(_LOG_FILE, "w") as f:
        for reward_name, total in reward_totals.items():
            f.write(f"{reward_name}: {total}\n")
    if (_REWARD_VERBOSE and temp % _RELOAD_FREQUENCY == 0) or do_log_now:
        print_statistics() if not block_verbose else None
        reward_initialize()
        temp = 0
    temp += 1


def load_reward(rewTerm: callable,
                rewardManager: callable
) -> "RewardManager":
    """
    Initializes and returns a RewardManager with predefined reward functions.

    :param rewTerm: A function that wraps a reward function with a weight.
    :param rewardManager: A class that manages reward functions.
    :return: A RewardManager object.
    """

    reward_functions: dict[str | "RewTerm"] = {}

    # Populate reward functions dictionary using predefined function weights
    for func, weight in REWARD_FUNCTION_WEIGHTS.items():
        reward_functions[func.__name__] = rewTerm(func, weight)

    return rewardManager(reward_functions)


def print_statistics(filename: str = _LOG_FILE) -> None:
    """
    Reads a log file and displays reward statistics in a formatted table.

    :param filename: The path to the log file.
    """
    try:
        with open(filename, 'r') as file:
            data = {}

            # Parse log file into a dictionary
            for line in file:
                key, value = line.strip().split(': ')
                data[key] = float(value)

        # Convert values to scientific notation and display as a markdown table
        df = pd.DataFrame.from_dict(data, orient='index', columns=['Total Reward'])
        df['Total Reward'] = df['Total Reward'].apply(lambda x: f'{x:.2e}')

        # Add grand total row
        grand_total = df['Total Reward'].apply(lambda x: float(x)).sum()
        df.loc['Grand Total'] = f'{grand_total:.2e}'

        print(df.to_markdown())

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


reward_initialize()
