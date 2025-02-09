# environment.py 
# requires hyperparam.py
# Reward functions for the WarehouseBrawl environment

from enum import Enum
from typing import List, Tuple

import numpy as np
from hyperparam import RewardMode, DamageRewardMode
from hyperparam import DEFAULT_PARAMS as PARAMS

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


def damage_interaction_reward(
    env: "WarehouseBrawl",
    mode: DamageRewardMode = PARAMS.DAMAGE_REWARD_MODE,
) -> float:
    """
    Computes the reward based on damage interactions between players.

    Modes:
    - ASYMMETRIC_OFFENSIVE (0): Reward is based only on damage dealt to the opponent
    - SYMMETRIC (1): Reward is based on both dealing damage to the opponent and avoiding damage
    - ASYMMETRIC_DEFENSIVE (2): Reward is based only on avoiding damage

    Args:
        env (WarehouseBrawl): The game environment
        mode (DamageRewardMode): Reward mode, one of DamageRewardMode

    Returns:
        float: The computed reward.
    """
    # Getting player and opponent from the enviornment
    player: "Player" = env.objects["player"]
    opponent: "Player" = env.objects["opponent"]

    # Reward dependent on the mode
    damage_taken = player.damage_taken_this_frame
    damage_dealt = opponent.damage_taken_this_frame

    if mode == DamageRewardMode.ASYMMETRIC_OFFENSIVE:
        reward = damage_dealt
    elif mode == DamageRewardMode.SYMMETRIC:
        reward = damage_dealt - damage_taken
    elif mode == DamageRewardMode.ASYMMETRIC_DEFENSIVE:
        reward = -damage_taken
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return reward

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
    reward = -zone_penalty if player.body.position.y >= zone_height else 0.0

    return reward

# Prolly ignore this, no weapons in the game
def stock_advantage_reward(
    env: "WarehouseBrawl",
    success_value: float = PARAMS.STOCK_SUCCESS_VALUE,
) -> float:

    """
    Computes the reward given for every time step your agent is edge guarding the opponent.

    Args:
        env (WarehouseBrawl): The game environment
        success_value (float): Reward value related to having/gaining a weapon (however you define it)
    Returns:
        float: The computed reward.
    """

    reward = 0.0
    reward += success_value if env.objects["player"].WeaponHeldThisFrame else 0.0

    return reward

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
    player_position_dif = np.array([player.body.position.x_change, player.body.position.y_change])

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

    return reward

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
    if  (abs(env.objects["player"].body.position.x) < zone_width/2 - a or # player in the middle
         abs(env.objects["opponent"].body.position.x) < zone_width/2 - a or # opponent in the middle
         env.objects["player"].body.position.x * env.objects["opponent"].body.position.x < 0 # both players are not on the same edge
         ):
        return reward
    
    # condition met
    if env.objects["opponent"].demange_taken_this_frame > 0:
        reward = success_value
    elif env.objects["player"].demange_taken_this_frame > 0:
        reward = fail_value

    return reward


def knockout_reward(
    env: "WarehouseBrawl",
    agent: str = "player",
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
        agent(str): The agent that was knocked out
        mode (RewardMode): Reward mode, one of RewardMode
        knockout_value_opponent (float): Reward value for knocking out opponent
        knockout_value_player (float): Reward penalty for player being knocked out

    Returns:
        float: The computed reward.
    """
    reward = 0.0

    # Mode logic to compute reward
    if mode == RewardMode.ASYMMETRIC_OPPONENT:
        if agent == "opponent":
            reward = knockout_value_opponent # Reward for opponent being knocked out
    elif mode == RewardMode.SYMMETRIC:
        if agent == "player":
            reward = -knockout_value_player  # Penalty for player getting knocoked out
        elif agent == "opponent":
            reward = knockout_value_opponent # Reward for opponent being knocked out
    elif mode == RewardMode.ASYMMETRIC_PLAYER:
        if agent == "player":
            reward = -knockout_value_player  # Penalty for player getting knocked out

    return reward

def win_reward(
    env: "WarehouseBrawl",
    agent: str = "player",
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

    reward = win_value if agent == "player" else -lose_value
    return reward