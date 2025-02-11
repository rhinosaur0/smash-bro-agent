# hyperparam.py 
# the hyperparameters object is used to store the reward values for the agent

from enum import Enum

class RewardMode(Enum):
    ASYMMETRIC_OPPONENT = 0
    SYMMETRIC = 1
    ASYMMETRIC_PLAYER = 2

class DamageRewardMode(Enum):
    ASYMMETRIC_OFFENSIVE = 0
    SYMMETRIC = 1
    ASYMMETRIC_DEFENSIVE = 2

# panalties shall be negative values
class Hyperparameters:
    def __init__(self,   
                 WIN_VALUE: float = 300.0, 
                 LOSE_VALUE: float = 200.0, 
                 KNOCKOUT_VALUE_PLAYER: float = 100.0, 
                 KNOCKOUT_VALUE_OPPONENT: float = 100.0, 
                 SUCCESS_VALUE: float = 50.0, 
                 FAIL_VALUE: float = 0.0, 
                 STOCK_SUCCESS_VALUE: float = 150.0, 
                 MOVE_TO_OPPONENT_SCALE: float = 1,
                 EDGE_GUARD_SUCCESS: float = 50.0,
                 EDGE_GUARD_FAIL: float = 0.0,
                 TOWARD_CENTRE_SCALE: float = 0.1,
                 DAMAGE_REWARD_SCALE: float = 1,
                 ZONE_HEIGHT: float = 7,
                 ZONE_WIDTH: float = 10.67,
                 ZONE_PENALTY: float = -20.0,
                 REWARD_MODE: int = RewardMode.SYMMETRIC,
                 DAMAGE_REWARD_MODE: int = DamageRewardMode.SYMMETRIC
                 ) -> None:
        self.WIN_VALUE = WIN_VALUE
        self.LOSE_VALUE = LOSE_VALUE
        self.KNOCKOUT_VALUE_PLAYER = KNOCKOUT_VALUE_PLAYER
        self.KNOCKOUT_VALUE_OPPONENT = KNOCKOUT_VALUE_OPPONENT
        self.SUCCESS_VALUE = SUCCESS_VALUE
        self.FAIL_VALUE = FAIL_VALUE
        self.STOCK_SUCCESS_VALUE = STOCK_SUCCESS_VALUE
        self.MOVE_TO_OPPONENT_SCALE = MOVE_TO_OPPONENT_SCALE
        self.DAMAGE_REWARD_SCALE = DAMAGE_REWARD_SCALE
        self.EDGE_GUARD_SUCCESS = EDGE_GUARD_SUCCESS
        self.EDGE_GUARD_FAIL = EDGE_GUARD_FAIL
        self.TOWARD_CENTRE_SCALE = TOWARD_CENTRE_SCALE
        self.ZONE_HEIGHT = ZONE_HEIGHT
        self.ZONE_WIDTH = ZONE_WIDTH
        self.ZONE_PENALTY = ZONE_PENALTY
        self.REWARD_MODE = REWARD_MODE
        self.DAMAGE_REWARD_MODE = DAMAGE_REWARD_MODE

# create an object of class hyperparameters that has default reward values
DEFAULT_PARAMS = Hyperparameters()
# create an an object of class hyperparameters that has really offensive reward values
OFFENSIVE_AGENT_PARAMS = Hyperparameters(
    WIN_VALUE = 300.0,
    LOSE_VALUE = 200.0,
    KNOCKOUT_VALUE_PLAYER = 100.0,
    KNOCKOUT_VALUE_OPPONENT = 100.0,
    SUCCESS_VALUE = 50.0,
    FAIL_VALUE = 0.0,
    STOCK_SUCCESS_VALUE = 150.0,
    MOVE_TO_OPPONENT_SCALE = 1.0,
    EDGE_GUARD_SUCCESS = 50.0,
    EDGE_GUARD_FAIL = 0.0,
    ZONE_HEIGHT = 4.2,
    ZONE_PENALTY = -20.0,
    REWARD_MODE = RewardMode.SYMMETRIC,
    DAMAGE_REWARD_MODE = DamageRewardMode.ASYMMETRIC_OFFENSIVE
)
# create an an object of class hyperparameters that has really defensive reward values
DEFENSIVE_AGENT_PARAMS = Hyperparameters(
    WIN_VALUE = 300.0,
    LOSE_VALUE = 200.0,
    KNOCKOUT_VALUE_PLAYER = 100.0,
    KNOCKOUT_VALUE_OPPONENT = 100.0,
    SUCCESS_VALUE = 50.0,
    FAIL_VALUE = -25.0,
    STOCK_SUCCESS_VALUE = 150.0,
    MOVE_TO_OPPONENT_SCALE = 1.0,
    EDGE_GUARD_SUCCESS = 50.0,
    EDGE_GUARD_FAIL = -25.0,
    ZONE_HEIGHT = 4.2,
    ZONE_PENALTY = -20.0,
    REWARD_MODE = RewardMode.SYMMETRIC,
    DAMAGE_REWARD_MODE = DamageRewardMode.ASYMMETRIC_DEFENSIVE
)