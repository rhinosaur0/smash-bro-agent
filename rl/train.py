# train.py

# Depreciated: This script is used to train the model using the RecurrentPPO algorithm, we can just 
# use andrew's script to train the model using the rPPO algorithm.

import gym
import wandb
from stable_baselines3 import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
import WarehouseEnv
from wandb.integration.sb3 import WandbCallback

# Define hyperparameters
NUM_ENVS = 4
# each call to env.step() in every environment will be counted
SAVE_FREQ = max(1000 // NUM_ENVS, 1)
SAVE_PATH = "./checkpoints/"
TOTAL_TIMESTEPS = 25000

# Initialize wandb
config = {
    "policy_type": "MlpLstmPolicy",
    "total_timesteps": TOTAL_TIMESTEPS
}

run = wandb.init(
    project="brawhalla-rl",
    name="rppo-warehouse",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    # save_code=True,  # optional
)

# Define callbacks
wandb_callback = WandbCallback(
    model_save_path=f"models/{run.id}",
    verbose=2
)

checkpoint_callback = CheckpointCallback(
    save_freq=SAVE_FREQ,
    save_path=SAVE_PATH,
    name_prefix="rppo_warehouse",
)

callback = CallbackList([checkpoint_callback, wandb_callback])

# Setup environment and model
# NOTE: I assumed the environment is a single env and wrap it to create multiple envs
env = DummyVecEnv([lambda: WarehouseEnv() for _ in range(NUM_ENVS)])

model = RecurrentPPO(
    config["policy_type"],
    env, 
    verbose=1,
    tensorboard_log=f"runs/{run.id}",
    policy_kwargs={
        "log_std_init": -2,  # Log of standard deviation of action distribution
        "net_arch": [dict(pi=[64, 64], vf=[64, 64])],  # Network architecture
    },
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
)

# Train the model
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=callback
)

model.save("checkpoints/rppo_warehouse_final")

wandb.finish()
