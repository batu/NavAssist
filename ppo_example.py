try:
  from rlnav.configs.configurations import setup_configurations, get_config_dict
  from rlnav.logging import test_model, WANDBMonitor
except ImportError as e:
  raise Exception("rlnav package is not installed! Please go into the PythonLibraries/rlnav and install the package locally by running 'pip install -e .' Also, please ensure that you have cloned to repo with --recurse-submodules flag.")

from mlagents_envs import environment
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import MultiAgentVecEnv

import random

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToMultiGymWrapper 
from pathlib import Path


def make_env(path, channels):
  def _init():
    unity_env = UnityEnvironment(str(path), base_port=5000 + random.randint(0,5000), side_channels=channels)
    env = UnityToMultiGymWrapper(unity_env, env_channel=channels[0])
    env = WANDBMonitor(env, config=None) # if Config is none, wandb syncing isn't used.
    return env
  return _init

if __name__ == "__main__":

  env_path = Path(f"GymEnvironments/0_Debug/Windows/Env.exe")
  
  config = get_config_dict(PPO) # This is where the algorithm, the observations and the environment parameters are configured. 
                                # You can go to the rlnav/configs folder to find the yaml files that define them.

  config["observation_config"]["use_occupancy"] = False # For example, lets not use the occupancy_grid observation. It is not neccessary for this env.

  wandb_config, network_config, alg_config, channels = setup_configurations(config) # In my work I use WANDB but we don't have it in this example.
  
  env = MultiAgentVecEnv(make_env(env_path, channels=channels)) # The default configuration has 32 agents in the environment.
  model = PPO("MlpPolicy", env, policy_kwargs=network_config, **alg_config)
  
  model.learn(total_timesteps=100_000)
  
  final_success_rate = test_model(env, model, test_count=500)
  print(final_success_rate)
