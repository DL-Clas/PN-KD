"""
Main file for intensive learning compression
"""

import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
# from cut_Res import target_model
from cut_Dens import target_model


# Customizing the Intensive Learning Environment
class MaximizeOutputEnv(gym.Env):
    def __init__(self, target_model):
        super(MaximizeOutputEnv, self).__init__()
        self.target_model = target_model

        # State-space input values, each in the range [0.3, 0.5
        self.observation_space = spaces.Box(low = 0.3, high = 0.5, shape = (4,), dtype = np.float32)

        # Action space continuous action (small adjustments between [-0.1, 0.1] for each value)
        self.action_space = spaces.Box(low = -0.1, high = 0.1, shape = (4,), dtype = np.float32)

        # Initialize the input tensor
        self.state = 0.3 + (0.5 - 0.3) * np.random.rand(4)
        self.state = self.state.astype(np.float32)  # 转换为 float32 类型

    def reset(self, *args, **kwargs):
        # Reset the state of the environment
        self.state = 0.3 + (0.5 - 0.3) * np.random.rand(4)
        self.state = self.state.astype(np.float32)  # 转换为 float32 类型
        return self.state, {}  # Return status and empty dictionary

    def step(self, action):
        # Adjusts the input tensor according to the action and limits the range to [0, 1]
        self.state = np.clip(self.state + action, 0.3, 0.5)

        # Calculate the output of the target model as a reward
        reward = float(self.target_model(self.state) * 1000)

        # Assuming no termination conditions
        done = False
        truncated = False
        return self.state, reward, done, truncated, {}


if __name__ == '__main__':
    # Training environment testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"use device is {device}")

    env = make_vec_env(lambda: MaximizeOutputEnv(target_model), n_envs = 1)

    # Training with PPO
    # model = PPO("MlpPolicy", env, verbose = 1)

    # Defining Action Noise for DDPG Intelligentsia
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean = np.zeros(n_actions), sigma = 0.1 * np.ones(n_actions))

    # Initialize the DDPG model
    model = DDPG("MlpPolicy", env, action_noise = action_noise, verbose = 1)
    model.learn(total_timesteps = 100)
    count = 0

    # Save the model after training is completec
    model_save_path = "DDPG_Brain_3_D169.zip"
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # # Loading a model from a saved file
    # loaded_model_path = "trained_ppo_model.zip"
    # model = DDPG.load(loaded_model_path)
    # print("Model loaded successfully")
    # Testing the trained model
    obs = env.reset()
    for i in range(20):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)  # 解包四个值
        print(f"Step {i + 1} -> movements: {action}, next state: {obs}, rewards: {reward}")
