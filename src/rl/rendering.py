import gymnasium as gym
import torch
from EnvForEval import PandaReachPoseEnv
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from wrapper import TorchToNumpyWrapper



def make_env():
    env = PandaReachPoseEnv(  
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode="human"   # enable viewer
    )

    env = TorchToNumpyWrapper(env)
    return env


# Build env
env = DummyVecEnv([make_env])

env = VecNormalize.load("checkpoints/vec_normalize_control.pkl", env)

env.training = False
env.norm_reward = False   
env.norm_obs = True       

# Load model
model = PPO.load("checkpoints/panda_grasp_ppo_control", env=env, device="cpu")

# Rollout
obs = env.reset()

rewards = []
done = [False]

while not done[0]:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    viewer = env.envs[0].viewer

    if viewer is not None:
        viewer.set_camera_xyz(0.9, 0.7, 0.8)
        viewer.set_camera_rpy(r=0, p=0, y=129)
        viewer.render()

        time.sleep(0.5)

        if done[0]:
            break
