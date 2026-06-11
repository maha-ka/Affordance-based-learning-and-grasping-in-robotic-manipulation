import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import torch.nn as nn
from gymnasium.wrappers import TimeLimit
from env import PandaReachPoseEnv
from wrapper import TorchToNumpyWrapper


seed = 42
device = 'cuda' if torch.cuda.is_available() else 'cpu'
np.random.seed(seed)
torch.manual_seed(seed)


def make_env():
    env = PandaReachPoseEnv(
        obs_mode="state",
        control_mode="pd_ee_delta_pose"
    )

    env = TorchToNumpyWrapper(env) 

    env = TimeLimit(env, max_episode_steps=200)
    env = Monitor(env)

    return env

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=False)



policy_kwargs = dict(
    activation_fn=nn.ELU, 
    net_arch=dict(
        pi=[128, 128],   # policy network
        vf=[128, 128]    # value network
    ),
    log_std_init=0.0  # initial_log_std
)


model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=4096,
    batch_size=512,
    n_epochs=8,
    ent_coef = 0.01,
    learning_rate=1e-4,
    clip_range=0.1,
    gamma=0.99,
    policy_kwargs = policy_kwargs,
    seed = seed,
    device='cpu',
    tensorboard_log="./tb_logs_control/"
)



model.learn(total_timesteps=500_000 )
env.save("vec_normalize_control.pkl")
model.save("panda_grasp_ppo_control")


