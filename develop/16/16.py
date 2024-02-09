import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from qcenv.environments import EasyTestEnv

# env = gym.make("CartPole-v1")
steps = [1, 10, 100, 1000, 10000]
for step in steps:
    env = EasyTestEnv()
    model = PPO(MlpPolicy, env, verbose=0)
    model.learn(total_timesteps=step)
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    # print(f"steps:{step}")
    # print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    rewards, length = evaluate_policy(
        model, env, n_eval_episodes=100, return_episode_rewards=True
    )
    print(f"steps:{step}")
    print(f"rewards:{rewards}")
    print(f"mean reward:{np.mean(rewards)}")
    print(f"length:{length}")
    print(f"mean length:{np.mean(length)}")
