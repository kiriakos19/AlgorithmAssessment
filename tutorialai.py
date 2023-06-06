import gym

from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3 import TD3


from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.env_util import make_atari_env

import os

#!python3 -m atari_py.import_roms /home/cool/Documents/ReinforcementLearningCourse/Roms/ROMS/

class Model:
    def __init__(self,enviroment_name, algorithm, policy, n_envs, n_stack, seed, verbose,total_timesteps):
        self.enviroment_name = enviroment_name
        self.algorithm= algorithm
        self.policy = policy
        self.n_envs = n_envs
        self.n_stack = n_stack
        self.seed = seed
        self.verbose = verbose
        self.total_timesteps = total_timesteps
        self.log_path = f"logs/{self.algorithm}_{self.policy}_{self.total_timesteps}"
        self.model_save_path = f"models/{self.algorithm}_{self.policy}_{self.total_timesteps}"

    def analyzeEnv(self):
        env = make_atari_env(self.enviroment_name)
        
        env.reset()
        print("environment action space: ",env.action_space)
        print("sample action: ", env.action_space.sample())
        print("observation space shape: ", env.observation_space.shape)
        print("sample observation: ", env.observation_space.sample())



    def trainModel(self):
        env = gym.make(self.enviroment_name)
        env.reset()
    
        
    
    
        env = make_atari_env(self.enviroment_name, n_envs=self.n_envs, seed= self.seed)
        env = VecFrameStack(env, n_stack=self.n_stack)
    
        if (self.lgorithm == 'A2C'):
            model = A2C(self.policy, env, verbose=self.verbose,tensorboard_log=self.log_path)
        if (self.algorithm == 'PPO'):
            model = PPO(self.policy, env, verbose=self.verbose, tensorboard_log=self.log_path)
        if (self.algorithm == 'DQN'):
            model = DQN(self.policy, env, verbose=self.verbose, tensorboard_log=self.log_path)
        if (self.algorithm == 'TD3'):
            model = TD3(self.policy, env, verbose=self.verbose, tensorboard_log=self.log_path)
        

        model.learn(total_timesteps=self.total_timesteps)
        model.save(self.model_save_path)
        del model

    def evaluateModel(self):
        env = make_atari_env('Breakout-v0')
        env = VecFrameStack(env, n_stack=4)
        model = A2C.load(f"{self.model_save_path}.zip", env)

        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=50)
        print("mean reward: ", mean_reward)
        print("std_reward: ", std_reward)

    



