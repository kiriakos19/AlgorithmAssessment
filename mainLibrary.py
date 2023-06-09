import gym

from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3 import TD3


from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.env_util import make_atari_env

import os

#!python3 -m atari_py.import_roms /home/mypc/

class Model:
    def __init__(self,enviroment_name, algorithm, policy, n_envs, n_stack, seed, verbose,total_timesteps,environment_type,learning_rate, gamma, is_image_based):
        self.enviroment_name = enviroment_name
        self.algorithm= algorithm
        self.policy = policy
        self.n_envs = n_envs
        self.n_stack = n_stack
        self.seed = seed
        self.verbose = verbose
        self.enviroment_type = environment_type
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.is_image_based = is_image_based
        self.total_timesteps = total_timesteps
        self.log_path = f"logs/{self.algorithm}_{self.enviroment_name}_{self.policy}_timesteps:{self.total_timesteps}_gamma:{self.gamma}_learningRate{self.learning_rate}"
        self.model_save_path = f"models/{self.algorithm}_{self.policy}_timestepes:{self.total_timesteps}_gamma:{self.gamma}_learningRate{self.learning_rate}"


    def analyzeEnv(self):
        env = gym.make(self.enviroment_name)  

        print("Environment action space: ",env.action_space)
        print("Sample action: ", env.action_space.sample())
        print("Observation space shape: ", env.observation_space.shape)
        print("Sample observation: ", env.observation_space.sample())
        print("Reward Range: ",env.reward_range)
        print("Maximum Episode Steps: ", env.spec.max_episode_steps)

        observation_space = env.observation_space   
        is_image_based = isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 3
        if is_image_based:
            print("The environment provides image-based observations.")
        else:
            print("The environment does not provides image-based observations.")
        env.reset()
        return is_image_based


    def trainModel(self):
#initialize default values for gamma
        if (self.gamma is None):
            self.gamma = 0.99
        
        if (self.enviroment_type == "Atari"):
            env = make_atari_env(self.enviroment_name, n_envs=self.n_envs, seed= self.seed)    
        else:
            env = gym.make(self.enviroment_name)
        
        if (self.is_image_based):
            env = VecFrameStack(env, n_stack=self.n_stack)
        env.reset() 
        if (self.algorithm == 'A2C'):
#initialize default values for learing rate
            if (self.learning_rate is None):
                self.learning_rate = 0.0007
            model = A2C(self.policy, env, verbose=self.verbose,learning_rate=self.learning_rate , gamma=self.gamma, tensorboard_log=self.log_path)
        elif (self.algorithm == 'PPO'):
            if (self.learning_rate is None):
                self.learning_rate = 0.0003
            model = PPO(self.policy, env, verbose=self.verbose, learning_rate=self.learning_rate, gamma=self.gamma, tensorboard_log=self.log_path)
        elif (self.algorithm == 'DQN'):
            if (self.learning_rate is None):
                self.learning_rate = 0.0001
            model = DQN(self.policy, env, verbose=self.verbose, learning_rate=self.learning_rate,  gamma=self.gamma,tensorboard_log=self.log_path)
        elif (self.algorithm == 'TD3'):
            if (self.learning_rate is None):
                self.learning_rate = 0.001
            model = TD3(self.policy, env, verbose=self.verbose, learning_rate=self.learning_rate, gamma=self.gamma, tensorboard_log=self.log_path)
        else :
            print("This algorithm is not implemeted yet!!!")
            return
        
        model.learn(total_timesteps=self.total_timesteps)
        model.save(self.model_save_path)
        del model

    def evaluateModel(self):
        env = make_atari_env(self.enviroment_name)
        env = VecFrameStack(env, n_stack=4)
        model = A2C.load(f"{self.model_save_path}.zip", env)

        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=50)

        print("mean reward: ", mean_reward)
        print("std_reward: ", std_reward)

class EvaluateModel:
    def __init__(self, fullPath, enviroment_name):
        self.fullPath = fullPath
        self.enviroment_name = enviroment_name

    print("test")
    def evaluateModel(self):
        env = make_atari_env(self.enviroment_name)
        env = VecFrameStack(env, n_stack=4)
        model = A2C.load(self.fullPath, env)

        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=50)

        print("mean reward: ", mean_reward)
        print("std_reward: ", std_reward)

    
# under construction
# class CompareModels:




