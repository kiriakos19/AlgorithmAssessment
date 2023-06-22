import gym
import numpy as np

from stable_baselines3 import A2C
from stable_baselines3 import DDPG , HerReplayBuffer
from stable_baselines3 import PPO
from stable_baselines3 import TD3


from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.env_util import make_atari_env

import os
import time

#!python3 -m atari_py.import_roms /home/mypc/

############################################# Main model Class ########################################################
class Model:
    def __init__(self,enviroment_name, algorithm, policy, n_envs, 
                 n_stack, seed, verbose,total_timesteps,environment_type,
                 learning_rate, gamma, is_image_based, eval_episodes):
        
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
        self.eval_episodes = eval_episodes
        self.is_image_based = is_image_based
        self.total_timesteps = total_timesteps
        self.log_path = f"logs/{self.algorithm}_{self.enviroment_name}_{self.policy}_timesteps:{self.total_timesteps}_gamma_{(str(self.gamma)).replace('.','_')}_learningRate_{(str(self.learning_rate)).replace('.','_')}"
        self.model_save_path = f"models/{self.algorithm}_{self.enviroment_name}_{self.policy}_timestepes:{self.total_timesteps}_gamma_{(str(self.gamma)).replace('.','_')}_learningRate_{(str(self.learning_rate)).replace('.','_')}"


    def analyzeEnv(self):
        env = gym.make(self.enviroment_name)  
        # action space represents the set of possible actions tha agent can take in the environment
        if isinstance(env.action_space, gym.spaces.Discrete):   
            # In a discrete environment the agent can choose from a finite se of distrinct actions     
            print("The environment has a discrete action space.\n")
        else:
            # In a continuous environment the agent can select any value within a contiuous range by real numbers or n-dimensionl vectors
            print("The environment has continuous action space")
            print("Is action space bounded?:",env.action_space.is_bounded())
            print("Max ",env.action_space.high)
            print("Min ",env.action_space.low)
        
        print("Environment action space: ",env.action_space)
        print("Action space shape: ",env.action_space.shape)
        print("Action space shape shample: ", env.action_space.sample())  
        # Observation space represents the information the agent receives from the environment at each time step
        print("Observation space shape: ", env.observation_space.shape)
        # Other helpfull informations for environment
        print("Is environment NonDeterministic?: ",env.spec.nondeterministic)
        print("Environment metadata: ",env.metadata)
        print("Reward Treshold: ", env.spec.reward_threshold)
        print("Reward Range: ",env.reward_range)
        print("Maximum Episode Steps: ", env.spec.max_episode_steps)
        
        # Code to check if environment is image based so later in traing use (VecFrameStack) and create vectorized environment
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
        env = CreateEnv(self)
        env.reset()
        model = AlgorithmChecker(self, env)
        print("Start Training")
        model.learn(total_timesteps=self.total_timesteps)
        print("Save model to path: ", self.model_save_path)
        model.save(self.model_save_path)
        del model

    def evaluateModel(self):
        env = CreateEnv(self)
        env.reset()
        mean_reward, std_reward = Evaluate(self, env)
        print("mean reward: ", mean_reward)
        print("std_reward: ", std_reward)

####################### End Class Model ###########################################################


############################################# helpfull functions ########################################################
def A2CModel(self, env):
        print("Start initialize A2C")
        if (self.learning_rate is None):
            self.learning_rate = 0.0007
        print("Learning rate value is: ", self.learning_rate)
        print("Gamma value is: ", self.gamma)
        print("Creating Model")
        model = A2C(self.policy, env, verbose=self.verbose,learning_rate=self.learning_rate ,
                         gamma=self.gamma, tensorboard_log=self.log_path)
        return model

def PPOModel(self,env):
    print("Start initalize PPO")

    if (self.learning_rate is None):
            self.learning_rate = 0.0003

    print("Learning rate value is: ", self.learning_rate)
    print("Gamma value is: ", self.gamma)
    print("Creating Model")
    model = PPO(self.policy, env, learning_rate=self.learning_rate , gamma=self.gamma,
                        verbose=self.verbose,  tensorboard_log=self.log_path) 
    return model

def DDPGModel(self, env):
    print("Start initialize DDPG")

    if (self.learning_rate is None):
        self.learning_rate = 0.001
    # create action noise because TD3 and DDPPG use deterministic policy
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), 
                                     sigma=float(0.1) * np.ones(n_actions))
    
    print("Learning rate value is: ", self.learning_rate)
    print("Gamma value is: ", self.gamma)
    print("Action noise :",action_noise)
    print("Creating Model")
    model = DDPG(self.policy,env,learning_rate= self.learning_rate ,gamma=self.gamma ,
                         verbose=self.verbose, action_noise = action_noise, tensorboard_log=self.log_path)
    return model

def TD3Model(self, env):
    print("Start initialize TD3")

    if (self.learning_rate is None):
        self.learning_rate = 0.001
    # create action noise because TD3 and DDPPG use deterministic policy
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1*np.ones(n_actions))
    print("Learning rate value is: ", self.learning_rate)
    print("Gamma value is: ", self.gamma)
    print("Action noise is: ", action_noise)
    
    print("Creating Model")
    model = TD3(self.policy, env,learning_rate=self.learning_rate ,verbose=self.verbose,
                        gamma=self.gamma,action_noise=action_noise ,tensorboard_log=self.log_path)
    return model

def CreateEnv(self):
    if (self.enviroment_type == "Atari"):
        print("Create atari environment")
        # create environmet with n_envs parallen environments
        env = make_atari_env(self.enviroment_name, n_envs=self.n_envs, seed= self.seed)   
        if (self.is_image_based):
            #create env with n_stack consecutive frames to stack together)
            env = VecFrameStack(env, n_stack=self.n_stack) 
    else:
        print("Create gym environment")            
        if (self.is_image_based):
            #create env with n_stack consecutive frames to stack together)
            env = make_vec_env(self.enviroment_name,self.n_envs)
            env = VecFrameStack(env, n_stack=self.n_stack)
        else:
            env = gym.make(self.enviroment_name)
    return env

def AlgorithmChecker(self, env):
    if (self.algorithm == 'A2C'):
        model = A2CModel(self, env)
    elif (self.algorithm == 'PPO'):
        model = PPOModel(self, env)        
    elif (self.algorithm == 'DDPG'):
        model = DDPGModel(self, env)
    elif (self.algorithm == 'TD3'):
        model = TD3Model(self, env)            
    else :
        print("This algorithm is not implemeted yet!!!")
        return
        
    return model

def Evaluate(self, env):
    if (self.algorithm == 'A2C'):
        print("Start evaluation for A2C algorithm")
        model = A2C.load(f"{self.model_save_path}.zip", env)
    elif (self.algorithm == 'PPO'):
        print("Start evaluation for PPO algorithm")
        model = PPO.load(f"{self.model_save_path}.zip", env)
    elif (self.algorithm == 'DDPG'):
        print("Start evaluation for DDPG algorithm")
        model = DDPG.load(f"{self.model_save_path}.zip", env)
    
    elif (self.algorithm == 'TD3'):
        print("Start evaluation for TD3 algorithm")
        model = TD3.load(f"{self.model_save_path}.zip", env)
    else :
        print("This algorithm is not implemeted yet!!!")
        return

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), 
                                                  n_eval_episodes=self.eval_episodes,render=True)
    return mean_reward, std_reward
    
############################################# End helpfull functions ########################################################

############################################# Evaluate Model ########################################################

class EvaluateModel:
    def __init__(self, model_save_path, enviroment_name, eval_episodes, algorithm, n_envs, enviroment_type, is_image_based, seed, n_stack):
        self.model_save_path = model_save_path
        self.enviroment_name = enviroment_name
        self.eval_episodes = eval_episodes
        self.algorithm = algorithm
        self.n_envs = n_envs
        self.enviroment_type =enviroment_type
        self.is_image_based = is_image_based
        self.seed = seed
        self.n_stack = n_stack

    print("test")
    def evaluateModel(self):

        env = CreateEnv(self)
        env.reset() 
        mean_reward, std_reward = Evaluate(self, env) 

        print("mean reward: ", mean_reward)
        print("std_reward: ", std_reward)


############################################# End Evaluate Model ########################################################  

############################################# Compare Models ########################################################

class CompareModels: 
    def __init__(self,enviroment_name, algorithm, policy, n_envs, 
                 n_stack, seed, verbose,total_timesteps,environment_type,
                 learning_rate, gamma, is_image_based, eval_episodes, algorithms):
        
        self.enviroment_name = enviroment_name
        self.algorithm= algorithm
        self.algorithms= algorithms
        self.policy = policy
        self.n_envs = n_envs
        self.n_stack = n_stack
        self.seed = seed
        self.verbose = verbose
        self.enviroment_type = environment_type
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.eval_episodes = eval_episodes
        self.is_image_based = is_image_based
        self.total_timesteps = total_timesteps
        self.log_path = f"logs/{self.algorithm}_{self.enviroment_name}_{self.policy}_timesteps:{self.total_timesteps}_gamma_{(str(self.gamma)).replace('.','_')}_learningRate_{(str(self.learning_rate)).replace('.','_')}"
        self.model_save_path = f"models/{self.algorithm}_{self.enviroment_name}_{self.policy}_timestepes:{self.total_timesteps}_gamma_{(str(self.gamma)).replace('.','_')}_learningRate_{(str(self.learning_rate)).replace('.','_')}"


    def Compare(self):
        data = dict()
        if (self.gamma is None):
            self.gamma = 0.99
        for algorithm in self.algorithms:
            self.algorithm = algorithm
            env = CreateEnv(self)
            env.reset() 
            print(self.algorithm)
            model = AlgorithmChecker(self, env)
            print("Start Training")
            model.learn(total_timesteps=self.total_timesteps)
            print("Save model to path: ", self.model_save_path)
            model.save(self.model_save_path)
            del model

            mean_reward, std_reward = Evaluate(self, env)
            print("mean reward: ", mean_reward)
            print("std_reward: ", std_reward)
            data[algorithm] = f"Mean reward: {mean_reward}, std_reward:{std_reward}"
            print(data)
        print(data)
    

############################################# End Compare Models ######################################################## 


############################################# Retrain Models ########################################################    
class RetrainModels: 
    def __init__(self, total_timesteps, algorithm, model_save_path, 
                 enviroment_type, enviroment_name, n_envs, seed, is_image_based, n_stack, eval_episodes):
        self.total_timesteps = total_timesteps
        self.algorithm = algorithm
        self.model_save_path = model_save_path
        self.enviroment_type = enviroment_type
        self.enviroment_name = enviroment_name
        self.n_envs = n_envs
        self.seed = seed
        self.is_image_based = is_image_based
        self.n_stack = n_stack    
        self.eval_episodes = eval_episodes




    def Retrain(self):
        env = CreateEnv(self)
        env.reset()



        if (self.algorithm == 'A2C'):
            print("Start evaluation for A2C algorithm")
            model = A2C.load(f"{self.model_save_path}.zip", env)
        elif (self.algorithm == 'PPO'):
            print("Start evaluation for PPO algorithm")
            model = PPO.load(f"{self.model_save_path}.zip", env)
        elif (self.algorithm == 'DDPG'):
            print("Start evaluation for DDPG algorithm")
            model = DDPG.load(f"{self.model_save_path}.zip", env)
        elif (self.algorithm == 'TD3'):
            print("Start evaluation for TD3 algorithm")
            model = TD3.load(f"{self.model_save_path}.zip", env)
        else :
            print("This algorithm is not implemeted yet!!!")
            return
        model.learn(total_timesteps=self.total_timesteps)
        
        self.model_save_path = self.model_save_path + "_retrained"
        print("Save model to path: ", self.model_save_path)
        model.save(self.model_save_path)

        del model
        env.reset()

        mean_reward, std_reward = Evaluate(self, env) 

        print("mean reward: ", mean_reward)
        print("std_reward: ", std_reward)


############################################# RetrainModels ########################################################              
         




