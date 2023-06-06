from tutorialai import Model


print("ola kala")

enviroment_name = 'Breakout-v0'
policy = 'CnnPolicy'
algorithm = 'A2C'
n_envs=4
n_stack=4
total_timesteps=10
seed=0
verbose=1

x = Model(enviroment_name, algorithm, policy, n_envs, n_stack, seed, verbose,total_timesteps)
x.analyzeEnv()