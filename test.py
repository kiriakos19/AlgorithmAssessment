from mainLibrary import Model, EvaluateModel


print("Test begin")

enviroment_name = 'Breakout-v0'
policy = 'CnnPolicy'
algorithm = 'A2C'
n_envs=4
n_stack=4
total_timesteps=22
seed=0
verbose=1
fullPath = 'models/A2C_CnnPolicy_300000.zip'

# evaluateModel= EvaluateModel(fullPath, enviroment_name)
# evaluateModel.evaluateModel()


newModel = Model(enviroment_name, algorithm, policy, n_envs, n_stack, seed, verbose,total_timesteps)
newModel.analyzeEnv()
newModel.trainModel()
newModel.evaluateModel()