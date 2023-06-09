from mainLibrary import Model, EvaluateModel


print("Test begin")

enviroment_name = 'Breakout-v0'
policy = 'CnnPolicy'
algorithm = 'A2C'
n_envs=4
n_stack=4
total_timesteps=300000
seed=0
verbose=1
fullPath = 'models/A2C_CnnPolicy_300000.zip'
environment_type = "Atari"
learning_rate = 0.007
gamma = None
is_image_based = False

# evaluateModel= EvaluateModel(fullPath, enviroment_name)
# evaluateModel.evaluateModel()


newModel = Model(enviroment_name, algorithm, policy, n_envs, n_stack, seed, verbose,total_timesteps,environment_type, learning_rate, gamma, is_image_based)
is_imageBased = newModel.analyzeEnv()
newModel.is_image_based = is_imageBased
newModel.trainModel()
# newModel.evaluateModel()