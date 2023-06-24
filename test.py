from mainLibrary import CompleteModelReview, EvaluateModel , CompareModels, RetrainModels


# set properties
enviroment_name = 'Seaquest-v4'
policy = 'CnnPolicy'
algorithm = 'PPO'
n_envs=4
n_stack=4
total_timesteps=1000000
seed=0
verbose=1
environment_type = "Atari"
learning_rate = None
gamma = None
is_image_based = True
eval_episodes = 10


algorithms = ["PPO","A2C"]


# model_save_path2 = 'models/PPO_Atlantis-v4_CnnPolicy_timestepes:300000_gamma_None_learningRate_None'
# model_save_path = 'models/PPO_CnnPolicy_timestepes:300000_gamma:None_learningRate_None'

# evaluateModel= EvaluateModel(model_save_path, enviroment_name, 
#                              eval_episodes,algorithm,n_envs, environment_type , is_image_based, seed, n_stack)
# evaluateModel.evaluateModel()


newModel = CompleteModelReview(enviroment_name, algorithm, policy, n_envs, n_stack, seed, 
                 verbose,total_timesteps,environment_type, learning_rate, gamma, is_image_based,eval_episodes)
is_imageBased = newModel.analyzeEnv()
newModel.is_image_based = is_imageBased
newModel.trainModel()
newModel.evaluateModel()


# compareModels = CompareModels(enviroment_name, algorithm, policy, n_envs, n_stack, seed, 
#                  verbose,total_timesteps,environment_type, learning_rate, gamma, is_image_based,eval_episodes, algorithms)

# compareModels.Compare()

# model = RetrainModels(total_timesteps,algorithm, model_save_path, environment_type, 
#                       enviroment_name, n_envs, seed, is_image_based, n_stack,eval_episodes)


# model.Retrain()

