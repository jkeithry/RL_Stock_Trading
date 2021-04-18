import ssrl
import numpy as np

from stable_baselines3 import PPO, DQN, A2C, TD3, HER
from stable_baselines3 import DDPG
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from stable_baselines3.common.env_checker import check_env

stock = 'M'
stock_df = ssrl.download_data(stock, "2009-01-01", "2021-03-01")

stock_df = ssrl.add_technical_indicator(stock_df).dropna()

training_data = ssrl.data_split(stock_df, "2009-01-01", "2018-12-31")
test_data = ssrl.data_split(stock_df, "2019-01-01", "2021-03-31")
# print(training_data.head(10))
# print(test_data.head(10))
env = ssrl.StockEnvTrain(training_data, 15000)
# check_env(env)
# exit(-1)
env_train, _ = env.get_sb_env()

n_actions = env_train.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# model settings
model_sac = SAC(policy='MlpPolicy',
                   env=env_train,
                   action_noise=action_noise,
                   verbose=0,
                   batch_size=128,
                   buffer_size=50000,
                   learning_rate=0.001
                   )

trained = model_sac.learn(total_timesteps=10000, log_interval=10)
# model_ddpg.save("ddpg_stockenvtrain")
ssrl.test_model(trained, test_data, 15000)

# add regularization to try and incetivize model to hold stock, use assets/current holdings -> pass to sigmoid to add reward from 0-1