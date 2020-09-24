import gym
import os
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.bench import Monitor

log_dir = './logs/'
os.makedirs(log_dir,exist_ok=True)

env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])

model = PPO2('MlpPolicy', env,verbose=1,tensorboard_log=log_dir)
model.learn(total_timesteps=10000)

state = env.reset()
for i in range(200):
    env.render()

    action,_ = model.predict(state)
    state,rewards,done,info = env.step(action)

    if done:
        break

env.close()