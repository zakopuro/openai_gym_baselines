import gym
import os
import numpy as np
import datetime
import pytz
from stable_baselines.results_plotter import load_results, ts2xy


log_dir = './logs/'
os.makedirs(log_dir, exist_ok=True)

num_update = 0
best_mean_reward = -np.inf

def callback(_locals, _globals):
    global num_update
    global best_mean_reward

    if (num_update + 1) % 10 == 0:
        _, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(y) > 0:
            mean_reward = np.mean(y[-10:])
            update_model = mean_reward > best_mean_reward
            if update_model:
                best_mean_reward = mean_reward
                # _locals['self'].save('breakout_model')

            print('time: {}, num_update: {}, mean: {:.2f}, best_mean: {:.2f}, model_update: {}'.format(
                datetime.datetime.now(pytz.timezone('Asia/Tokyo')),
               num_update, mean_reward, best_mean_reward, update_model))
    num_update += 1
    return True