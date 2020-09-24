import gym
import os
import numpy as np
import datetime
import pytz
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.results_plotter import load_results, ts2xy

# グローバル変数
num_update = 0 # 更新数
best_mean_reward = -np.inf # ベスト平均報酬

# コールバック関数の実装 (2)
def callback(_locals, _globals):
   global num_update
   global best_mean_reward

   # 100更新毎の処理 (3)
   if (num_update + 1) % 100 == 0:
       # 報酬配列の取得 (4)
       _, y = ts2xy(load_results(log_dir), 'timesteps')
       if len(y) > 0:
           # 平均報酬がベスト平均報酬以上の時はモデルを保存 (5)
           mean_reward = np.mean(y[-100:])
           update_model = mean_reward > best_mean_reward
           if update_model:
               best_mean_reward = mean_reward
            #    _locals['self'].save('best_model')

           # ログ
           print("time: {}, num_update: {}, mean: {:.2f}, best_mean: {:.2f}, model_update: {}".format(
               datetime.datetime.now(pytz.timezone('Asia/Tokyo')),
               num_update, mean_reward, best_mean_reward, update_model))
   num_update += 1
   return True

# ログフォルダの生成
log_dir = './logs/'
os.makedirs(log_dir, exist_ok=True)

# 環境の生成
env = gym.make('CartPole-v1')
env = Monitor(env, log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])

# モデルの生成
model = PPO2('MlpPolicy', env, verbose=0)

# モデルの訓練 (1)
model.learn(total_timesteps=100000, callback=callback)