import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from stable_baselines.common import set_global_seeds

# 定数
ENV_ID = 'CartPole-v1' # 環境ID
NUM_ENVS = [1, 2, 4, 8, 16] # 環境数
NUM_EXPERIMENTS = 3 # 実験回数
NUM_STEPS = 5000 # ステップ数
NUM_EPISODES = 20 # 評価エピソード数

# 環境を生成する関数
def make_env(env_id, rank, seed=0):
   def _init():
       env = gym.make(env_id)
       env.seed(seed + rank)
       return env
   set_global_seeds(seed)
   return _init

# 評価関数
def evaluate(model, env, num_episodes=100):
    # 任意回数のエピソードを実行
    all_episode_rewards = [] # 任意回数のエピソードの平均報酬
    for i in range(num_episodes):
        episode_rewards = [] # 1エピソードの平均報酬
        done = False
        state = env.reset()
        while not done:
            action, _ = model.predict(state)
            state, reward, done, info = env.step(action)
            episode_rewards.append(reward) # 1ステップの報酬の追加
        all_episode_rewards.append(sum(episode_rewards)) # 1エピソードの報酬の追加

    # 平均報酬の計算
    return np.mean(all_episode_rewards)

# メイン関数の定義
def main():
    # 訓練と評価
    reward_averages = []
    reward_std = []
    training_times = []
    total_env = 0
    num_steps = []
    for num_envs in NUM_ENVS:
        total_env += num_envs
        print('process:', num_envs)

        # 訓練環境の生成
        train_env = DummyVecEnv([make_env(ENV_ID, i+total_env) for i in range(num_envs)])

        # 評価環境の生成
        eval_env = DummyVecEnv([lambda: gym.make(ENV_ID)])

        # 10秒間のステップ数の計算
        train_env.reset()
        model = PPO2('MlpPolicy', train_env, verbose=0)
        start = time.time()
        model.learn(total_timesteps=NUM_STEPS)
        steps_per_sec = NUM_STEPS/(time.time() - start)
        num_steps.append(int(steps_per_sec*10))

        # 任意回数の実験実行
        rewards = []
        times = []
        for experiment in range(NUM_EXPERIMENTS):
            # 訓練
            train_env.reset()
            model = PPO2('MlpPolicy', train_env, verbose=0)
            start = time.time()
            model.learn(total_timesteps=int(steps_per_sec*10))
            times.append(time.time() - start) # 1訓練の時間の追加

            # 評価
            mean_reward = evaluate(model, eval_env, num_episodes=NUM_EPISODES)
            rewards.append(mean_reward) # 1評価の平均報酬の追加

        # 環境のクローズ
        train_env.close()
        eval_env.close()

        # 統計の収集
        reward_averages.append(np.mean(rewards)) # 平均報酬
        reward_std.append(np.std(rewards)) # 標準偏差
        training_times.append(np.mean(times)) # 学習速度

    # 環境数と平均報酬
    plt.errorbar(NUM_ENVS, reward_averages, yerr=reward_std, capsize=2)
    plt.xlabel('number of envs')
    plt.ylabel('mean reward')
    plt.show()

    # 環境数とステップ数
    plt.bar(range(len(NUM_ENVS)), num_steps)
    plt.xticks(range(len(NUM_ENVS)), NUM_ENVS)
    plt.xlabel('number of envs')
    plt.ylabel('number of steps')
    plt.show()

# メインの実装
if __name__ == "__main__":
    main()