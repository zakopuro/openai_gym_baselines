import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from stable_baselines.common import set_global_seeds


ENV_ID = 'CartPole-v1'
NUM_ENVS = [1,2,4,8,16]
NUM_EXPERIMENTS = 3
NUM_STEPS = 5000
NUM_EPISODES = 20

def make_env(env_id,rank,seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init


def evaluate(model, env, num_episodes=100):
    all_epsode_reward = []
    for i in range(num_episodes):
        episode_reward = []
        done = False
        state = env.reset()
        while not done:
            action,_ = model.predict(state)
            state,reward,done,info = env.step(action)
            episode_reward.append(reward)
        all_epsode_reward.append(sum(episode_reward))

    return np.mean(all_epsode_reward)


def main():
    reward_averages = []
    reward_std = []
    training_times = []
    total_env = 0
    for num_envs in NUM_ENVS:
        total_env += num_envs
        print(f'process:{num_envs}')

        if num_envs == 1:
            train_env = DummyVecEnv([lambda : gym.make(ENV_ID)])
        else:
            train_env = SubprocVecEnv([make_env(ENV_ID, i+total_env) for i in range(num_envs)], start_method='spawn')

        eval_env = DummyVecEnv([lambda: gym.make(ENV_ID)])

        rewards = []
        times = []
        for experiment in range(NUM_EXPERIMENTS):
            train_env.reset()
            model = PPO2('MlpPolicy', train_env,verbose=0)
            start = time.time()
            model.learn(total_timesteps=NUM_STEPS)
            times.append(time.time() - start)

            mean_reward = evaluate(model, eval_env, num_episodes=NUM_EPISODES)
            rewards.append(mean_reward)

        train_env.close()
        eval_env.close()

        reward_averages.append(np.mean(rewards)) # 平均報酬
        reward_std.append(np.std(rewards)) # 標準偏差
        training_times.append(np.mean(times)) # 学習速度

        # プロセスと平均報酬
    plt.errorbar(NUM_ENVS, reward_averages, yerr=reward_std, capsize=2)
    plt.xlabel('number of envs')
    plt.ylabel('mean reward')
    plt.savefig('./data/process_mean.png')
    plt.show()

    # # プロセスと秒間ステップ数
    training_steps_per_second = [NUM_STEPS / t for t in training_times]
    plt.bar(range(len(NUM_ENVS)), training_steps_per_second)
    plt.xticks(range(len(NUM_ENVS)), NUM_ENVS)
    plt.xlabel('number of envs')
    plt.ylabel('steps per second')
    plt.savefig('./data/process_step.png')
    plt.show()

# メインの実装
if __name__ == "__main__":
    main()