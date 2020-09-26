import gym
import time
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor
from util import callback, log_dir
from baselines.common.atari_wrappers import *

ENV_ID = 'BreakoutNoFrameskip-v0'
NUM_ENV = 8

def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = FireResetEnv(env)
        env = WarpFrame(env)
        env = ScaledFloatFrame(env)
        env = ClipRewardEnv(env)
        env = EpisodicLifeEnv(env)

        if rank == 0:
            env = Monitor(env, log_dir, allow_early_resets=True)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

def main():
    train_env = DummyVecEnv([make_env(ENV_ID, i) for i in range(NUM_ENV)])
    model = PPO2('CnnPolicy', train_env, verbose=0)
    model.learn(total_timesteps=1280000, callback=callback)

    test_env = DummyVecEnv([make_env(ENV_ID, 9)])

    state = test_env.reset()
    total_reward = 0
    while True:
        test_env.render()
        time.sleep(1/60)
        action, _ = model.predict(state)
        state, reward, done, info = test_env.step(action)

        total_reward += reward[0]
        if done:
            print('reward:', total_reward)
            state = test_env.reset()
            total_reward = 0

if __name__ == "__main__":
    main()