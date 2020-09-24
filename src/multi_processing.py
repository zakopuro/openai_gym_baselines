import gym
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from stable_baselines.common import set_global_seeds

ENV_ID = 'CartPole-v1'
NUM_ENV = 4

def make_env(env_id, rank, seed=0):
   def _init():
       env = gym.make(env_id)
       env.seed(seed + rank)
       return env
   set_global_seeds(seed)
   return _init

def main():
    train_env = SubprocVecEnv([make_env(ENV_ID, i) for i in range(NUM_ENV)])
    model = PPO2('MlpPolicy', train_env, verbose=1)
    model.learn(total_timesteps=10000)
    test_env = DummyVecEnv([lambda: gym.make(ENV_ID)])

    state = test_env.reset()
    for i in range(200):
        test_env.render()
        action, _ = model.predict(state)
        state, rewards, done, info = test_env.step(action)

        # エピソード完了
        if done:
            break

    # 環境のクローズ
    test_env.close()

# メインの実装 (4)
if __name__ == "__main__":
    main()