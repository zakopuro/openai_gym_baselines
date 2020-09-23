import gym

env = gym.make('CartPole-v1')
state = env.reset()

while True:
    env.render()
    action = env.action_space.sample()
    state,reward,done,info = env.step(action)
    print('reward:',reward)

    if done:
        print('done')
        break

env.close()