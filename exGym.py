import gym
env = gym.make("FrozenLake-v0")

num_episodes = 1000
for i in range(num_episodes):
    state = env.reset()
    done=None
    while not done:
        #env.render()
        action = env.action_space.sample()
        state, reward, done, _ =env.step(action)
    env.render()
    print(state)

env.close()