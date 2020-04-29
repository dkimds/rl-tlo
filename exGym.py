import gym
from gym.envs.registration import register

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={
        'map_name': '4x4',
        'is_slippery': False
    }
)

env = gym.make("FrozenLake-v3")

num_episodes = 1000
for i in range(num_episodes):
    env.reset()
    done=None
    while not done:
        #env.render()
        action = env.action_space.sample()
        state, reward, done, _ =env.step(action)
    env.render()