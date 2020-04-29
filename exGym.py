import gym
from gym.envs.registration import register
import numpy as np
import random

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={
        'map_name': '4x4',
        'is_slippery': False
    }
)

env = gym.make("FrozenLake-v3")

Q = np.zeros([env.observation_space.n, env.action_space.n])

def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return random.choice(indices)

num_episodes = 1000
for i in range(num_episodes):
    state = env.reset()
    done=None
    while not done:
        action = rargmax(Q[state,:])
        new_state, reward, done, _ =env.step(action)
        Q[state,action] = reward + np.max(Q[new_state,:])