import gym
from gym.envs.registration import register
import numpy as np
import random
import matplotlib.pyplot as plt

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

num_episodes = 1000
rList = []
successRate = []

def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return random.choice(indices)

for i in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done=None
    while not done:
        action = rargmax(Q[state,:])
        new_state, reward, done, _ =env.step(action)
        Q[state,action] = reward + np.max(Q[new_state,:])
        total_reward += reward
        state = new_state
    rList.append(total_reward)
    successRate.append(sum(rList)/(i+1))

print("Final Q-Table")
print(Q)
print("Success Rate : ", successRate[-1])
plt.plot(range(len(successRate)), successRate) # 성공률 그래프
plt.plot(range(len(rList)), rList)             # reward 그래프
plt.show()