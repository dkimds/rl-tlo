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

Q = np.zeros([env.observation_space.n, env.action_space.n]) # (16, 4): 4*4 map + 상하좌우 4개

num_episodes = 1000
rList = []
successRate = []

e = 0.1 # exploit & exploration
r = 0.9 # discount rate

def rargmax(vector):
    m = np.amax(vector)                     # Return the maximum of an array or maximum along an axis (0 or 1)
    indices = np.nonzero(vector == m)[0]    # np.nonzero(True/False vector) -> 최대값인 요소들만 걸러내
    return random.choice(indices)           # 랜덤으로 하나 선택

for i in range(num_episodes):   # num_episodes회 학습하고, 업데이트 
    state = env.reset()         # 리셋
    total_reward = 0            # 그래프 그리기용(성공: 1, 실패: 0)
    done = None
    while not done:
        rand = random.random()
        if (rand < e):
            action = env.action_space.sample() # 완전 랜덤 선택
        else:
            action = rargmax(Q[state,:])
        new_state, reward, done, _ =env.step(action)
        Q[state,action] = reward + r * np.max(Q[new_state,:])
        total_reward += reward
        state = new_state
    rList.append(total_reward)                 # 이번 게임에서 보상이 0인지 1인지
    successRate.append(sum(rList)/(i+1))       # 지금까지 성공률 

print("Final Q-Table")
print(Q)
print("Success Rate : ", successRate[-1])
plt.plot(range(len(successRate)), successRate) # 성공률 그래프
plt.plot(range(len(rList)), rList)             # reward 그래프
plt.show()