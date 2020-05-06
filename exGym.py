import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf

def one_hot_encoder(state_in):
    return np.identity(16)[state_in:state_in + 1]

env = gym.make("FrozenLake-v0")
input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.7

X = tf.placeholder(shape = [1, input_size], dtype = tf.float32)
W = tf.Variable(tf.random_uniform([input_size, output_size], 0, 0.01))

Qpredict = tf.matmul(X, W)
Y=tf.placeholder(shape =[1, output_size], dtype=tf.float32)
loss= tf.reduce_sum(tf.square(Y-Qpredict))
train=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

num_episodes = 1000
e = 0.2
r = 0.9 # discount rate
rList = []
successRate = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_episodes):   # num_episodes회 학습하고, 업데이트 
        state = env.reset()         # 리셋
        total_reward = 0            # 그래프 그리기용(성공: 1, 실패: 0)
        done = None
        local_loss = []
        while not done:
            Qs = sess.run(Qpredict, feed_dict = {X:one_hot_encoder(state)})
            rand = random.random()
            if (rand < e / (i/50 + 10)):
                action = env.action_space.sample() # 완전 랜덤 선택
            else:
                action = np.argmax(Qs)
            new_state, reward, done, _ =env.step(action)
            if done:
                Qs[0, action] = reward
            else:
                new_Qs = sess.run(Qpredict, feed_dict = {X: one_hot_encoder(new_state)})
                Qs[0:action]= reward + r*np.argmax(new_Qs)
            sess.run(train, feed_dict={X:one_hot_encoder(state), Y:Qs})
            total_reward += reward
            state = new_state
        rList.append(total_reward)                 # 이번 게임에서 보상이 0인지 1인지
        successRate.append(sum(rList)/(i+1))       # 지금까지 성공률 
env.close()

print("Final Qs")
print(Qs)
print("Success Rate : ", successRate[-1])
plt.plot(range(len(successRate)), successRate) # 성공률 그래프
plt.plot(range(len(rList)), rList)             # reward 그래프
plt.show()