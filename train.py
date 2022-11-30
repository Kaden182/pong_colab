#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#-*- coding: utf-8 -*-

# 检查版本
import gym
import parl
import paddle
#assert paddle.__version__ == "2.2.0", "[Version WARNING] please try `pip install paddlepaddle==2.2.0`"
assert parl.__version__ == "2.0.3", "[Version WARNING] please try `pip install parl==2.0.3`"
assert gym.__version__ == "0.18.0", "[Version WARNING] please try `pip install gym==0.18.0`"

import os
import gym
import numpy as np
import parl
import time



from agent import Agent
from model import Model
from policy_gradient import PolicyGradient
from parl.utils import logger
#from memory_profiler import profile

LEARNING_RATE = 5e-4


# 训练一个episode,一轮结束退出（某一方先获得20分结束一轮）
#@profile
def run_train_episode(agent, env):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        obs = preprocess(obs)  # from shape (210, 160, 3) to (100800,)，把图像拍平
        obs_list.append(obs)
        action = agent.sample(obs) #通过前向传播得到的概率选择动作
        action_list.append(action)

        obs, reward, done, info = env.step(action)
        reward_list.append(reward)
        env.render()
            
        if done:
            break
    return obs_list, action_list, reward_list #返回这一轮/episode（包含多个img）训练得到的：（观察空间列表，动作列表，奖励列表）


# 评估 agent, 跑 5 轮，总reward求平均
def run_evaluate_episodes(agent, env, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            obs = preprocess(obs)  # from shape (210, 160, 3) to (100800,)
            action = agent.predict(obs)
            obs, reward, isOver, _ = env.step(action) #reward取值： -1 ，0 ，1 ，某方先到20，则一轮结束
            episode_reward += reward
            if render:
                env.render()
            if isOver:
                break
        eval_reward.append(episode_reward) #5轮的总分
    return np.mean(eval_reward) #5轮的均分


def preprocess(image):
    """ 预处理 210x160x3 uint8 frame into 6400 (80x80) 1维 float vector """
    image = image[35:195]  # 裁剪
    image = image[::2, ::2, 0]  # 下采样，缩放2倍
    image[image == 144] = 0  # 擦除背景 (background type 1)
    image[image == 109] = 0  # 擦除背景 (background type 2)
    image[image != 0] = 1  # 转为灰度图，除了黑色外其他都是白色
    return image.astype(np.float).ravel()


def calc_reward_to_go(reward_list, gamma=0.99):
    """calculate discounted reward"""
    reward_arr = np.array(reward_list)
    for i in range(len(reward_arr) - 2, -1, -1):
        # G_t = r_t + γ·r_t+1 + ... = r_t + γ·G_t+1
        reward_arr[i] += gamma * reward_arr[i + 1]
    # normalize episode rewards
    reward_arr -= np.mean(reward_arr)
    reward_arr /= np.std(reward_arr)
    return reward_arr


def main():
    env = gym.make('Pong-v0')   
    obs_dim = 80 * 80
    act_dim = env.action_space.n
    logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    # 根据parl框架构建agent
    model = Model(obs_dim=obs_dim, act_dim=act_dim)
    alg = PolicyGradient(model, lr=LEARNING_RATE)
    agent = Agent(alg)

    """
    # 加载模型并评估,5轮评估，取平均
    if os.path.exists('./models/model_2000.ckpt'):
        print("load model and assess:")
        agent.restore('./models/model_2000.ckpt')
        total_reward=run_evaluate_episodes(agent, env, render=True) # 评估 agent, 跑 5 轮，总reward求平均
        logger.info('Test reward: {}'.format(total_reward))
        env.close()
        exit()
    """
    
    agent.restore('./models/model_2000.ckpt') #加载一下1000轮版本的model

    rounds=100 # 训练100轮
    #训练1000组，每组训练rounds轮
    for j in range(1,1001):
        total_episode_loss=0#统计10轮的总损失值
        for i in range(rounds):#训练rounds轮,保存一次模型
            obs_list, action_list, reward_list = run_train_episode(agent, env)#跑一轮
           
            #将每轮训练得到的（观察空间列表，动作列表，奖励列表） np数组化
            batch_obs = np.array(obs_list)
            batch_action = np.array(action_list)
            batch_reward = calc_reward_to_go(reward_list) #基于决策算法的reward计算机制, reward_list是一轮中每次交互得到的分数的列表，每个元素的取值为-1，0，1

            
            #获得一轮训练得到的损失函数
            one_episode_loss=agent.learn(batch_obs, batch_action, batch_reward)#拿这一轮的数据去训练
            #统计10轮的总损失值
            total_episode_loss += one_episode_loss
            
            
            #每训练10轮，打印下第i轮的日志
            if i % 10 == 0:
                eval_loss = total_episode_loss/10 # 计算10轮的平均损失值
                total_episode_loss=0 # 10轮的总损失值清0
                logger.info("Episode {}, Reward Sum {}, eval_loss {} .".format(i, sum(reward_list), eval_loss))
            """
            #训练100轮，评估一下模型
            if (i + 1) % 100 == 0:
                total_reward = run_evaluate_episodes(agent, env, render=True)#跑5轮取平均
                logger.info('Test reward: {}'.format(total_reward))
            """
        # save the parameters to ./model.ckpt
        agent.save('./models/model_'+str(2000+j*rounds)+'.ckpt') #每训练rounds轮，记录一下模型
    
    env.close()#关闭环境，否则报错：KeyError: (<weakref at 0x000001C96BE838B8; to 'Win32Window' at 0x000001C9A0475EF0>,)
    
if __name__ == '__main__':
    main()
