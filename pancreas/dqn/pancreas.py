#!/usr/bin/env python3

import gym
from gym.envs.registration import register
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from simglucose.analysis.risk import risk_index
from agent import Agent
import sys


def custom_reward(BG_last_hour):
    if len(BG_last_hour) < 2:
        return 0
    else:
        _, _, risk_current = risk_index([BG_last_hour[-1]], 1)
        _, _, risk_prev = risk_index([BG_last_hour[-2]], 1)
        return risk_prev - risk_current


register(
        id='simglucose-adolescent2-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adolescent#002',
                'reward_fun': custom_reward}
        )

# Initialize our simulation
env = gym.make('simglucose-adolescent2-v0')
env.seed(0)
print('State shape: {}'.format(env.observation_space.shape[0]))
print('Number of actions: {}'.format(env.action_space.shape[0]))

agent = Agent(state_size=1, action_size=2, seed=0)

# Watch the untrained agent
# state = env.reset()
# for i in range(200):
#     action = agent.act(np.array(state[0]))
#     env.render()
#     state, reward, done, _ = env.step(action)
#     if done:
#         break
# env.close()


def dqn(n_episodes=2000,
        max_t=1000,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995):
    '''
    Do the deep q learning

    Parameters
    ----------
    n_episodes (int): The max number of training episodes
    max_t (int): Maximum number of timesteps per episode
    eps_start (float): starting value of the e-greedy action selection
    eps_end (float): min value of epsilon
    eps_decay (float): The factor to decrease epsilon
    '''

    # Scores from each episode
    scores = []
    # We want only the last 100 scores
    scores_window = deque(maxlen=100)
    eps = eps_start

    for i_ep in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(np.array(state[0]), eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(
            i_ep, np.mean(scores_window)), end="")
        if i_ep % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_ep, np.mean(scores_window)))
    torch.save(agent.local_network.state_dict(), 'checkpoint.pth')
    return scores


if __name__ == '__main__':
    train = sys.argv[1]
    if train:
        scores = dqn(2000, 100000000000, 10.0, 0.01, 0.95)

        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()

    else:
        # load the weights from file
        agent.local_network.load_state_dict(torch.load('checkpoint.pth'))

        for i in range(3):
            state = env.reset()
            for j in range(100000000000):
                env.render()
                action = agent.act(np.array(state[0]))
                state, reward, done, _ = env.step(action)
                if done:
                    break

        env.close()
