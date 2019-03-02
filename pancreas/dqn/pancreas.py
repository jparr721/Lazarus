import gym
from gym import spaces
from gym.envs.registration import register
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from agent import Agent
register(
        id='simglucose-adolescent2-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adolescent#002'}
        )

# Initialize our simulation
env = gym.make('simglucose-adolescent2-v0')
env.seed(0)
print('State shape: {}'.format(env.observation_space.shape[0]))
print('Number of actions: {}'.format(env.action_space.shape[0]))

agent = Agent(state_size=1, action_size=1, seed=0)

# Watch the untrained agent
state = env.reset()
for i in range(200):
    action = agent.act(np.array(state[0]))
    env.render()
    state, reward, done, _ = env.step(action)
    if done:
        break
env.close()


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
        pass

