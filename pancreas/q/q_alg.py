#!/usr/bin/env python3

import gym
from gym.envs.registration import register
import numpy as np

register(
        id='simglucose-adolescent2-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adolescent#002'}
        )
env = gym.make('simglucose-adolescent2-v0')


def q_train(eps=2000, gamma=0.8, eta=0.8, max_t=1000):
    env.reset()

    Q = np.zeros([359, 2])

    # Params
    reward_list = []

    for i in range(eps):
        print(f'Episode {i} out of {eps} total')
        state = env.reset()
        state = int(state[0])
        r_all = 0
        for j in range(max_t):
            action = np.argmax(
                    Q[int(state), :] + np.random.randn(1, 2) * (1./(i + 1)))

            # Get new state & reward
            next_state, reward, done, _ = env.step(action)
            next_state = int(next_state[0])

            # Update Q table
            Q[state, action] = Q[state, action] + \
                eta * (
                    reward + gamma * np.max(
                        Q[next_state, :] - Q[state, action]))
            r_all += reward
            state = next_state
            if done:
                break

        reward_list.append(r_all)

    env.render()

    print('Reward sum for all eps: {}'.format(sum(reward_list)/eps))
    print('Final Q Table Vals: {}'.format(Q))

    return Q


def q_test(Q, eps=100):
    total_epochs, total_penalties = 0, 0

    for i in range(eps):
        print(f'Episode {i} out of {eps} total')
        state = env.reset()
        state = int(state[0])
        epochs, penalties, reward = 0, 0, 0
        done = False

        while not done:
            action = np.argmax(Q[state])

            state, reward, done, _ = env.step(action)

            if reward == -10:
                penalties += 1
            epochs += 1
        total_penalties += penalties
        total_epochs += epochs

    env.render()

    print(f'Results after {eps} eps')
    print(f'Average steps per ep: {total_epochs/eps}')
    print(f'Average penalties per ep: {total_penalties/eps}')


if __name__ == '__main__':
    q_train()
    q_test()
