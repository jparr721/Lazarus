#!/usr/bin/env python3

import gym
from gym.envs.registration import register
import numpy as np


def custom_reward(BG_last_hour):
    if BG_last_hour[-1] > 180:
        return -2
    elif BG_last_hour[-1] < 70:
        return -4
    else:
        return 2


register(
        id='simglucose-adolescent2-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adolescent#002',
                'reward_fun': custom_reward}
        )
env = gym.make('simglucose-adolescent2-v0')

bin_size = 20


def normalize_to_int(bin_size, value):
    return int(value) % bin_size


def q_train(eps=1000, gamma=0.9, eta=0.8, max_t=1000):
    env.reset()

    Q = np.zeros([bin_size, 2])

    # Params
    reward_list = []

    for i in range(eps):
        print(f'Episode {i} out of {eps} total')
        state = env.reset()
        state = normalize_to_int(bin_size, state[0])
        r_all = 0
        for j in range(max_t):
            action = np.argmax(
                    Q[state, :] + np.random.randn(1, 2) * (1./(i + 1)))

            # Get new state & reward
            next_state, reward, done, _ = env.step(action)
            next_state = normalize_to_int(bin_size, next_state[0])

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
    print('Final Q Table Vals: \n{}'.format(Q))

    return Q


def multi_train():
    pass


def q_test(Q, eps=100):
    total_epochs, total_penalties = 0, 0

    for i in range(eps):
        print(f'Episode {i} out of {eps} total')
        state = env.reset()
        state = int(state[0]) % 20
        epochs, penalties, reward = 0, 0, 0
        done = False

        while not done:
            action = np.argmax(Q[state])

            state, reward, done, _ = env.step(action)
            state = int(state[0]) % 20

            if reward == -10:
                penalties += 1
            epochs += 1
        total_penalties += penalties
        total_epochs += epochs

    print(f'Results after {eps} eps')
    print(f'Average steps per ep: {total_epochs/eps}')
    print(f'Average penalties per ep: {total_penalties/eps}')
    env.render()


if __name__ == '__main__':
    Q = q_train()
    q_test(Q)
