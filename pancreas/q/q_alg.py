#!/usr/bin/env python3

import gym
from gym.envs.registration import register
import numpy as np


def custom_reward(BG_last_hour):
    if BG_last_hour[-1] > 180:
        return -1
    elif BG_last_hour[-1] < 70:
        return -4
    else:
        return 1


register(
        id='simglucose-adolescent2-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adolescent#002',
                'reward_fun': custom_reward}
        )
env = gym.make('simglucose-adolescent2-v0')

bin_size = 4


def normalize_to_int(bin_size, value):
    return int(value) % bin_size


def q_train(eps=1000, gamma=0.9, eta=0.8, max_t=1000):
    env.reset()

    Q = np.zeros([bin_size, 2])
    print(f'max_t is {max_t}')

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
        if i % 10 == 0:
            env.render()
        state = env.reset()
        state = int(state[0]) % bin_size
        epochs, penalties, reward = 0, 0, 0
        done = False

        while not done:
            action = np.argmax(Q[state])

            state, reward, done, _ = env.step(action)
            state = int(state[0]) % bin_size

            if reward == -10:
                penalties += 1
            epochs += 1
        total_penalties += penalties
        total_epochs += epochs

    print(f'Results after {eps} eps')
    print(f'Average steps per ep: {total_epochs/eps}')
    print(f'Average penalties per ep: {total_penalties/eps}')


if __name__ == '__main__':
    Q = q_train(max_t=100000000)
    # Q
    '''
    79 reward
    Q = [[ 2.64343710, 0.]
         [ 1.76128000, 3000.03584127]
         [ 3000.03722093,-2.46272000]
         [ 3000.03502982, 0.]
         [ 0., 3000.03723685]
         [ 3000.03737494, 0.]
         [ 3000.03452541, 1.02400000]
         [-2.40000000, 3000.03719000]
         [ 3000.03572769, .8]
         [ 0., 3000.03461056]
         [ 3000.03744561, 0.]
         [ 0., 3000.03717848]
         [ 3000.03783949, 0.]
         [ 3000.03531968,-2.62400000]
         [ 0., 3000.03517301]
         [ 0., 3000.03479427]
         [ 3000.03694965, 0.]
         [ 3000.03852017,-2.91328000]
         [ 3000.03887275, 0.]
         [ .8,  3000.03678427]]
 '''
    q_test(Q)
