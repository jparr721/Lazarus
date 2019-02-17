import gym
from queue import Queue


class RandomAgent:
    '''
    Random agents perform random actions inside of our environment. This is
    to facilitate the establishing of a baseline. This allows us to have
    something to go off of in our reinforcement learning algorithm.

    Parameters:
    -----------
    env_name: Name of the OpenAI gym environment to play in
    max_eps: The maximum number of episodes to run the agent for.
    '''
    def __init__(self, env_name, max_eps):
        self.env = gym.make(env_name)
        self.max_episodes = max_eps
        self.global_moving_average_reward = 0
        self.res_queue = Queue()

    def run(self):
        reward_avg = 0
        for episode in range(self.max_episodes):
            done = False
            self.env.reset()
            reward_sum = .0
            steps = 0
            while not done:
                # Sample randomly from the action space and step
                _, reward, done, _ = self.env.step(
                        self.env.action_space.sample())
                steps += 1
                reward_sum += reward
            self.global_moving_average_reward = record(episode,
                                                       reward_sum,
                                                       0,
                                                       self.global_moving_average_reward,
                                                       self.res_queue, 0, steps)
            reward_avg += reward_sum
        final_avg = reward_avg / float(self.max_episodes)
        print('Average score for {} episodes: {}'.format(self.max_episodes, final_avg))
        return final_avg
