import gym
from gym.registration import register


class Simulation:
    def __init__(self):
        self._env = None

    def register_environment(self, env_name, patient):
        register(
                id=env_name,
                entry_point='simglucose.envs:T1DSimEnv',
                kwargs={'patient_name': patient}
                )
        self._env = gym.make(env_name)

    def run(self, reward_function):
        reward = 1
