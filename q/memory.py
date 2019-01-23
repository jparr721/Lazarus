# Copyright 2018 Grand Valley State University DEN Lab. All Rights Reserved
#==============================================================================


class Memory:
    '''
    The memory class stores all the results of the action of the agent
    in the system. It also handles the retreival of the actions. We can
    also utilize it to batch train the network

    Parameters
    ----------
    max_memory - The maximum number of (state, action, reward, next_state)
                 tuples that we can hold
    '''
    def __init__(self, max_memory):
        self._max_memory = max_memory
        sel._samples = []

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, number_of_samples):
        pass
