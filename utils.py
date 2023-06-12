from collections import namedtuple
import random
import numpy as np

# namedtuple, a type of container dictionary
# contain keys and the mapped / hashed value 
# can be access from keys or iterations, just as a table, can be stored by: S = Transition(s,a,s',R)
# can be get by: S.state or S[0]
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
  # design the replay buffer for the experience storage and manipulate
    def __init__(self, capacity):   
        # use a list to store the experience
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:                            # get the space if there is still some
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)                  # use the dictionary for info and put it in list memory
        self.position = (self.position + 1) % self.capacity             # use a position pointer to fill and update the list

    def sample(self, batch_size):
        # use random.sample to sample the transition, sample n samples from the list (without replacement)
        return random.sample(self.memory, batch_size)

    def __len__(self):
        # get the current length of the replay buffer
        return len(self.memory)


def moving_average(data_set, periods=10):
    # this is just a moving average function, with a specific period
    weights = np.ones(periods) / periods
    # use np.convolve to get the convolve of two one-dim sequence, and get the the moving average 
    # mode=valid: M, N -> max(M,N) - min(M,N) + 1, the convolve is calculate where the two sequence can cover completely
    # this will output a different size output
    averaged = np.convolve(data_set, weights, mode='valid')
    
    # this will generate the average value before the one period is collected
    pre_conv = []
    for i in range(1, periods):
        pre_conv.append(np.mean(data_set[:i]))
    
    # combine them will output a averaged sequence with the same length, which is nicer.
    averaged = np.concatenate([pre_conv, averaged])
    return averaged
