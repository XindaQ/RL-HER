import numpy as np

class BitFlipEnv:
    # this is the flip game envs
    def __init__(self, size, shaped_reward=False, dynamic=False):
        # Initialize env params
        self.shaped_reward = shaped_reward          # flag for shaped reward
        self.steps_done = None                      # flag for done of one episode
        self.max_steps = 200                        # the length for each episode
        self.size = size                            # the number of card in the flip game

        # Initialize the state and target
        self.state = np.array([])                   # get the state             
        self.target = np.array([])                  # get the target, which has same length with state

        # Dynamic goal settings
        self.dynamic = dynamic                      # flag for the dynamic HER or not 


    def step(self, action):
        """
        :param action: an int number between 0 and (size - 1) to flip in self.state
        :return: the new state, the reward of making that action (-1: not final, 0: final) and the 'done' bit
        """
        self.state[action] = 1 - self.state[action] # flip the action bit
        self.steps_done += 1                        # count the step

        next_state = np.copy(self.state)            # get the copied s'

        if all(self.state == self.target):          # return true if every element is true for iteration        
            # New state is the target
            reward = 0.0
            done = True                             # close the eposide
        else:
            if self.shaped_reward:                                                  # get the shaped reward
                # Shaped reward: the distance between state and target
                reward = -np.sum(self.state != self.target, dtype=np.float32)       # the count of diff bits
            else:
                reward = -1.0                                                       # the unshaped reward
            # Check if run is done
            if self.steps_done < self.max_steps:                                    # also terminate if too long
                done = False
            else:
                done = True
        
        # set a simple changing references of the targets in the dynamic case we want to study 
        if self.dynamic:
            self.target = 1 - self.target                                           # change the target to oppo if dynamic

        return next_state, reward, done                     # return the s'. R, D

    def reset(self):
        """
        Resets the new bit-flip environment
        :return: an initialized environment state
        """
        self.steps_done = 0                                                     # reset the env at each eposide begining
        self.state = np.random.randint(2, size=self.size)                       # get random init state
        self.target = np.random.randint(2, size=self.size)                      # get random target

        # If target and initial state are equal, regenerate the target.
        while all(self.state == self.target):                                   # make them diff
            self.target = np.random.randint(2, size=self.size)

        return np.copy(self.state)                                              # output the copy of init state
