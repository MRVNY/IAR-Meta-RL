import numpy as np

STIMU_1 = 0
STIMU_2 = 1

class Monkey2():  
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.timestep = 0
        self.done = False
        self.rewarded_stimu = np.random.choice([STIMU_1,STIMU_2])
        self.rewards = [0,0]
        self.rewards[self.rewarded_stimu] = 1
        
    def trial(self, action):
        self.timestep +=  1
        return self.rewards[action], self.timestep==5, self.timestep