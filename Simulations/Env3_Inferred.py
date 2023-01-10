import numpy as np
from Env_Model import *

STIMU_L = 0
CENTER = 1
STIMU_R = 2
FIXATION = np.array([0,1,0])

class Inferred(Env):
    
    def __init__(self):
        super().__init__(nb_actions=3, nb_obs=3, action_per_trial = 2)
    
    def reset(self):
        self.timestep = -1
        self.done = False
        self.rewarded_stimu = np.random.choice([STIMU_L,STIMU_R])
        self.reward = 0
        return FIXATION
    
    def test_reset(self):
        return self.reset()
    
    def step1(self, action):
        self.reward = 0
        if action != 1:
            self.reward = -1
        self.target = np.random.choice([STIMU_L,STIMU_R])
        obs = [0,0,0]
        obs[self.target] = 1
        return np.array(obs), 0, False, self.timestep
    
    def step2(self, action):
        if self.reward != -1:
            if action == 1:
                self.reward = -1
            elif action == self.rewarded_stimu:
                self.reward = 1
                
        return FIXATION, self.reward, self.timestep==9, self.timestep
    
    def trial(self, action):
        self.timestep += 1
        if self.timestep % 2 == 0:
            return self.step1(action)
        else: return self.step2(action)
