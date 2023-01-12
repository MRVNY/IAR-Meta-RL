import numpy as np
from random import choice
from Env_Model import *

L = 0
R = 1

class Two_Armed_Bandit(Env):
    def __init__(self):
        super().__init__(nb_actions=2)
        self.actionL_history = []
        self.actionR_history = []
        self.rewardL_history = []
        self.rewardR_history = []
        self.held_out = True
        self.reset()
        
    def get_random_p0(self):
        p0 = np.random.uniform(0,.5)
        if (0.1<p0 < 0.2 )or (0.3 < p0 < 0.4) and self.held_out:
            return self.get_random_p0()
        return p0
    
    #set the baseline probability of reward for action a.
    #sampling from a uniform Benoulli distribution and held fix for the entire episode
    def reset(self):
        self.timestep = 0 
        
        self.last_seen_counter = [0,0]
        self.action_counter = [0,0]
        self.reward_counter = [0,0]
        self.action_proba = [0,0]
        self.last_seen = [0,0]
        
        self.timestepmax = np.random.randint(50,100)
        variance = self.get_random_p0()
        self.baseline_prob = [variance,0.5-variance]
        
    def test_reset(self):
        #self.held_out = False
        if len(self.actionL_history) > 1000:
            self.actionL_history = []
            self.actionR_history = []
            self.rewardL_history = []
            self.rewardR_history = [] 
        self.reset()

    ##get action from the network
    def trial(self,action):
        self.timestep += 1    
        
        if self.last_seen[action] == 0: 
            self.last_seen_counter[action] += 1
        # p_action = 1 - np.power((1-self.baseline_prob[action]),
        #                         self.action_counter[action]+1)
        
        p_action = 1 - np.power((1-self.baseline_prob[action]),
                                self.last_seen[action]+1)
        
        self.action_counter[action] += 1
        reward = int(np.random.rand() < p_action)
        self.reward_counter[action] += reward
        
        self.last_seen[action] = 1
        self.last_seen[1-action] += 1
        
        if self.timestep > self.timestepmax: 
            done = True
            self.actionL_history.append(self.action_counter[L])
            self.actionR_history.append(self.action_counter[R])
            self.rewardL_history.append(self.reward_counter[L])
            self.rewardR_history.append(self.reward_counter[R])
            
        else: 
            done = False
        
        return reward,done,self.timestep
    
    def get_abcd(self):
        return self.actionL_history, self.actionR_history, self.rewardL_history, self.rewardR_history