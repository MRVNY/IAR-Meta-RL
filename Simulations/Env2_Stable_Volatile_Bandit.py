import numpy as np
from Env_Model import *

STABLE_FIRST = 0
VOLATILE_FIRST = 1

HIGH_PROBA = 0.8
LOW_PROBA = 0.2

STABLE_PROBA = [0.25,0.75]

R = 1

class Stable_Volatile_Bandit(Env):
    
    def __init__(self, nb_episodes=40000, nb_trials=400):
        super().__init__(nb_actions=3, entropy_var=True)
        self.nb_episodes = nb_episodes
        self.nb_trials = nb_trials
        self.beta_e = 1
        self.training = True
        self.state = np.random.choice([STABLE_FIRST,VOLATILE_FIRST])
        self.reset()
        self.r_history = []
        
    def test_update_r(self):
        if self.state == STABLE_FIRST:
            if self.timeStep < 100:
                self.r = np.random.choice(STABLE_PROBA)
            elif self.timeStep % 25 == 0:
                self.r = (HIGH_PROBA - LOW_PROBA) * np.random.rand + LOW_PROBA
                
        else:
            if self.timeStep < 100:
                if self.timeStep % 25 == 0:
                    self.r = (HIGH_PROBA - LOW_PROBA) * np.random.rand + LOW_PROBA
            else:
                self.r = np.random.choice(STABLE_PROBA)
                
        self.r_history.append(self.r)

    def train_update_r(self):
        if self.timeStep == 0:
            self.log_k = np.random.choice([-4.5,-3.5])
            self.k = np.exp(self.log_k)
            self.v = np.random.choice([0,0.2])
            
            if self.v == 0:
                self.r = np.random.choice(STABLE_PROBA)
            if self.v == 2:
                self.r = np.random.choice([0,1])
                
        else:
            if np.random.rand() < self.k:
                self.v = 0.2 - self.v
            if np.random.rand() < self.v:
                self.r = 1 - self.r
        
    def reset(self):
        self.beta_e = self.beta_e - 1/self.nb_episodes
        # self.episode += 1
        self.timeStep = 0
        self.state = 1 - self.state
        self.train_update_r()

    def test_reset(self):
        # self.episode += 1
        self.timeStep = 0
        self.state = 1 - self.state
        self.training = False
        self.test_update_r()
        
    def trial(self, action):
        if self.training:
            self.train_update_r()
        else:
            self.test_update_r()
        
        reward = R * (action==np.random.choice([0,1],p=[self.r,1-self.r]))
        
        self.timeStep += 1
        
        return reward, self.timeStep==self.nb_trials, self.timeStep-1
    
    def get_learning_rate(self):
        return 0
    
    def get_r_history(self):
        return self.r_history