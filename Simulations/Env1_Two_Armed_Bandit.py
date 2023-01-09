import numpy as np
import random
from random import choice
from Env_Model import *

class Two_Armed_Bandit(Env):
    def __init__(self):
        super().__init__(nb_actions=3)
        self.reset()
    
    #set the baseline probability of reward for action a.
    #sampling from a uniform Benoulli distribution and held fix for the entire episode
    def reset(self):
        self.timestep = 0 
        self.nb_al = 0
        self.nb_ar = 0
        self.nb = [0,0]
        self.prev_action = -1
        self. timestepmax = np.random.randint(50,100)
        #print("timestepmax",self.timestepmax)
        variance = np.random.uniform(0,.1)
        self.baseline_prob = [variance,0.5-variance]
        #print("baseline prob",self.baseline_prob)
        
    def test_reset(self):
        self.reset()

    ##get action from the network
    def pullArm(self,action):
        self.timestep += 1    
        p_action_init = self.baseline_prob[action]
        p_action = p_action_init   
        #print("action",action)
        #print("prev_actions",prev_actions)        
        if action == 0 and self.prev_action != -1 :
            if self.prev_action == 0: 
                self.nb_al+=1
                p_action = 1 - np.power((1-p_action),self.nb_al +1)
                #print("nb_al",self.nb_al)
            else:
                self.nb_al = 0 
                #print("nb_al",self.nb_al)
            reward = random.choices([1,0],weights=[p_action,1-p_action])[0]
            #print("reward_proba",p_action,"reward",reward)
        elif action == 1 and self.prev_action != -1: 
            if self.prev_action == 1: 
                self.nb_ar+=1
                p_action = 1 - np.power((1-p_action),self.nb_ar +1)
                #print("nb_ar",self.nb_ar)
            else:
                self.nb_ar = 0 
                #print("nb_ar",self.nb_ar)
            reward = random.choices([1,0],weights=[p_action,1-p_action])[0]
            #print("reward_proba",p_action,"reward",reward)
        else:
            if action == 0 : 
                self.nb_al+=1
                reward = random.choices([1,0],weights=[p_action,1-p_action])[0]
                #print("reward_proba",p_action,"reward",reward)
            else :
                self.nb_ar+=1
                reward = random.choices([1,0],weights=[p_action,1-p_action])[0]
                #print("reward_proba",p_action,"reward",reward)
        #print("timestep",self.timestep,"action",action)
        if self.timestep > self.timestepmax: 
            #print("nombre",self.nb_al,self.nb_ar)
            done = True
        else: done = False
        
        self.prev_action = action
        return  reward,done,self.timestep