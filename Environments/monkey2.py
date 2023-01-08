import numpy as np

STIMU_1 = 0
STIMU_2 = 1

class Monkey2():  
    def __init__(self):
        self.rewarded_stimu = np.random.choice([STIMU_1,STIMU_2])
        self.reset()
        
    def reset(self):
        # The rewarded and unrewarded cues were randomly determined at the beginning of the episode and held fixed for the duration of the episode
        if np.random.rand() < 0.5:
            self.rewarded_stimu = np.random.choice([STIMU_1,STIMU_2])
        self.rewards = [0,0]
        self.rewards[self.rewarded_stimu] = 1
    
    # Concretely, after the central fixation cue on step 1, 
    # the agent was presented with one of the two stimulus cues (tabular one-hot vectors) on step 2, 
    # indicating that it must either produce an action left (aL) or an action right (aR) on step 3. 
    # The reward was then delivered on step 4, followed by the start of the next trial.

    def step1(self):
        
        step2()
        
    def step2(self):
        
        step3()
    
    def step3(self):
        
        step4()
        
    def step4(self):
        
        return
    
    def trial(self):
        