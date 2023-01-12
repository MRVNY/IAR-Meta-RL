import numpy as np

STIMU_L = 0
CENTER = 1
STIMU_R = 2
FIXATION = np.array([0,1,0])

class Env():
    
    def __init__(self, nb_actions, nb_obs=0, entropy_var=False, action_per_trial=1):
        self.nb_obs = nb_obs
        self.entropy_var = entropy_var
        self.action_per_trial = action_per_trial
        self.nb_actions = nb_actions
        