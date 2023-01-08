import numpy as np

rng = np.random.default_rng()

class TwoArmsBanditIndependant:
  
  def __init__(self, p0, p1):
    self.p0 = p0
    self.p1= p1

  def pull(self, a):
##    a = tf.get_static_value(a)
    if a == 0:
      reward = rng.random((1,), dtype=np.float32) < self.p0
    elif a == 1:
      reward = rng.random((1,), dtype=np.float32) < self.p1
    else:
      raise 
    return reward

class TwoArmsBanditCorrelated:
  
  def __init__(self, p):
    self.p = p

  def pull(self, a):
##    a = tf.get_static_value(a)
    if a == 0:
      reward = rng.random((1,), dtype=np.float32) < self.p
    elif a == 1:
      reward = rng.random((1,), dtype=np.float32) < 1 - self.p
    else:
      raise 
    return reward

class Episode:

  def __init__(self, task_label):
    # how many trials in the episode
    #self.nTrials = np.random.randint(50, 101)
    self.nTrials = 100

    if task_label == 'independant':
      probabilities = np.random.uniform(0, 1, 2)
      self.p0, self.p1 = probabilities[0], probabilities[1]
      self.task = TwoArmsBanditIndependant(self.p0,self.p1)
    
    if task_label == 'correlated':
      self.p = np.random.uniform(0, 1, 1)
        
# an even simpler task: always press right arm
#      self.p = 0.9
      self.task = TwoArmsBanditCorrelated(self.p)