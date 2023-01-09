import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.stats import entropy
from scipy import signal
import scipy as sp

class Agent():
    def __init__(self, 
                 learning_rate, gamma, beta_v, beta_e,  #loss func
                 env, nb_trials, nb_episodes,       #train
                 path,                              #tfboard & ckpt
                 nb_hidden = 48
                 ) -> None:
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.beta_v = beta_v
        self.beta_e = beta_e
        
        self.env = env
        self.nb_trials = nb_trials
        self.nb_episodes = nb_episodes
        self.action_per_trial = env.action_per_trial
        self.entropy_var = env.entropy_var
        
        self.nb_inputs = env.nb_actions + env.nb_obs + 2
        self.nb_actions = env.nb_actions
        self.nb_hidden = nb_hidden
        
        self.path = path
        self.log_dir = path+'/logs/'
        self.ckpt_dir = path+'/ckpt/'
        self.test_dir = path+'/test/'
        
        self.model, self.optimizer = self.LSTM_Model()
        
        

    def LSTM_Model(self):
        inputs = layers.Input(shape=(self.nb_inputs))
        state_h = layers.Input(shape=(self.nb_hidden))
        state_c = layers.Input(shape=(self.nb_hidden))

        common, states = layers.LSTMCell(self.nb_hidden)(inputs, states=[state_h, state_c], training=True)
        action = layers.Dense(self.nb_actions, activation="softmax")(common)
        critic = layers.Dense(1)(common)

        model = keras.Model(inputs=[inputs,state_h,state_c], outputs=[action, critic, states], )
        optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        
        return model, optimizer

    def discount(self, x):
        return sp.signal.lfilter([1], [1, -self.gamma], x[::-1], axis=0)[::-1]

    def compute_loss(self, action_probs, values, rewards, entropy):
        """Computes the combined actor-critic loss."""
        
        bootstrap_n = tf.shape(rewards)[0]
        
        value_plus = np.append(values, bootstrap_n)
        rewards_plus = np.append(rewards, bootstrap_n)
        discounted_rewards = self.discount(rewards_plus)[:-1]
        advantages = rewards + self.gamma * value_plus[1:] - value_plus[:-1]
        advantages = self.discount(advantages)

        critic_loss = self.beta_v * 0.5 * tf.reduce_sum(input_tensor=tf.square(discounted_rewards - tf.reshape(values,[-1])))
        actor_loss = -tf.reduce_sum(tf.math.log(action_probs + 1e-7) * advantages)
        if self.entropy_var:
            entropy_loss = self.env.beta_e * entropy
        else:
            entropy_loss = self.beta_e * entropy

        total_loss = actor_loss + critic_loss + entropy

        return total_loss, actor_loss, critic_loss, entropy_loss

    def train(self):
        train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        
        for episode in range(self.nb_episodes):
            with tf.GradientTape() as tape:
                if(self.env.nb_obs>0):
                    obs = self.env.reset()
                else: 
                    self.env.reset()
                    obs = []
                
                action_probs_history = []
                critic_value_history = []
                rewards_history = []
                reward = 0.0
                action_onehot = np.zeros((self.nb_actions))
                cell_state = [tf.zeros((1,self.nb_hidden)),tf.zeros((1,self.nb_hidden))]
                entropy = 0.0
            
                for timestep in range(self.nb_trials * self.action_per_trial):
                    input = np.concatenate((obs, action_onehot, [reward], [timestep]),dtype = np.float32)
                    input = tf.expand_dims(input,0)
                    
                    # Predict action probabilities and estimated future rewards from environment state
                    action_probs, critic_value, cell_state = self.model([input,cell_state[0],cell_state[1]])
                    
                    critic_value_history.append(tf.squeeze(critic_value))

                    # Sample action from action probability distribution
                    action_probs = tf.squeeze(action_probs)
                    action = np.random.choice(self.nb_actions, p=action_probs.numpy())
                    action_probs_history.append(action_probs[action])
                    action_onehot = np.zeros((self.nb_actions))
                    action_onehot[action] = 1.0

                    # Apply the sampled action in our environment
                    if(self.env.nb_obs>0):   
                        obs, reward, done, _ = self.env.trial(action)
                    else: reward, done, _ = self.env.trial(action)
                    rewards_history.append(reward)
                    
                    # entropy
                    entropy += sp.stats.entropy(action_probs)
                    
                    if done: break
                    
                
                total_loss, actor_loss, critic_loss, entropy_loss = self.compute_loss(
                    tf.convert_to_tensor(action_probs_history,dtype=tf.float32), 
                    tf.convert_to_tensor(critic_value_history, dtype=tf.float32), 
                    tf.convert_to_tensor(rewards_history, dtype=tf.float32), 
                    entropy)
                        
                # Backpropagation
                grads = tape.gradient(total_loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                
                # Log
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss/total_loss', total_loss, step=episode)
                    tf.summary.scalar('loss/actor_loss', actor_loss, step=episode)
                    tf.summary.scalar('loss/critic_loss', critic_loss, step=episode)
                    tf.summary.scalar('loss/entropy', entropy_loss, step=episode)
                    tf.summary.scalar('game/reward', np.sum(rewards_history), step=episode)
                    tf.summary.histogram('game/action_probs', action_probs_history, step=episode)
                
            # Checkpoint
            if episode % 2000 == 0:
                checkpoint = tf.train.Checkpoint(self.model)
                checkpoint.save(self.ckpt_dir+'checkpoints_'+str(episode)+'/two_steps.ckpt')
                
        self.model.save(self.path+'/model.h5')
        
    def test(self, test_episode, test_model):
        test_summary_writer = tf.summary.create_file_writer(self.test_dir)

        for episode in range(test_episode):
            if(self.env.nb_obs>0):
                obs = self.env.test_reset()
            else: 
                self.env.test_reset()
                obs = []
            
            action_probs_history = []
            rewards_history = []
            reward = 0.0
            action_onehot = np.zeros((self.nb_actions))
            cell_state = [tf.zeros((1,self.nb_hidden)),tf.zeros((1,self.nb_hidden))]
            entropy = 0.0
            
            for timestep in range(self.nb_trials):
                input = np.concatenate((obs, action_onehot, [reward], [timestep]),dtype = np.float32)
                input = tf.expand_dims(input,0)
                
                # Predict action probabilities and estimated future rewards from environment state
                action_probs, _, cell_state = test_model([input,cell_state[0],cell_state[1]])
                
                # Sample action from action probability distribution
                action_probs = tf.squeeze(action_probs)
                action = np.random.choice(self.nb_actions, p=action_probs.numpy())
                action_probs_history.append(action_probs[action])
                action_onehot = np.zeros((self.nb_actions))
                action_onehot[action] = 1.0

                # Apply the sampled action in our environment
                if(self.env.nb_obs>0):   
                    obs, reward, done, _ = self.env.trial(action)
                else: reward, done, _ = self.env.trial(action)
                rewards_history.append(reward)
                
                # entropy
                entropy += sp.stats.entropy(action_probs)
                
                if done: break

            with test_summary_writer.as_default():
                tf.summary.scalar('loss/entropy', entropy, step=episode)
                tf.summary.scalar('game/reward', np.sum(rewards_history), step=episode)
                tf.summary.histogram('game/action_probs', action_probs_history, step=episode)
