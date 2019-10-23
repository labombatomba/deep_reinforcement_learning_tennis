import numpy as np
from collections import namedtuple, deque
import random
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPG():
    '''
    Deep determinstic policy gradient agent
    '''
    
    def __init__(self, 
                 state_size,
                 action_size,
                 random_seed,
                 gamma,
                 lr_actor, 
                 lr_critic,
                 weight_decay, 
                 tau,
                 buffer_size,
                 batch_size,
                 update_rate,
                 updates_per_step):
        '''
        Initialize an DDPG Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            nbr_env (int): number of environments the agent is acting with
            
        '''    
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.seed = random.seed(random_seed)
        
        # Hyperparameter
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.tau = tau
        self.update_rate = update_rate
        self.updates_per_step = updates_per_step

        
        
        # Instantiate Actor Networks
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        
        # Instantiate Critic Networks
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        
        # Instantiate Optimizers
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=lr_critic,
                                           weight_decay=weight_decay)
        
        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        
                
        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, random_seed)
          
            
        self.update_counter = 0
            
            
            
        
    def step(self, state, action, reward, next_state, done):
        '''
        Save experience in replay memory, and use random sample from buffer to learn.
        '''
        
        # Store experiences
        self.memory.add(state, action, reward, next_state, done)
        
        
        if len(self.memory) > self.memory.batch_size:
            
            # Update counter
            self.update_counter += 1 
            
            if self.update_counter >= self.update_rate:
                for _ in range(self.updates_per_step):
                    experiences = self.memory.sample()
                    self.learn(experiences, self.gamma)
        
                self.update_counter = 0
        

            
        
        
    def act(self, state, add_noise=True):
        '''
        Returns actions for given state as per current policy.
        '''
        
        state = torch.from_numpy(state).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()        
        self.actor_local.train()
        
        if add_noise:
            action += self.noise.sample()
            
        return np.clip(action, -1, 1)

        
        
        
    def reset(self):
        '''
        '''
        self.noise.reset()
        
        
        
    def learn(self, experiences, gamma):
        '''
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        '''
        
        states, actions, rewards, next_states, dones = experiences
        
        # update critic ##########################################################
        
        # compute predicted Q values
        next_actions = self.actor_target(next_states)
        next_Q_targets = self.critic_target(next_states, next_actions)
        
        # compute Q values for current states
        Q_targets = rewards + (gamma*next_Q_targets*(1 - dones))
        
        # compute loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # update weigths
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        
        # update actor ##########################################################
        
        # compute loss
        pred_actions = self.actor_local(states)
        actor_loss = -self.critic_local(states, pred_actions).mean()
        
        # update weights
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        
        # update target networks
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)        
        
        
        
    def soft_update(self, local_model, target_model, tau):
        '''
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        '''
        
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

        
        
        
        
        

class OUNoise:
    '''
    Ornstein-Uhlenbeck process.
    '''

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        '''
        Initialize parameters and noise process.
        '''
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        '''
        Reset the internal state (= noise) to mean (mu).
        '''
        self.state = copy.copy(self.mu)

    def sample(self):
        '''
        Update internal state and return it as a noise sample.
        '''
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
    
    
    
class ReplayBuffer:
    '''
    Fixed-size buffer to store experience tuples.
    '''

    def __init__(self, action_size, buffer_size, batch_size, seed):
        '''Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        '''
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", 
                                     field_names=["state", "action", "reward", "next_state", "done"])
        
        self.seed = random.seed(seed)
    
    
    def add(self, state, action, reward, next_state, done):
        '''
        Add a new experience to memory.
        '''
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    
    def sample(self):
        '''
        Randomly sample a batch of experiences from memory.
        '''
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)