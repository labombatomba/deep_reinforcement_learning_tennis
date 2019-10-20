import numpy as np
from collections import namedtuple, deque
import random
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class MADDPG():
    
    
    def __init__(self,
                 state_size,
                 action_size,
                 nbr_agents,
                 random_seed,
                 gamma,
                 lr_actor,
                 lr_critic,
                 weight_decay,
                 tau,
                 update_rate,
                 updates_per_step,
                 buffer_size,
                 batch_size):
        
        
        self.state_size = state_size
        self.action_size = action_size
        self.nbr_agents = nbr_agents
        self.update_rate = update_rate
        self.updates_per_step = updates_per_step
        
        
        # Instantiate DDPG Agents
        ddpg_parameter = dict()
        ddpg_parameter['state_size'] = state_size
        ddpg_parameter['action_size'] = action_size
        ddpg_parameter['nbr_agents'] = nbr_agents
        ddpg_parameter['random_seed'] = random_seed
        ddpg_parameter['gamma'] = gamma
        ddpg_parameter['lr_actor'] = lr_actor
        ddpg_parameter['lr_critic'] = lr_critic
        ddpg_parameter['weight_decay'] = weight_decay
        ddpg_parameter['tau'] = tau                   

        self.agents = []
        
        for _ in range(nbr_agents):
            self.agents.append(DDPG(**ddpg_parameter))


        # Instantiate Replay Memory
        self.memory = ReplayBuffer(buffer_size,
                                   batch_size,
                                   random_seed)
        
        self.reset()
        

        
        
    def reset(self):
        
        for agent in self.agents:
            agent.reset()
            
        self.update_counter = 0
        
        
        
        
    def act(self, states):
        
        actions = []
        
        for state, agent in zip(states, self.agents):
            actions.append(agent.act(state))
        
        return np.asarray(actions)

        
    
    
    def step(self, states, actions, rewards, next_states, dones, train=True):
        
        # store in replay buffer
        self.memory.add(states.flatten(),
                        actions.flatten(),
                        np.asarray(rewards),
                        next_states.flatten(),
                        np.asarray(dones))
        
        
        if train:
            
            # Update counter
            self.update_counter += 1 
            
            if self.update_counter % self.update_rate == 0:
            
                self.update_counter = 0
            
                if len(self.memory) > self.memory.batch_size:
                    
                    for _ in range(self.updates_per_step):
                        experiences = self.memory.sample()
                        self.learn(experiences)        
        
        
        

    def learn(self, experiences):
        
        states, actions, rewards, next_states, dones = experiences
            
        # Evaluate Policies and augment actions
        next_actions = torch.zeros_like(actions).to(device)
        
        for k, ddpg_agent in enumerate(self.agents):
            
            states_k = states[0, k*self.state_size:(k+1)*self.state_size]
            actions_k = ddpg_agent.actor_local(states_k)
            next_actions[:, k*self.action_size:(k+1)*self.action_size] = actions_k
            
            
        # Update Local Networks
        for ddpg_agent in self.agents:
             ddpg_agent.learn(states,
                              next_states,
                              actions, 
                              next_actions,
                              rewards,
                              dones)
                
                
        # Update Target Networks
        for ddpg_agent in self.agents:
            ddpg_agent.soft_update(ddpg_agent.critic_local,
                                   ddpg_agent.critic_target)
            
            ddpg_agent.soft_update(ddpg_agent.actor_local,
                                   ddpg_agent.actor_target)
            
            
        
        
            
            
class DDPG():
    
    
    def __init__(self, 
                 state_size,
                 action_size,
                 nbr_agents,
                 random_seed,
                 gamma,
                 lr_actor,
                 lr_critic,
                 weight_decay,
                 tau):
    

        self.state_size = state_size
        self.action_size = action_size
        self.nbr_agents = nbr_agents

        self.seed = random.seed(random_seed)

        # Hyperparameter
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.tau = tau


        # Instantiate Actor Networks
        self.actor_local = Actor(state_size, 
                                 action_size,
                                 random_seed).to(device)

        self.actor_target = Actor(state_size, 
                                  action_size, 
                                  random_seed).to(device)



        # Instantiate Critic Networks
        self.critic_local = Critic(state_size*nbr_agents, 
                                   action_size*nbr_agents, 
                                   random_seed).to(device)

        self.critic_target = Critic(state_size*nbr_agents, 
                                    action_size*nbr_agents, 
                                    random_seed).to(device)


        # Instantiate Optimizers
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=lr_actor)

        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=lr_critic,
                                           weight_decay=weight_decay)


        # Instantiate Noise Process
        self.noise = OUNoise(action_size, random_seed)



        # Reset agent
        self.reset()
        
        
        
    def reset(self):
        self.noise.reset()
        # FIXXME - reset weights of ANNs
        
        
        
    def act(self, state, add_noise=True):

        state = torch.from_numpy(state).float().to(device)
        
        self.actor_local.eval()
        action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        if add_noise:
            action += self.noise.sample()
        
        return np.clip(action, -1, 1)
        

    
    

    
    def learn(self, states, next_states, actions, next_actions, rewards, dones):
        
    
        # --------------------- update local critic ---------------------------- #
        
        # Get predicted next-state actions and Q values from target models
        Q_targets_next = self.critic_target(next_states,
                                            next_actions)
         
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Update critic weights
        self.critic_optimizer.zero_grad()
        #critic_loss.backward()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()
    
    
        # ------------------------- update local actor ------------------------- #
        # Compute actor loss
        actor_loss = -self.critic_local(states, next_actions).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        #actor_loss.backward()
        self.actor_optimizer.step()
        
        

        
        
    def soft_update(self, local_model, target_model):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            
            target_param.data.copy_(self.tau*local_param.data + \
                                    (1.0-self.tau)*target_param.data)

            

            
            
            
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

    def __init__(self, buffer_size, batch_size, seed):
        '''Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        '''

        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", 
                                     field_names=["states", 
                                                  "actions",
                                                  "rewards",
                                                  "next_states",
                                                  "dones"])
        
        self.seed = random.seed(seed)
    
    
    def add(self, states, actions, rewards, next_states, dones):
        '''
        Add a new experience to memory.
        '''
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)
    
    
    def sample(self):
        '''
        Randomly sample a batch of experiences from memory.
        '''
        
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.states for e in experiences if e is not None]))
        states = states.float().to(device)
        
        actions = torch.from_numpy(np.vstack([e.actions for e in experiences if e is not None]))
        actions = actions.float().to(device)
        
        rewards = torch.from_numpy(np.vstack([e.rewards for e in experiences if e is not None]))
        rewards = rewards.float().to(device)
        
        next_states = torch.from_numpy(np.vstack([e.next_states for e in experiences if e is not None]))
        next_states = next_states.float().to(device)
        
        dones = torch.from_numpy(np.vstack([e.dones for e in experiences if e is not None]).astype(np.uint8))
        dones = dones.float().to(device)

        return (states, actions, rewards, next_states, dones)
    

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)