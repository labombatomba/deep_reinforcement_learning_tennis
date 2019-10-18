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
    '''
    Multiagent deep deterministic policy gradient super agent
    '''

    def __init__(self,
                 dim_observation,
                 dim_action,
                 nbr_agents,
                 random_seed,
                 gamma, 
                 lr_actor,
                 lr_critic,
                 weight_decay,
                 tau,
                 buffer_size,
                 batch_size,
                 update_rate):
        
        '''
        Initialize an MDDPG parent class.
        
        Params
        ======
            dim_observation (int): dimension of individual observation space (same for all agents)
            dim_action (int): dimension of individual action space (same for all agents)
            random_seed (int): random seed
            nbr_agents (int): number of agents
        '''  
    

        self.dim_observation = dim_observation
        self.dim_action = dim_action
        self.nbr_agents = nbr_agents
        self.update_rate = update_rate
        
        
        # Instantiate DDPG agents
        self.agents = []
        
        agent_parameter = dict()
        agent_parameter['state_size'] = self.dim_observation
        agent_parameter['action_size'] = self.dim_action
        agent_parameter['nbr_agents'] = self.nbr_agents
        
        agent_parameter['gamma'] = gamma
        agent_parameter['lr_actor'] = lr_actor
        agent_parameter['lr_critic'] = lr_critic
        agent_parameter['weight_decay'] = weight_decay
        agent_parameter['tau'] = tau
        agent_parameter['random_seed'] = random_seed
        

        for _ in range(self.nbr_agents):
            self.agents.append(DDPG(**agent_parameter))
        
        
        # Instantiate replay memory
        self.memory = ReplayBuffer(buffer_size, batch_size, random_seed)
        
        self.reset()
         
            
    
    def reset(self):
        '''
        FIXXME
        '''
        for agent in self.agents:
            agent.reset()
            
        self.update_counter = 0
        
        
        
        
    def act(self, observations):
        '''
        FIXXME
        '''
        
        actions = []
        
        for state, agent in zip(observations, self.agents):
            actions.append(agent.act(state))
            
        return np.asarray(actions)

        
     
    
    def step(self, states, actions, rewards, next_states, dones):
        '''
        Save experience in replay memory, and use random sample from
        buffer to train the agents
        
        
        Arguments
        =========
            states (float): (nbr_agents x dim_observation) numpy array
            actions (float): (nbr_agents x dim_action) numpy array
            rewards (float): (1 x nbr_agents) list
            dones (bool) : (1 x nbr_agents) list
        '''
                
        # Store experiences
        self.memory.add(states.flatten(),
                        actions.flatten(),
                        rewards,
                        next_states.flatten(),
                        dones)
        
        # Update counter
        self.update_counter += 1 
        
        if self.update_counter % self.update_rate == 0:
            
            # Reset update counter
            self.update_counter = 0
            
            # Update only if sufficient number of samples are available
            if len(self.memory) >= self.memory.batch_size:
                
                experiences = self.memory.sample()
                self.learn(experiences)
                
                
                
    def learn(self, experiences):

        states, actions, rewards, next_states, dones = experiences

        # Calculate next actions and augment them
        next_actions = torch.zeros(self.memory.batch_size, 
                                   self.nbr_agents*self.dim_action).float().to(device)
        
        for k in range(self.nbr_agents):
            states_k = states[:, k*self.dim_observation:(k+1)*self.dim_observation]
            next_actions[:, k*self.dim_action:(k+1)*self.dim_action] = \
            self.agents[k].actor_target(states_k)
            
        
        for k in range(self.nbr_agents):
            
            # Update local critic networks
            self.agents[k].update_critic(states, 
                                         next_states,
                                         actions,
                                         next_actions, 
                                         rewards[:,k],
                                         dones[:,k])
            
            
            # Update local actor networks 
            self.agents[k].update_actor(states, actions)
         
            
        # Update target networks           
        for agent in self.agents:
            agent.soft_update(agent.critic_local, agent.critic_target)
            agent.soft_update(agent.actor_local, agent.actor_target)        
        
            

class DDPG():
    '''
    Deep determinstic policy gradient agent
    '''
    
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
        '''
        Initialize an DDPG Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            nbr_agents (int): number of agents interacting in the environment
            random_seed (int): random seed
            
        '''    
        
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
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        
        # Instantiate Critic Networks
        self.critic_local = Critic(state_size*nbr_agents, action_size*nbr_agents, random_seed).to(device)
        self.critic_target = Critic(state_size*nbr_agents, action_size*nbr_agents, random_seed).to(device)
        
        # Instantiate Optimizers
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=lr_critic,
                                           weight_decay=weight_decay)
        
        # Noise process
        self.noise = OUNoise(action_size, random_seed)
              
        
        
    def act(self, state, add_noise=True):
        '''
        Returns action for given state as per current policy.
        '''
        
        state = torch.from_numpy(state).float().to(device)
        
        self.actor_local.eval()
        action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        if add_noise:
            action += self.noise.sample()
        
        return np.asarray(action)
    
    

        
        
    def reset(self):
        '''
        '''
        self.noise.reset()
 
       
        
        
    def update_critic(self, states, next_states, actions, next_actions, reward, done):
        '''
        Updates deep Q-networks (critic) using the augmented states and actions
        
        Params
        ======
            states : augmented observation vector
            next_states : augmented consecutive observation vector
            actions : augmented action vector
            next_actions : augmented next action vector
            reward : individual reward
            done : individual done flag
        '''
        
        
        # compute Q target values
        next_Q_targets = self.critic_target(next_states, next_actions)
        Q_targets = reward + (self.gamma*next_Q_targets*(1 - done))                     
                             
                            
        # compute loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # update weigths
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        
        
    def update_actor(self, states, actions):
        '''
        Updates deep policy networks (actor) using the augmented state vector
        
        Params
        ======
            states : augmented observation vector
            actions : augmented action vector
        '''
    
        # compute loss
        actor_loss = -self.critic_local(states, actions).mean()
        
        # update weights
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step() 
        
        
        
    def soft_update(self, local_model, target_model):
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
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

        

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