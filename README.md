[//]: # "Image References"

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"



# Project 3: Colaboration and Competition
### Introduction

The content in this repository was created in order to accomplish the third  project assignment of Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) and utilizes Unity's  [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

#### Objective
The task is to control two rackets to bounce a ball over a net. 

![Trained Agent][image1]


#### Action Space
Two continuous actions are available (numbers between -1.0 and 1.0), corresponding to movement toward (or away from) the net, and jumping. 

#### Observation Space
The observation space consists of 24 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.


#### Reward
If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.

#### Goal
The training of the Agents is over, once an average reward of +0.5 over 100 consecutive episodes is obtained. The score is calculated by the maximum of the individual immediate rewards.


### Dependencies

#### Files
- `Tennis_DDPG.ipynb`	(Main file for training.)
- `DDPG_Agent.py` (Deep deterministic policy gradient (DDPG) agent)
- `model.py` (Actor and critic network.)
- `ddpg_critic_A.pth` (weights of trained critic network of agent A)
- `ddpg_actor_A.pth` (weights of trained actor network of agent A)
- `ddpg_critic_B.pth` (weights of trained critic network of agent B)
- `ddpg_actor_B.pth` (weights of trained actor network of agent B)
- `Validate_Tennis_DDPG.ipynb` (Visualizes a smart agent.)

#### Environments

##### Anaconda Environment

Follow the instructions in https://github.com/udacity/deep-reinforcement-learning section **Dependencies** to set up the anaconda environment.


##### Unity Environment

1. Download and unpack the environment for your operating system from the links below:


- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

See https://github.com/udacity/deep-reinforcement-learning/blob/master/p3_collab-compet/README.md for further information.

2. Set the path to the environment binary in `Tenns_DDPG.ipynb` (code cell 2) according to your environment. For example:
    ```python
    env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64")
    ```


#### Getting Started
For training run Tennis_DDPG.ipynb`. For validation instantiate two agents from `DDPG_Agent.py` and load actor and critic weights as follows or simply run `Validate_Tennis_DDPG.ipynb.ipynb` for watching a smart agent.
```python
agent_A.actor_local.load_state_dict(torch.load('ddpg_actor_A.pth'))
agent_A.critic_local.load_state_dict(torch.load('ddpg_critic_A.pth'))
agent_B.actor_local.load_state_dict(torch.load('ddpg_actor_B.pth'))
agent_B.critic_local.load_state_dict(torch.load('ddpg_critic_B.pth'))
```


