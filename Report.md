# Udacity Deep Reinforcement Learning Project 3: Collaboration and Competition
Starter code and project details can be found [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet).

--------

## Environment Detials and Untrained Agent
The goal of this project is to train an agent to solve the [Unity Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.
Below is details about the environment and a video of untrained agents acting in the environment.

<p align="center">
    <img src = "https://github.com/JSheldon3488/DeepRL_Collaboration_Competition/blob/master/images/untrained_tennis.gif">
</p>

**Reward:** +0.1 for individual agent if it hits the ball over the net. -0.01 If an agent lets a ball hit the ground or hits the ball out of bounds.

**Observation Space:** consists of 3 stacks of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  

**Action Space:** 2 continuous actions corresponding to movement toward (or away from) the net and jumping. 

**Goal:** Get an average score of +0.5 (over 100 consecutive episode window, after taking the maximum score per episode over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.
 
 --------
 
## Training the Agent: MADDPG (Multi-Agent Actor-Critic)
[Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf) 
is very similar to other actor-critic algorithms that are used to solve problems with continuous actions spaces. 
The big difference is that it extends the actor-critic method to take advantage of multiple agents.
It has decentralized actors just like in DDPG that selects an action based on an observation,
but now you use a centralized critic that takes in additional information about the state and actions of the other agents and outputs a Q-Value.
More details can be read in the paper linked in above. The outline of the algorithm is below.

<p align="center">
    <img src = "images\MADDPG_Algorithm.png">
</p>

## Results:
I could not get this algoirhtm to work in practice. No matter which combination of hyperparameters, model architecture, or how long I ran the simulation,
I could not get this algorithm to work. I pivoted to trying to solve the environment with two single DDPG Agents. The results are below.

-----------

## Training the Agents Attempt 2: Multiple DDPG Agents
[Deep Deterministic Policy Gradients](https://arxiv.org/abs/1509.02971) is an actor-critic algorithm that is used to solve problems with
continuous actions spaces. More details can be read in the paper linked in the last sentence. The outline of the algorithm is below.

<p align="center">
    <img src = "https://github.com/JSheldon3488/DeepRL_Continuous_Control/blob/master/images/DDPG_Algorithm.png">
</p>

**Actor Network Architecture (each agent):**
  - Layer 1: Input_State (24) -> Fully_Connected (24, 512) -> Relu(512)
  - Layer 2: Fully_Connected (512, 256) -> Relu(256)
  - Layer 3: Fully_Connected (256, 2) -> tanh (2)
  - Output: 2 actions between (-1,1) 

**Critic Network Architecture (each agent):**
  - Layer 1: Input_State (24) -> Fully_Connected (24, 512) -> Relu(512)
  - Layer 2: Concat (Layer_1_Output + action_size 2) -> Fully_Connected (514,256) -> Relu(256)
  - Layer 3: Fully_Connected (256, Q_value 1)
   - Output: 1 Q_Value

**Hyperparameters (each agent):** 
 - Replay Buffer Size: 100,000
 - Batch Size: 256
 - Update Every: 1 (how often in timesteps to update networks)
 - Number of Updates: 1 (how many times to update per update_every)
 - Discount Rate Gamma: 0.99 (Q-Value Calculation)
 - Network Soft Update Rate Tau: 0.01
 - Learning Rate Actor: 0.0001
 - Learning Rate Critic: 0.001
 - Weight Decay: 0
 - Exploration Rate Epsilon: (start=1, decay=0.998, min=0)

## Results:
Below is a graph of the training results and a short video of the trained agents interacting in the environment. 
The agent solves the environment in approximately 900 episodes.

<p align="center">
<img src= "https://github.com/JSheldon3488/DeepRL_Collaboration_Competition/blob/master/images/MADDPG_results.png" >
</p>

<p align="center">
    <img src = "https://github.com/JSheldon3488/DeepRL_Collaboration_Competition/blob/master/images/trained_tennis.gif">
</p>

----------

## Future Ideas
 - Train the agent longer to see max score and if agent is stable.
 - Figure out what adjustments need to be made to make MADDPG work.
 - Implement Prioritized Experience Replay
 - Setup a way to automatically run experiements and compare results between different architectures.
 
