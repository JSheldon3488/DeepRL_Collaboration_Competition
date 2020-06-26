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
<!--
#TODO: Add better explanation of MADDPG
-->

<!--
#TODO: Add image of algorithm
-->
<p align="center">
    <img src = "images\MADDPG_Algorithm.png">
</p>

<!--
#TODO: Fill in network information
-->
**Changes to Algorithm:**

**Actor Network Architecture:**

**Critic Network Architecture:**

**Hyperparameters:** 

-----------

## Results:
Below is a graph of the training results and a short video of the trained agents interacting in the environment. 
The agent solves the environment in approximately ? episodes.

<p align="center">
<img src= "https://github.com/JSheldon3488/DeepRL_Collaboration_Competition/blob/master/images/MADDPG_results.png" >
</p>

<p align="center">
    <img src = "https://github.com/JSheldon3488/DeepRL_Collaboration_Competition/blob/master/images/trained_tennis.gif">
</p>

----------

## Future Ideas
 - Tune the hyperparameters to solve the environment faster.
 - Implement Prioritized Experience Replay
 - Setup a way to automatically run experiements and compare results between different architectures.
 
