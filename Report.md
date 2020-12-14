# Instructions

Follow the instructions in either `Navigation_local.ipynb` for local computer or `Navigation_WS.ipynb` for udacity workspace to get started with training your own agent! Note only `Navigation_local.ipynb` is fully updated.

# Implementation
A checkpoint file is included 'checkpoint_Score_16.06.pth' for a fully trained agent with an average score of 16 per game. 
There is a `model.py` file included which includes the torch NN model, consisting of 3 fuly connected layers of  128, 64 and 32 neurons. RELUs are used as activation functions. 

`dqn_agent.py` contains the dqn training functions and the experience replay buffer that contains all the past experience SARSA tupples to be used for training. It also contains an action function `act(state)` used to return a policy action.  
It contains two DQN methods, `learn()` with a standard DQN learning algorithm and `learn_ddqn()` using the double-DQN algorithm. The DDQN should be more robust towards overestimating the Q-values during early stages of the training. 
The hyper parameter used for training the agent was:
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

`Navigation_local.ipynb` contains cells to both train a new agent or use a pretrained agent and do an evaluation run.
https://github.com/SorenRusbjerg/RL_Navigation/blob/master/Navigation_local_ddqn_1200episodes.html contains a view of the training of the agent using 1200 episodes. Here the score can be seen during the training.

# Future work
The DQN algorithm could be extended with a prioritized replay buffer to give higher probability to more important SARSA samples compared to less important samples. 
Also extending with a Dueling DQN, could result in faster training, as it will seperate the state value function and advantage function, giving the action values gain to the state value function. This provides better policies for environments with many similar valued action-value functions. However as we only have 4 actions in this game, it might not give much extra improvement.

More experimentation with hyper paramers such as the NN layer composition and the discount factor GAMMA could also prove to be usefull.









