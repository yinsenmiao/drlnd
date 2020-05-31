[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

In this project, I demonstrated and compared the performances of several variants of Deep Q-learning (DQN), also commonly referred to as the value-based methods. Perceptively, the following DQN architectures were considered:

* Vanilla Deep Q-learning
* Vanilla Deep Q-learning with dueling network
* Double Deep Q-learning
* Double Deep Q-learning with dueling network

We utilized an Unity reinforcement learning to perform comparison studies among the above four variants. The environment  of our interest consists of a continuous state space of 37 dimensions and 4 actions i.e. move left, move right, move forward and move backward. An agent can move around to collect yellow bananas for a reward of +1 score, while avoiding eating the blue bananas for a penalty of -1 score.  An agent with random action can be seen as the following animation:

<img src="images/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif" alt="Random Action Agent" style="zoom:120%;" />

In the following sections, I will discuss the implementation of the above DQN architectures, compares their performance on the banana collection task and present some ideas for future improvement. 

### Learning Algorithm

The main idea for various DQNs is to learn the action-value function, noted as <img src="./svgs/a18656976616796e481e7c608b8a2b40.svg" align=middle width=49.48137479999998pt height=24.65753399999998pt/> (<img src="./svgs/6f9bad7347b91ceebebd3ad7e6f6f2d1.svg" align=middle width=7.7054801999999905pt height=14.15524440000002pt/> is the current state and <img src="./svgs/44bc9d542a92714cac84e01cbbb7fd61.svg" align=middle width=8.68915409999999pt height=14.15524440000002pt/> is the corresponding action), using deep neural network as an non-linear approximation. Q-learning is a type of Temporal-Difference (TD) learning that learn from each step in a given learning episode.  Given a tuple of <img src="./svgs/e0d38b661c89bf91dd0d076749caf0f6.svg" align=middle width=119.46786884999997pt height=24.65753399999998pt/>, the Q-learning aims to update the following action-value table
<p align="center"><img src="./svgs/2fbe3efbd752b2a147655796d44399a7.svg" align=middle width=474.70498514999997pt height=29.58934275pt/></p>
where <img src="./svgs/e7b77ded6018d0dc509e61ffebc61518.svg" align=middle width=66.04313264999999pt height=22.465723500000017pt/> are the state, action, reward and <img src="./svgs/cf83185198a68ea312b2d4387b1af3fe.svg" align=middle width=31.68963764999999pt height=22.465723500000017pt/>  is the next state that follows. However, the above Q-table becomes infeasible as the dimension of state and action space increases.  Then as first proposed by the *[Deep Q learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)* paper, we attempt to approximate this Q-table using a deep network parameterized by a weight matrix, and iteratively update the weight parameter using the following rule:
<p align="center"><img src="./svgs/6193ad8f85544bcec6aa565a710a7bcb.svg" align=middle width=480.68170425pt height=29.58934275pt/></p>

where <img src="./svgs/d75649fbfd453bfa21eed2bb87fa9bf2.svg" align=middle width=22.48486679999999pt height=26.17730939999998pt/> are the weights of a fixed separate target network during the learning step, decoupling the target from the parameter updating step. 

In the implementation, we use two feed-forward networks: one online network that is keeping on learning; the other target network whose parameters are set by the online network every few iteration. Since the two networks sometime share the same weights, their network architectures are the same and are specified as following:

* Layer 1, Fully connected layer, Input dimension 37 (the size of the state space), Output dimension 128 with Relu activation;
* Layer 2, Fully connected layer, Input dimension 128, Output dimension 64 with Relu activation;
* Layer 3, Fully connected layer, Input dimension 64, Output dimension 4 (the size of the action space) with Relu activation.

We also applied a technique called **experiences replays** (EP) to stabilize the training, where a EP buffer is used to store batches of <img src="./svgs/a06e37c55121bf5779c15d3be77939f4.svg" align=middle width=119.46786554999998pt height=24.65753399999998pt/> experience tuple from past episodes.  After running through a large number of episodes, we randomly sample a few experience tuples and learn from them to update the parameters of Q-network. This stochastic sampling breaks the sequential nature of experiences, reduce correlation, and stabilizes the learning process considerably. 

To further improve performance of DQN, we implement two variants of it. 

## Double DQN
Since the Vanilla DQN can overestimate the action-values. the paper *[Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)*  proposed an alternative Q-target value which takes the action that maximizes the current Q network given the next state.  According to the paper, the weights <img src="./svgs/31fae8b8b78ebe01cbfbe2fe53832624.svg" align=middle width=12.210846449999991pt height=14.15524440000002pt/> is updated using the following gradient, instead.
<p align="center"><img src="./svgs/525eaa0ab80383d19570c151fdf70a58.svg" align=middle width=593.42330355pt height=29.58934275pt/></p>

## Dueling Network

<img src="./images/dueling_network.png" alt="dueling network" style="zoom: 30%;"/>

The *[Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)* proposes a different neural network to approximate the Q functions. In the final layer, the dueling network separates the target into two tasks: (1) Estimate the state-value function, and (2) Estimate the the advantages for each action within the state. Then those two estimates are combined together and are used to approximate the Q-values as below:
<p align="center"><img src="./svgs/41a4f167bb44e7065164c255c7befcec.svg" align=middle width=521.7990656999999pt height=43.76915895pt/></p>

<img src="./svgs/31fae8b8b78ebe01cbfbe2fe53832624.svg" align=middle width=12.210846449999991pt height=14.15524440000002pt/> is shared between those two networks (all the layers before the last layer), while  <img src="./svgs/c745b9b57c145ec5577b82542b2df546.svg" align=middle width=10.57650494999999pt height=14.15524440000002pt/> and <img src="./svgs/8217ed3c32a785f0b5aad4055f432ad8.svg" align=middle width=10.16555099999999pt height=22.831056599999986pt/> are the parameters of the two streams of the fully connected layers for estimating the state-value <img src="./svgs/4c88b510bc4f548b60c1ec2fbfc9c89d.svg" align=middle width=41.89507739999999pt height=24.65753399999998pt/> and advantage  <img src="./svgs/539452e6f183b66b5f4471c3ec75fecc.svg" align=middle width=58.03667099999999pt height=24.65753399999998pt/>, respectively. 

## Hyper-parameter Setting

| Parameter    | values | Meaning                                                      |
| ------------ | ------ | ------------------------------------------------------------ |
| lr           | 1e-3   | The learning rate for the deep q network.                    |
| BUFFER_SIZE  | 1e6    | The size for the replay memory                               |
| UPDATE_CYCLE | 8      | The weight is trained every 8 steps.                         |
| GAMMA        | 0.99   | The reward decay factor.                                     |
| TAU          | 1e-3   | The portion of new local_parameters are updated to target parameters. |
| BATCH_SIZE   | 64     | The batch size for each training of the deep q network.      |



# Results

Please follow the instructions in `Navigation_solution.ipynb` to train the agent for 3000 episodes using the above hypermeter setting. We then calculate the average reward for a testing set of 10 episodes. We can see that all the four methods achieve an average score above 15, which show the agents learned to play the game from interacting with the simulation environment.  Each method takes more than 1 hour to train for 3000 episodes. 

|     Model      | Avg Reward (Test) | Time (seconds) |
| :------------: | :---------------: | :------------: |
|      DQN       |       17.2        |    4146.114    |
| DQN + Dueling  |       17.9        |    5343.128    |
|      DDQN      |       15.7        |    4535.469    |
| DDQN + Dueling |       15.3        |    5536.724    |

I also plot the trace plot of reward versus the training episode overlaying the SMA 100 per each methods. 

|                             DQN                              |                        DQN + Dueling                         |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="images/dqn_model.png" alt="dqn_model" style="zoom: 33%;" /> | <img src="images/dqn_dueling_model.png" alt="dqn_dueling_model" style="zoom: 33%;" /> |
|                           **DDQN**                           |                      **DDQN + Dueling**                      |
| <img src="images/ddqn_model.png" alt="ddqn_model" style="zoom:33%;" /> | <img src="images/ddqn_dueling_model.png" alt="ddqn_dueling_model" style="zoom:33%;" /> |

If we plot the SMAs for all four methods together, we can see the following graph:

<img src="images/compare.png" alt="compare" style="zoom:40%;" />

where we can see when dueling is turned on the rewards start much higher than when without.  However, the benefits of double DQN is not significant better than the Vanilla DQN. This disadvantage might be due to this specific problem or because of some bug in my code. 

### Future work

I plan to investigate a comparison studies among all four methods to learn the strategies directly from the 84 x 84 RGB image from the agent's first-person view. I would like to understand the robustness of the approaches when a different hyper-parameter was used and further optimize it using Bayesian inference. Lastly, I will learn more about some state-of-art value-based approaches such as the Rainbow learning algorithm.