# Pytorch Rainbow implementation for BSuite

In reinforcement learning, it can be difficult to get an agent to behave the way it is expected to (e.g. [1]). Furthermore, implementation details can make the difference between a successful agent and an agent that fails to learn [2].  
Behaviour suite by DeepMind is "a collection of carefully-designed experiments
that investigate core capabilities of reinforcement learning (RL) agents" [3].  These experiments can be used to compare the performance of different algorithms and therefore help to mitigate the problems mentioned above.

DQN is one of the most well-known RL algorithms and gained recognition for achieving human level performance on 49 Atari games [4].
In the Rainbow publication, six improvements on the standard DQN algorithm are combined to achieve even better performance on the set of Atari games [5].

DQN and Rainbow are great starting points to learn more about Reinforcement Learning and see how these algorithmic improvements change the agent's performance on the BSuite experiments.
During the next couple of  weeks I will be implementing DQN and the six improvements from Rainbow in [PyTorch](https://pytorch.org/).

## The code

The first step in this project entailed reading the chapters on DQN and Rainbow in the "Deep Reinforcement Learning Hands-On" book by Maxim Lapan [C1] and working through the corresponding [code](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On).
The structure and use of PyTorch methods in this repository is therefore highly influenced by that book.  
When I first started working on DRL, I completed Udacity's Deep Reinforcement Learning Nanodegree ([C2], here's their excellent [Github repository](https://github.com/udacity/deep-reinforcement-learning)) and read through OpenAi's Spinning Up repository [C3], so these two sources probably also had an influence on the code.  I also read through the baseline implementations of the BSuite repository, to get a better understanding of how the agent interacts with the environment.

I'll add further sources that helped me to better understand some detail of the algorithmic improvements and their implementation when I discuss their results.

## Results  
- [Basic DQN client and the first results](https://github.com/pluebcke/dqn_experiments/blob/master/results/basic_dqn.md)

## Todo
There are still some things I need to implement for the full Rainbow algorithm. My next steps in that direction are:
- Implement the improvements and run them on the BSuite experiments:
  - DDQN
  - Dueling DDQN
  - n-Step roll out
  - Prioritized Replay
  - Distributional DQN
  - Noisy DDQ
- Write better dosctrings and fix some errors within them.

## References
[1] https://www.alexirpan.com/2018/02/14/rl-hard.html, last  visited: 2020-03-03  
[2] Engstrom, Logan, et al. "Implementation Matters in Deep RL: A Case Study on PPO and TRPO." International Conference on Learning Representations. 2019.  
[3] Osband, Ian, et al. "Behaviour Suite for Reinforcement Learning." arXiv preprint arXiv:1908.03568 (2019).  
[4] Mnih, Volodymyr, et al. Human-level control through deep reinforcement learning. Nature, 2015, 518. Jg., Nr. 7540, S. 529-533.  
[5] Hessel, Matteo, et al. Rainbow: Combining improvements in deep reinforcement learning. In: Thirty-Second AAAI Conference on Artificial Intelligence. 2018.  

#### Code References:  
[C1] Lapan, Maxim. Deep Reinforcement Learning Hands-On, Packt Publishing Ltd, 2018.  
[C2] https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893, last visited 2020-05-03.  
[C3] https://spinningup.openai.com/en/latest/, last visited 2020-05-03.


#### Other references:  
Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.  
<em>Probably the most influential Reinforcement Learning book, this text is in my opinion a "must-read", but much less "hands-on" than the two other books I've mentioned.</em>

Morales, Miguel. Grokking Deep Reinforcement Learning, Manning Publications, 2020.
<em>Another more recent book about DRL that is very reader-friendly, with some nice explanations and illustrations. I've not used it for the code itself, but only because I had already started working through the other book [C1].</em>
