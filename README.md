# Solving Control Problems using Reinforcement Learning

This repository contains my final project for the ARI3212 Reinforcement Learning course at the University of Malta. The project investigates how different reinforcement learning algorithms perform in solving two control problems from the OpenAI Gymnasium suite:

- LunarLander-v3 (Discrete Actions)
- LunarLanderContinuous-v3 (Continuous Actions)

Both value-based and actor-critic methods are explored, focusing on:
- Deep Q-Networks (DQN)
- Double DQN
- Deep Deterministic Policy Gradient (DDPG)

---

## Report

You can read the full project report here: [Gianluca_Aquilina_Report_RL.pdf](./Gianluca_Aquilina_Report_RL.pdf)

---

## Notebooks

### Experiment1.ipynb – DQN and Double DQN on LunarLander-v3

This notebook applies Deep Q-Networks (DQN) and Double DQN to the LunarLander-v3 environment, with focus on model tuning and robustness testing under noisy observations.

#### Core Components
1. Replay Memory  
   - Stores experiences and samples mini-batches for training.
   - Ensures the agent has enough initial transitions before training begins.

2. Q-Network (DQN)  
   - A simple two-layer neural network with `tanh` activation.
   - Output layer gives Q-values for each discrete action.

3. Exploration Strategy  
   - Epsilon-greedy action selection with exponential decay (0.9 → 0.05).
   - Encourages exploration early in training, then gradually exploits the learned policy.

---

#### Part 1: Hyperparameter Tuning

- Parameters Tuned:
  - learning_rate: [0.0005, 0.001, 0.01]
  - hidden_size: [64, 128, 256]

- Tuning Method:
  - Grid search across 9 configurations.
  - Tracked episodes needed to reach an average reward of 195 over the last 50 episodes.

- Best Configuration:
  - learning_rate = 0.001, hidden_size = 128
  - Solved environment in 621 episodes
  - Best model saved as `standard_dqn.pth`

---

#### Part 2: Double DQN Improvement

- Implemented Double DQN as an enhancement to the standard DQN.
- Separates action selection (via online network) from action evaluation (via target network) to reduce overestimation.
- Trained using same best configuration: 128 hidden units, 0.001 learning rate.
- Saved model as `double_dqn.pth`

---

#### Part 3: Robustness Testing Under State Noise

- Noise Injection:
  - Injected Gaussian noise into state observations to simulate uncertainty.

- Noise Levels:
  - LOW: σ = 0.05
  - MEDIUM: σ = 0.15
  - HIGH: σ = 0.25

- Evaluation:
  - Each model tested over 100 episodes for each noise level.
  - Tracked average reward and standard deviation.

#### Key Observations

| Noise Level | Standard DQN (mean ± std) | Double DQN (mean ± std) |
|-------------|----------------------------|---------------------------|
| LOW         | Higher average, stable     | Lower average, less stable|
| MEDIUM      | Mild drop in reward        | Sharp performance drop    |
| HIGH        | Gradual degradation        | Larger performance collapse|

- Standard DQN showed stronger resistance to noise.
- Double DQN struggled as noise increased, with more learning instability.
- Highlights importance of model robustness for real-world noisy environments.

---

### Experiment2.ipynb – DDPG on LunarLanderContinuous-v3

This notebook applies the Deep Deterministic Policy Gradient (DDPG) algorithm to the LunarLanderContinuous-v3 environment. The focus is on applying, tuning, and stress-testing DDPG in a continuous action setting.

#### Core Components
1. Actor-Critic Networks  
   - Actor outputs continuous actions using a Tanh layer.  
   - Critic evaluates Q-values from state-action pairs.

2. Replay Memory  
   - Stores transitions for experience replay.  
   - Minimum buffer size enforced before training begins.

3. Noise for Exploration  
   - Uses Ornstein-Uhlenbeck noise for smooth, temporally correlated exploration.

---

#### Part 1: Hyperparameter Tuning

- Parameters Tuned:
  - τ (soft update rate): [0.001, 0.005, 0.01]
  - batch_size: [32, 64, 128]

- Tuning Method:
  - Grid search across 9 combinations.
  - Tracks return and episodes needed to solve the environment.

- Best Configuration:
  - τ = 0.005, batch_size = 128
  - Solved environment in 49 episodes with final average return of 212.81

---

#### Part 2: Noise Robustness Analysis

- Noise Levels Tested:
  - LOW: σ = 0.1
  - MEDIUM: σ = 0.2
  - HIGH: σ = 0.3

- Evaluation Method:
  - Reuses trained model to test on each noise level.
  - Records episode rewards, solution time, and standard deviation.

#### Key Observations

| Noise Level | Avg Return | Episodes to Solve | Std Dev |
|-------------|------------|-------------------|---------|
| LOW         | 212.81     | 49                | ±12.5   |
| MEDIUM      | 196.38     | 98                | ±21.7   |
| HIGH        | 198.38     | 76                | ±24.3   |

- DDPG remains robust under increasing noise.
- Slight performance drops, but consistently solves the task.
- Medium noise introduced the most learning instability.
- High noise converged faster than medium — highlighting model adaptability.

---

## Algorithms Overview

### Deep Q-Networks (DQN)
- Discrete action spaces
- Uses experience replay and target networks
- Tuned with learning rate and hidden layer size
- Epsilon-greedy exploration strategy
- Improved using Double DQN to reduce overestimation

### Deep Deterministic Policy Gradient (DDPG)
- Continuous action spaces
- Actor-Critic architecture
- Uses soft target network updates and OU noise for exploration
- Tuned with tau (soft update rate) and batch size

---

## References

1. Mnih et al., Human-level control through deep reinforcement learning, Nature (2015)  
2. Van Hasselt et al., Deep Reinforcement Learning with Double Q-learning, AAAI (2016)  
3. Sutton & Barto, Reinforcement Learning: An Introduction  
4. University of Malta lecture notes (2024)

---

## Author

Gianluca Aquilina  
University of Malta  
gianniaqu@gmail.com
