
## 🎮 13. Reinforcement Learning – Notes

### 📌 Overview:

Reinforcement Learning (RL) is a paradigm where agents **learn by interacting** with an environment to maximize cumulative rewards over time. It's the foundation of **AlphaGo, robotics, and autonomous systems**.

---

### 🧾 13.1 Markov Decision Processes (MDP)

#### ✅ Key Components:

* **S**: States (e.g., grid positions)
* **A**: Actions (e.g., move up/down)
* **P**: Transition probabilities $P(s' | s, a)$
* **R**: Rewards for state transitions
* **γ**: Discount factor for future rewards

#### ✅ Bellman Equation:

$$
V(s) = \max_a \left[ R(s, a) + \gamma \sum_{s'} P(s'|s,a)V(s') \right]
$$

---

### 🔁 13.2 Q-Learning & Deep Q-Networks (DQN)

#### ✅ Q-Learning (Tabular):

```python
Q[s][a] = Q[s][a] + α * (r + γ * max(Q[s’]) - Q[s][a])
```

#### ✅ DQN (Deep Q-Network):

* Uses a neural net to approximate Q-values
* Stabilized with **experience replay** and **target networks**

```python
Q(s, a) ≈ NN(s)[a]
```

#### ✅ Frameworks:

* TensorFlow / PyTorch
* `stable-baselines3`

---

### 🧠 13.3 Policy Gradient Methods

#### ✅ Difference from Q-Learning:

* **Policy-based** methods directly learn π(a|s)
* Suitable for **continuous action spaces**

#### ✅ REINFORCE Algorithm:

$$
\theta = \theta + \alpha \cdot \nabla_\theta \log \pi_\theta(a|s) \cdot G_t
$$

#### ✅ Advanced Variants:

* **Actor-Critic**
* **Proximal Policy Optimization (PPO)**
* **Trust Region Policy Optimization (TRPO)**

---

### 🧪 13.4 OpenAI Gym

#### ✅ A toolkit for training RL agents:

* Simulates environments: CartPole, MountainCar, Atari, etc.

```python
import gym
env = gym.make("CartPole-v1")
state = env.reset()
```

#### ✅ Common Workflow:

1. Initialize environment
2. Agent selects action
3. Environment returns new state + reward
4. Train policy/Q-values

