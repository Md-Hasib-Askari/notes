# Q-Learning

## Overview
Q-Learning is a model-free reinforcement learning algorithm that learns the optimal action-value function (Q-function) to maximize cumulative rewards.

## Key Concepts
- **Q-Function**: Represents the expected reward for taking an action in a given state
- **Bellman Equation**: Recursive relationship for updating Q-values
- **Exploration vs Exploitation**: Balancing exploration of new actions and exploitation of known rewards

## Algorithm
### Q-Learning Update Rule
```python
Q(s, a) = Q(s, a) + α * [r + γ * max(Q(s', a')) - Q(s, a)]
```

### Implementation
```python
import numpy as np

# Initialize Q-table
num_states = 10
num_actions = 4
Q = np.zeros((num_states, num_actions))

# Parameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    while not done:
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.choice(num_actions)
        else:
            action = np.argmax(Q[state])
        
        # Take action and observe reward and next state
        next_state, reward, done = env.step(action)
        
        # Update Q-value
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        
        state = next_state
```

## Advantages
- Simple and easy to implement
- Works well for discrete state and action spaces

## Disadvantages
- Inefficient for large state-action spaces
- Requires careful tuning of hyperparameters

## Applications
- **Games**: Board games, grid-world environments
- **Robotics**: Simple control tasks
- **Finance**: Decision-making under uncertainty

## Best Practices
1. Use epsilon decay for better exploration.
2. Normalize rewards for stable training.
3. Experiment with different values of α and γ.

## Resources
- **Papers**: "Q-Learning: A Tutorial"
- **Libraries**: NumPy, OpenAI Gym
