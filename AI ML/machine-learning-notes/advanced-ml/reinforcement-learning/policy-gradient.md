# Policy Gradient Methods

## Overview
Policy gradient methods are a class of reinforcement learning algorithms that optimize the policy directly by maximizing the expected reward.

## Key Concepts
- **Policy**: A mapping from states to actions
- **Objective**: Maximize the expected cumulative reward
- **Gradient Estimation**: Use stochastic gradient ascent to update the policy

## Algorithm
### REINFORCE
```python
import numpy as np
import tensorflow as tf

# Define policy network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_actions, activation='softmax')
])

# Training loop
for episode in range(num_episodes):
    states, actions, rewards = collect_episode_data()
    
    with tf.GradientTape() as tape:
        logits = model(states)
        action_probs = tf.nn.softmax(logits)
        loss = compute_loss(action_probs, actions, rewards)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

## Advantages
- Works well in high-dimensional action spaces
- Can learn stochastic policies

## Disadvantages
- High variance in gradient estimates
- Requires careful tuning of learning rates

## Applications
- **Robotics**: Control tasks
- **Games**: Strategy optimization
- **Finance**: Portfolio management

## Best Practices
1. Use baseline subtraction to reduce variance.
2. Normalize rewards for stable training.
3. Experiment with different architectures for the policy network.

## Resources
- **Papers**: "Policy Gradient Methods for Reinforcement Learning"
- **Libraries**: TensorFlow, PyTorch, OpenAI Gym
