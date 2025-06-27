# RNN Basics

## Overview
Recurrent Neural Networks (RNNs) are designed to work with sequential data by maintaining hidden states that carry information from previous time steps.

## Key Concepts

### Architecture
- **Hidden State**: Memory that persists across time steps
- **Weight Sharing**: Same weights used at each time step
- **Backpropagation Through Time (BPTT)**: Training method for RNNs

### Types
1. **Vanilla RNN**: Basic recurrent layer
2. **LSTM**: Long Short-Term Memory (solves vanishing gradient)
3. **GRU**: Gated Recurrent Unit (simpler than LSTM)

## Quick Implementation

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense

# Simple RNN
model = Sequential([
    SimpleRNN(50, return_sequences=True, input_shape=(timesteps, features)),
    SimpleRNN(50),
    Dense(1, activation='sigmoid')
])

# LSTM
model = Sequential([
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(1)
])

# GRU
model = Sequential([
    GRU(64, return_sequences=True),
    GRU(32),
    Dense(1)
])
```

## Common Applications
- **Text Generation**: Language modeling
- **Sentiment Analysis**: Text classification
- **Time Series**: Stock prices, weather forecasting
- **Machine Translation**: Seq2seq models

## Key Parameters
- `units`: Number of RNN units (hidden size)
- `return_sequences`: Return full sequence or last output
- `dropout`: Regularization during training
- `recurrent_dropout`: Dropout on recurrent connections

## Best Practices
1. **Use LSTM/GRU** for longer sequences (>10 steps)
2. **Batch Normalization** can help with training stability
3. **Gradient Clipping** prevents exploding gradients
4. **Bidirectional RNNs** for non-causal tasks

## Common Issues
- **Vanishing Gradients**: Use LSTM/GRU instead of vanilla RNN
- **Slow Training**: RNNs are inherently sequential
- **Memory Usage**: Scales with sequence length

## Quick Tips
- Start with LSTM for most sequential tasks
- Use `return_sequences=True` for intermediate layers
- Consider attention mechanisms for very long sequences
- Transformers often outperform RNNs for many tasks
