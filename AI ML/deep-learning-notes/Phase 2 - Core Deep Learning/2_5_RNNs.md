## 📘 Topic 2.5: Recurrent Neural Networks (RNNs)

### 🎯 Goal

Understand how RNNs process sequential data by maintaining memory of previous inputs and tackle their key limitations.

---

## 🧠 Core Concepts

### 1. **Why RNNs?**

CNNs and FNNs can’t handle *sequential dependencies*.

RNNs introduce:

* **Hidden state**: carries information from previous time steps.
* **Weight sharing** across time steps.

```plaintext
x₁ → h₁ → y₁
     ↓
x₂ → h₂ → y₂
     ↓
...
```

---

### 2. **RNN Cell Mechanics**

At time step `t`, RNN updates:

```math
hₜ = tanh(Wₓₕ·xₜ + Wₕₕ·hₜ₋₁ + b)
yₜ = Wₕy·hₜ
```

Where:

* `xₜ`: current input
* `hₜ`: hidden state
* `yₜ`: output
* All weights `W` are shared across time steps

---

### 3. **Backpropagation Through Time (BPTT)**

* RNNs are trained using **BPTT**, a variant of backpropagation.
* Gradients flow backward across time steps.

⚠️ Can suffer from:

* **Vanishing gradients** (gradients → 0)
* **Exploding gradients** (gradients → ∞)

---

### 4. **Variants to Overcome Gradient Issues**

#### 🔹 LSTM (Long Short-Term Memory)

* Introduces gates (input, forget, output) to control information flow.

#### 🔹 GRU (Gated Recurrent Unit)

* Simplified version of LSTM (no separate memory cell).

> Both outperform vanilla RNNs on most real-world tasks.

---

## 🔧 PyTorch RNN Example

```python
import torch.nn as nn

rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=1, batch_first=True)
output, hn = rnn(input)  # input: (batch, seq_len, input_size)
```

### LSTM:

```python
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=1, batch_first=True)
```

---

## 🔧 TensorFlow RNN Example

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense

model = Sequential([
    SimpleRNN(50, return_sequences=True, input_shape=(10, 1)),
    Dense(1)
])
```

### LSTM:

```python
LSTM(50, return_sequences=True)
```

---

## 🧱 Use Cases

| Use Case               | Description                    |
| ---------------------- | ------------------------------ |
| Text generation        | Predict next character/word    |
| Sentiment analysis     | Classify movie/product reviews |
| Time-series prediction | Forecast future values         |
| Speech recognition     | Convert voice to text          |

---

## 🧪 Exercises

### ✅ Conceptual

1. Why do RNNs struggle with long-term dependencies?
2. How does an LSTM prevent vanishing gradients?
3. Compare LSTM and GRU in terms of parameters and speed.

### ✅ Practical

* Implement a character-level RNN to generate text.
* Train an LSTM to predict sine waves.
* Try training GRU vs. LSTM on the same dataset and compare accuracy and speed.
