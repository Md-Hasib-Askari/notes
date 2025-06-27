# TensorFlow and Keras

## Overview
TensorFlow is an open-source deep learning framework developed by Google. Keras is a high-level API built on top of TensorFlow, designed for fast experimentation and ease of use.

## Key Features
- **Scalability**: Supports distributed training and deployment
- **Ease of Use**: Keras simplifies model building and training
- **Extensive Ecosystem**: Includes TensorFlow Hub, TensorFlow Lite, and TensorFlow.js
- **Production-Ready**: Tools for serving and deploying models

## Installation
```bash
# Install TensorFlow
pip install tensorflow

# For GPU support
pip install tensorflow --index-url https://pypi.org/simple
```

## Basic Workflow

### Tensors
```python
import tensorflow as tf

# Create tensors
x = tf.constant([1, 2, 3])
y = tf.constant([4, 5, 6])

# Operations
z = x + y
print(z)

# GPU tensors
x_gpu = x.gpu()
print(x_gpu)
```

### Neural Networks with Keras
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define a simple model
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
X = tf.random.normal((100, 10))
y = tf.random.uniform((100,), maxval=2, dtype=tf.int32)
model.fit(X, y, epochs=10, batch_size=32)
```

### Training Loop
```python
import tensorflow as tf

# Data
X = tf.random.normal((100, 10))
y = tf.random.uniform((100,), maxval=2, dtype=tf.int32)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X, y, epochs=10, batch_size=32)
```

## Advanced Features

### Custom Layers
```python
from tensorflow.keras.layers import Layer

# Define a custom layer
class CustomLayer(Layer):
    def __init__(self):
        super(CustomLayer, self).__init__()

    def call(self, inputs):
        return inputs ** 2

# Use the custom layer
layer = CustomLayer()
x = tf.constant([1.0, 2.0, 3.0])
output = layer(x)
print(output)
```

### Data Loading
```python
import tensorflow as tf

# Load dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Data pipeline
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32).shuffle(1000)
for batch in train_dataset:
    print(batch)
```

### Distributed Training
```python
import tensorflow as tf

# Strategy
strategy = tf.distribute.MirroredStrategy()

# Build and compile model within strategy scope
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=10, batch_size=32)
```

## Best Practices
1. Use `tf.keras` for model definition and training.
2. Use `tf.data` for efficient data pipelines.
3. Use `tf.function` for performance optimization.
4. Use `tf.saved_model` for model serialization.
5. Use `TensorBoard` for monitoring training.

## Common Pitfalls
- **Memory Issues**: Use `tf.data` for efficient data loading.
- **Device Compatibility**: Ensure tensors and models are on the same device (CPU/GPU).
- **Gradient Issues**: Use `tf.GradientTape` for custom training loops.

## Applications
- **Computer Vision**: Image classification, object detection
- **Natural Language Processing**: Text classification, sequence modeling
- **Reinforcement Learning**: Policy optimization, value function approximation

## Resources
- **TensorFlow Documentation**: Official guide and tutorials
- **Keras Documentation**: High-level API for TensorFlow
- **TensorFlow Hub**: Pre-trained models and datasets
- **TensorFlow Lite**: Tools for mobile and embedded devices
- **TensorFlow.js**: Tools for browser-based machine learning
