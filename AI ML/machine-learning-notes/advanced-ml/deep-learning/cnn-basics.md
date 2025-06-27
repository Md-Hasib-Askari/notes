# CNN Basics - Convolutional Neural Networks

## Overview
Convolutional Neural Networks (CNNs) are specialized deep learning architectures designed for processing grid-like data such as images. They use convolutional layers to automatically learn spatial hierarchies of features.

## Core Components

### 1. Convolutional Layer
- **Purpose**: Extract features using learnable filters/kernels
- **Operation**: Sliding window convolution across input
- **Parameters**: Filter size, stride, padding, number of filters

```python
import tensorflow as tf
from tensorflow.keras import layers

# Basic conv layer
conv_layer = layers.Conv2D(
    filters=32,         # Number of output filters
    kernel_size=(3, 3), # Filter size
    stride=1,           # Step size
    padding='same',     # Padding type
    activation='relu'
)
```

### 2. Pooling Layer
- **Purpose**: Reduce spatial dimensions, control overfitting
- **Types**: Max pooling, Average pooling
- **Effect**: Translation invariance, computational efficiency

```python
# Max pooling
pool_layer = layers.MaxPooling2D(
    pool_size=(2, 2),
    strides=2
)
```

### 3. Fully Connected Layer
- **Purpose**: Final classification/regression
- **Location**: Usually at the end after flattening
- **Function**: Combine features for final prediction

## Basic CNN Architecture

```python
import tensorflow as tf
from tensorflow.keras import Sequential, layers

def create_basic_cnn(input_shape, num_classes):
    model = Sequential([
        # Feature extraction layers
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Classification layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Example usage
model = create_basic_cnn((28, 28, 1), 10)  # MNIST-like data
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

## Key Concepts

### Feature Maps
- Output of convolutional layers
- Each filter produces one feature map
- Deeper layers capture more complex features

### Receptive Field
- Area of input that influences a particular output
- Increases with deeper layers
- Important for understanding what the network "sees"

### Parameter Sharing
- Same filter used across entire input
- Reduces parameters compared to fully connected
- Enables translation equivariance

## Common Patterns

### 1. Increasing Depth, Decreasing Size
```python
# Typical progression
Conv2D(32, (3,3)) -> MaxPool -> Conv2D(64, (3,3)) -> MaxPool -> Conv2D(128, (3,3))
```

### 2. Batch Normalization
```python
layers.Conv2D(32, (3, 3)),
layers.BatchNormalization(),
layers.Activation('relu')
```

### 3. Dropout for Regularization
```python
layers.Dropout(0.25)  # After pooling
layers.Dropout(0.5)   # Before final dense layer
```

## Practical Example: Image Classification

```python
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build model
model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
```

## Best Practices

### 1. Data Preprocessing
- Normalize pixel values (0-1 or -1 to 1)
- Data augmentation for small datasets
- Proper train/validation/test splits

### 2. Architecture Design
- Start simple, add complexity gradually
- Use standard patterns (Conv-Pool-Conv-Pool...)
- Consider pre-trained models for transfer learning

### 3. Training Tips
- Use appropriate learning rates
- Monitor validation loss for overfitting
- Apply regularization (dropout, batch norm)
- Save best model checkpoints

## Common Applications
- **Image Classification**: Cat vs Dog, CIFAR-10
- **Object Detection**: YOLO, R-CNN
- **Medical Imaging**: X-ray analysis, MRI segmentation
- **Computer Vision**: Face recognition, OCR

## Quick Reference

| Layer Type | Purpose | Key Parameters |
|------------|---------|----------------|
| Conv2D | Feature extraction | filters, kernel_size, activation |
| MaxPool2D | Dimensionality reduction | pool_size, strides |
| Flatten | Reshape for dense layers | None |
| Dense | Classification/regression | units, activation |
| Dropout | Regularization | rate |

## Next Steps
1. Experiment with different architectures
2. Try transfer learning with pre-trained models
3. Learn about advanced CNN architectures (ResNet, VGG, etc.)
4. Practice with real-world image datasets
