# PyTorch

## Overview
PyTorch is an open-source deep learning framework developed by Facebook. It is widely used for research and production due to its dynamic computation graph and ease of use.

## Key Features
- **Dynamic Computation Graph**: Enables flexibility in model design and debugging
- **GPU Acceleration**: Seamless integration with CUDA for GPU support
- **Extensive Libraries**: Includes `torchvision`, `torchaudio`, and `torchtext` for specialized tasks
- **Community Support**: Large and active community with extensive documentation

## Installation
```bash
# Install PyTorch
pip install torch torchvision torchaudio

# For GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Basic Workflow

### Tensors
```python
import torch

# Create tensors
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

# Operations
z = x + y
print(z)

# GPU tensors
x_gpu = x.to('cuda')
print(x_gpu)
```

### Neural Networks
```python
import torch
import torch.nn as nn

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Instantiate and use the model
model = SimpleModel()
x = torch.randn(5, 10)
output = model(x)
print(output)
```

### Training Loop
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Data
X = torch.randn(100, 10)
y = torch.randn(100, 1)

# Model
model = nn.Linear(10, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
```

## Advanced Features

### Autograd
```python
import torch

# Automatic differentiation
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2

y.backward()
print(x.grad)  # Gradient of y with respect to x
```

### Custom Layers
```python
import torch
import torch.nn as nn

# Define a custom layer
class CustomLayer(nn.Module):
    def __init__(self):
        super(CustomLayer, self).__init__()

    def forward(self, x):
        return x ** 2

# Use the custom layer
layer = CustomLayer()
x = torch.tensor([1.0, 2.0, 3.0])
output = layer(x)
print(output)
```

### Data Loading
```python
from torch.utils.data import DataLoader, Dataset

# Custom dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# DataLoader
data = torch.randn(100, 10)
labels = torch.randn(100, 1)
dataset = CustomDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

for batch in dataloader:
    print(batch)
```

## Best Practices
1. Use `torch.nn.Module` for model definition.
2. Use `torch.optim` for optimization.
3. Use `torch.utils.data.DataLoader` for efficient data loading.
4. Use `torch.cuda` for GPU acceleration.
5. Use `torch.save` and `torch.load` for model serialization.

## Common Pitfalls
- **Gradient Issues**: Always call `optimizer.zero_grad()` before backpropagation.
- **Memory Management**: Use `.detach()` to avoid unnecessary computation graph retention.
- **Device Compatibility**: Ensure tensors and models are on the same device (CPU/GPU).

## Applications
- **Computer Vision**: Image classification, object detection
- **Natural Language Processing**: Text classification, sequence modeling
- **Reinforcement Learning**: Policy optimization, value function approximation

## Resources
- **PyTorch Documentation**: Official guide and tutorials
- **PyTorch Forums**: Community support
- **TorchVision**: Pre-trained models and datasets for vision tasks
- **TorchText**: Tools for NLP tasks
- **TorchAudio**: Tools for audio processing
