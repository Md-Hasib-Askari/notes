## 🏷️ Topic 10: **Image Classification Projects**

### 🎯 Goal:

Build practical projects using everything you've learned — from reading image data to training a CNN and evaluating it.

---

### ✅ Steps in a Classification Project:

1. **Define the Problem**

   * Example: Classify images as cats vs dogs, digits (MNIST), clothes (FashionMNIST), etc.

2. **Load and Preprocess Dataset**

   * Use `torchvision.datasets`, `ImageFolder`, or custom loading.
   * Apply transforms: resize, normalize, augment.

3. **Define the Model**

   * Use custom CNN or fine-tune a pretrained one.

4. **Train the Model**

   * Set loss function and optimizer.
   * Loop over epochs → forward → loss → backward → update.

5. **Evaluate Performance**

   * Accuracy, confusion matrix, precision/recall.
   * Visualize predictions.

6. **Save/Load Model**

   * Save: `torch.save(model.state_dict(), 'model.pth')`
   * Load: `model.load_state_dict(torch.load(...))`

---

### 🧪 Example: Classify MNIST Digits

```python
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch

# 1. Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 2. Model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(x)

model = Net()

# 3. Training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    for images, labels in trainloader:
        preds = model(images)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("Training done.")
```

---

### 🗂️ Datasets to Try:

| Dataset           | Description                  | Link                                |
| ----------------- | ---------------------------- | ----------------------------------- |
| **MNIST**         | 70k grayscale digits         | `torchvision.datasets.MNIST`        |
| **FashionMNIST**  | Clothing categories          | `torchvision.datasets.FashionMNIST` |
| **CIFAR-10**      | 60k color images, 10 classes | `torchvision.datasets.CIFAR10`      |
| **Custom Folder** | Your own images in folders   | `torchvision.datasets.ImageFolder`  |

---

### 🧪 Mini Project Ideas:

* Fruit classifier (apple/orange/banana)
* COVID-19 X-ray classification
* Hotdog vs not-hotdog classifier (yes, it’s a thing)
* Garbage classifier (plastic, paper, metal)
* Handwritten Bangla digit classifier

---

### 📚 Summary:

* Time to **apply** CNNs + data loading + augmentation in real-world tasks.
* Practice with real datasets and try **tweaking** architecture and hyperparameters.
* These projects will prep you for transfer learning and more complex tasks.
