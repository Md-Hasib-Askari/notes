

## 🧱 7. Frameworks – Notes

### 📌 Overview:

Frameworks provide abstractions and utilities to build, train, and deploy deep learning models efficiently. The two leading frameworks are **TensorFlow** and **PyTorch**.

---

### 🧠 7.1 TensorFlow (with Keras)

#### ✅ Highlights:

* Developed by Google
* Ideal for production & deployment
* Integrates well with **TFLite**, **TensorFlow Serving**, and **TensorBoard**

#### ✅ Example:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)
```

#### ✅ Tools:

* `TensorBoard` for visualization
* `TFLite` for mobile/embedded deployment

---

### 🔥 7.2 PyTorch

#### ✅ Highlights:

* Developed by Facebook (Meta)
* Pythonic, dynamic computation graphs
* Preferred for research and flexibility

#### ✅ Example:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

model = Net()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

output = model(input_tensor)
loss = criterion(output, target)
loss.backward()
optimizer.step()
```

#### ✅ Tools:

* `TorchScript` for deployment
* `TensorBoard` integration supported

---

### 🤗 7.3 Hugging Face Transformers

#### ✅ Highlights:

* Pretrained SOTA models for NLP, vision, audio
* Supports PyTorch and TensorFlow
* Thousands of ready-to-use models on [🤗 Model Hub](https://huggingface.co/models)

#### ✅ Example:

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
print(classifier("This is awesome!"))
```

#### ✅ Popular Models:

* BERT, RoBERTa, GPT, T5, CLIP, Whisper

---

### 🧰 7.4 Other Useful Libraries

| Library                   | Purpose                       |
| ------------------------- | ----------------------------- |
| `fastai`                  | High-level PyTorch API        |
| `Lightning`               | PyTorch training abstraction  |
| `Optuna`                  | Hyperparameter tuning         |
| `ONNX`                    | Interoperability + deployment |
| `DeepSpeed`, `Accelerate` | Distributed training          |

