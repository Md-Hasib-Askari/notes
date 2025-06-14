## 📘 Topic 2.3: Training Deep Models

### 🎯 Goal

Learn techniques to avoid:

* Overfitting (memorizing data)
* Underfitting (failing to learn patterns)
* Poor convergence (training that doesn’t go anywhere)

---

### 🔍 1. Overfitting vs. Underfitting

| Type         | Description                                  | Symptoms                               |
| ------------ | -------------------------------------------- | -------------------------------------- |
| Overfitting  | Model performs well on train, poorly on test | High train accuracy, low test accuracy |
| Underfitting | Model performs poorly on both                | Low accuracy everywhere                |

#### 🧪 Solution Strategies:

* **Overfitting?** Add regularization, dropout, or more data.
* **Underfitting?** Use bigger models or train longer.

---

### 🔐 2. Regularization Techniques

#### 🔸 L1 & L2 Regularization

* Adds penalty to large weights in the loss function.
* **L1** promotes sparsity.
* **L2** (weight decay) keeps weights small but dense.

**PyTorch example:**

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)  # L2
```

---

#### 🔸 Dropout

* Randomly zeroes out neurons during training to prevent co-adaptation.
* Forces model to learn redundant, robust features.

**PyTorch:**

```python
self.dropout = nn.Dropout(0.5)
x = self.dropout(x)
```

**TensorFlow:**

```python
tf.keras.layers.Dropout(0.5)
```

---

### ⚖️ 3. Batch Normalization

* Normalizes layer inputs to stabilize and speed up training.
* Reduces internal covariate shift.
* Acts like mild regularization.

**PyTorch:**

```python
self.bn = nn.BatchNorm1d(num_features)
```

**TensorFlow:**

```python
tf.keras.layers.BatchNormalization()
```

---

### 📉 4. Learning Rate Scheduling

Too high → divergence. Too low → slow learning.

#### 🔁 Popular Schedulers:

* **StepLR**
* **ExponentialLR**
* **ReduceLROnPlateau** (adaptive)

**PyTorch Example:**

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
for epoch in range(epochs):
    train(...)
    scheduler.step()
```

**Keras Example:**

```python
tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
```

---

### 🧪 Exercises

#### ✅ Conceptual

1. Why is dropout not used during inference?
2. Compare the effects of L1 vs. L2 regularization.
3. What is internal covariate shift?

---

#### ✅ Practical Challenge

**Task:** Modify your XOR model to include:

* Dropout
* Batch Normalization
* L2 Regularization

Try this in both PyTorch and TensorFlow.
