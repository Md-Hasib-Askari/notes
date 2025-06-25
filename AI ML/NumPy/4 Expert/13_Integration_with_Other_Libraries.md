## 🔴 **13. Integration with Other Libraries**

NumPy is the foundational building block of the scientific Python ecosystem. Understanding how it interacts with other libraries boosts your data science, ML, and scientific computing workflow.

---

### 🔹 1. **NumPy + Pandas**

#### Convert between arrays and DataFrames:

```python
import pandas as pd

# NumPy to DataFrame
arr = np.array([[1, 2], [3, 4]])
df = pd.DataFrame(arr, columns=['A', 'B'])

# DataFrame to NumPy
arr_back = df.to_numpy()
```

> ✅ Use Pandas for labeled data operations; NumPy underlies all of it.

---

### 🔹 2. **NumPy + Matplotlib / Seaborn**

#### Example: Plot a sine wave

```python
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title("Sine Wave")
plt.show()
```

> ✅ NumPy arrays are the default data format for Matplotlib and Seaborn.

---

### 🔹 3. **NumPy + SciPy**

SciPy builds on NumPy and provides **advanced scientific computations**, like:

* Integration: `scipy.integrate.quad`
* Optimization: `scipy.optimize`
* Signal processing: `scipy.signal`
* Linear algebra: `scipy.linalg` (extends `numpy.linalg`)

```python
from scipy import integrate

result, _ = integrate.quad(np.sin, 0, np.pi)
print(result)  # ~2.0
```

> ✅ Use SciPy when you need more than basic numerical tools.

---

### 🔹 4. **NumPy + TensorFlow / PyTorch**

Deep learning frameworks like TensorFlow and PyTorch **interoperate with NumPy** seamlessly.

#### PyTorch example:

```python
import torch

a = np.array([[1, 2], [3, 4]])
t = torch.from_numpy(a)         # NumPy → Tensor

b = t.numpy()                   # Tensor → NumPy
```

#### TensorFlow example:

```python
import tensorflow as tf

a = np.array([1, 2, 3])
t = tf.convert_to_tensor(a)

b = t.numpy()
```

> 🚨 Tensors and NumPy arrays **share memory**, so modifying one can affect the other (unless copied).

---

### 🔹 5. **NumPy + scikit-learn**

Scikit-learn accepts NumPy arrays for training ML models:

```python
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3]])
y = np.array([2, 4, 6])

model = LinearRegression()
model.fit(X, y)
```

---

### 🔹 6. **General Interconversion Summary**

| From               | To                             | Conversion Tool |
| ------------------ | ------------------------------ | --------------- |
| NumPy → Pandas     | `pd.DataFrame()`               |                 |
| Pandas → NumPy     | `df.to_numpy()` or `df.values` |                 |
| NumPy → Torch      | `torch.from_numpy()`           |                 |
| Torch → NumPy      | `tensor.numpy()`               |                 |
| NumPy → TensorFlow | `tf.convert_to_tensor()`       |                 |
| TensorFlow → NumPy | `tensor.numpy()`               |                 |

---

### 📝 Summary

| Library      | Role                      | Integration Style                      |
| ------------ | ------------------------- | -------------------------------------- |
| Pandas       | DataFrames / tabular data | Seamless conversion to/from NumPy      |
| Matplotlib   | Visualization             | Takes NumPy arrays as input            |
| SciPy        | Scientific computing      | Built directly on top of NumPy         |
| PyTorch / TF | Deep learning             | Tensor/array interconversion           |
| scikit-learn | ML pipeline               | Takes NumPy arrays for features/labels |
