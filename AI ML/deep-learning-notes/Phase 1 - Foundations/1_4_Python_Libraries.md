## 📘 Topic 1.4: **Python + Libraries (NumPy, Pandas, Matplotlib)**

These are the **essential tools** for handling data and performing numerical operations in deep learning projects.

---

### 🧮 NumPy (Numerical Python)

#### 🔑 Key Concepts

* **Arrays** (vectors, matrices, tensors)
* **Broadcasting** (automatic expansion of dimensions)
* **Linear algebra operations**
* **Random number generation**
* **Vectorization** (fast computation without loops)

#### ✅ Examples

```python
import numpy as np

# Array creation
a = np.array([1, 2, 3])
b = np.array([[1, 2], [3, 4]])

# Shape and reshape
print(a.shape)
b_reshaped = b.reshape(1, 4)

# Broadcasting
x = np.array([1, 2, 3])
y = x + 10  # broadcasts 10 to each element

# Dot product
c = np.dot(a, a)  # 1*1 + 2*2 + 3*3

# Random
rand_nums = np.random.randn(3, 3)
```

---

### 📊 Pandas (Data Analysis)

#### 🔑 Key Concepts

* **Series**: 1D labeled array
* **DataFrame**: 2D table with columns
* **Indexing**, **filtering**, **grouping**, **aggregation**
* **Missing data handling**

#### ✅ Examples

```python
import pandas as pd

# Create DataFrame
df = pd.DataFrame({
    'name': ['A', 'B', 'C'],
    'score': [90, 85, 78],
    'passed': [True, True, False]
})

# Accessing data
print(df['score'])     # Column
print(df.iloc[1])      # Row by index

# Filtering
print(df[df['score'] > 80])

# Grouping
grouped = df.groupby('passed').mean()
```

---

### 📈 Matplotlib (Plotting)

#### 🔑 Key Concepts

* **Line plots**, **bar charts**, **histograms**, **scatter plots**
* Customize with titles, legends, axes labels

#### ✅ Examples

```python
import matplotlib.pyplot as plt

# Line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title("Sine Wave")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.show()

# Scatter plot
x = np.random.rand(100)
y = np.random.rand(100)
plt.scatter(x, y)
plt.title("Random Scatter")
plt.show()
```

---

### 🧠 Intuition for Deep Learning

* **NumPy** powers tensor operations and model math (under the hood of PyTorch/TensorFlow)
* **Pandas** helps with preprocessing and feature engineering
* **Matplotlib** is key for debugging, visualizing training curves, and data insights

---

### 🧪 Exercises

#### ✅ Coding Tasks

1. Load a CSV file with `pandas`, filter rows, and plot a histogram of a numerical column.
2. Create a 2D NumPy array, compute its transpose, inverse (if square), and eigenvalues.
3. Simulate 1000 samples from a normal distribution and plot them as a histogram with Matplotlib.
