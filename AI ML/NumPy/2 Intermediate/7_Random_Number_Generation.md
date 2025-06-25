## 🟡 **7. Random Number Generation in NumPy**

NumPy provides powerful tools to generate random numbers for simulations, testing, and probabilistic modeling via `np.random`.

---

### 🔹 1. **Basic Random Functions**

```python
import numpy as np
```

#### 🔸 `np.random.rand()`

* Generates random floats in the half-open interval \[0.0, 1.0)
* Uniform distribution

```python
np.random.rand(3)        # [0.67 0.23 0.89]
np.random.rand(2, 3)     # 2x3 array
```

#### 🔸 `np.random.randn()`

* Samples from the **standard normal distribution** (mean = 0, std = 1)

```python
np.random.randn(3)       # e.g. [0.65 -1.2 0.05]
np.random.randn(2, 2)
```

#### 🔸 `np.random.randint()`

* Generates random **integers** between `low` (inclusive) and `high` (exclusive)

```python
np.random.randint(1, 10)         # Single integer between 1 and 9
np.random.randint(1, 10, size=5) # Array of 5 integers
```

---

### 🔹 2. **Random Choice from an Array**

```python
a = np.array([10, 20, 30, 40])

# Randomly select one value
np.random.choice(a)

# Select 3 values with replacement
np.random.choice(a, size=3)

# Without replacement
np.random.choice(a, size=3, replace=False)
```

---

### 🔹 3. **Shuffling and Permutation**

#### 🔸 `np.random.shuffle()`

* Shuffles the array **in-place**

```python
a = np.array([1, 2, 3, 4])
np.random.shuffle(a)
```

#### 🔸 `np.random.permutation()`

* Returns a **shuffled copy** of the array

```python
a = np.array([1, 2, 3, 4])
shuffled = np.random.permutation(a)
```

---

### 🔹 4. **Set Random Seed**

* Ensures reproducibility of results (e.g. for testing and debugging)

```python
np.random.seed(42)

print(np.random.rand(2))     # Always same output if seed is same
```

---

### 🔹 5. **Distributions in `np.random`**

You can generate samples from various distributions:

| Distribution      | Function                  | Example                                    |
| ----------------- | ------------------------- | ------------------------------------------ |
| Uniform           | `np.random.uniform()`     | `np.random.uniform(1, 10, size=5)`         |
| Normal (Gaussian) | `np.random.normal()`      | `np.random.normal(loc=0, scale=1, size=5)` |
| Binomial          | `np.random.binomial()`    | `np.random.binomial(n=10, p=0.5, size=5)`  |
| Poisson           | `np.random.poisson()`     | `np.random.poisson(lam=3, size=5)`         |
| Beta              | `np.random.beta()`        | `np.random.beta(a=2, b=5, size=5)`         |
| Exponential       | `np.random.exponential()` | `np.random.exponential(scale=1.0, size=5)` |

---

### 📝 Summary

| Task                      | Function Example                                  |
| ------------------------- | ------------------------------------------------- |
| Random floats (0 to 1)    | `np.random.rand(3, 2)`                            |
| Random normal values      | `np.random.randn(4)`                              |
| Random integers           | `np.random.randint(0, 100, size=10)`              |
| Random choice             | `np.random.choice([1, 2, 3], size=2)`             |
| Shuffle array in-place    | `np.random.shuffle(arr)`                          |
| Get random permutation    | `np.random.permutation(arr)`                      |
| Set random seed           | `np.random.seed(42)`                              |
| Sample from distributions | `np.random.normal()`, `np.random.binomial()` etc. |
