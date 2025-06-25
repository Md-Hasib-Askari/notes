## 🔵 **11. Masked Arrays and NaN Handling in NumPy**

When dealing with real-world data, you'll often encounter missing or invalid values. NumPy provides tools to handle such data using **NaNs** and **masked arrays**.

---

### 🔹 1. **Handling NaN (Not a Number)**

#### 🔸 Create arrays with `np.nan`:

```python
a = np.array([1.0, 2.0, np.nan, 4.0])
```

#### 🔸 Check for NaNs:

```python
np.isnan(a)    # [False False  True False]
```

#### 🔸 Remove or ignore NaNs:

```python
a[~np.isnan(a)]     # [1. 2. 4.]
```

---

### 🔹 2. **NaN-Safe Statistical Functions**

| Function             | NaN-Safe Version           | Description                   |
| -------------------- | -------------------------- | ----------------------------- |
| `np.sum()`           | `np.nansum()`              | Ignores NaNs in sum           |
| `np.mean()`          | `np.nanmean()`             | Ignores NaNs in mean          |
| `np.std()`           | `np.nanstd()`              | Ignores NaNs in std deviation |
| `np.min()` / `max()` | `np.nanmin()` / `nanmax()` | Ignores NaNs in min/max       |

#### Example:

```python
a = np.array([1, 2, np.nan, 4])
np.nanmean(a)     # 2.333...
```

---

### 🔹 3. **Filling NaNs**

```python
a = np.array([1, 2, np.nan, 4])
a = np.nan_to_num(a, nan=0)   # Replace NaN with 0
```

Or replace with mean:

```python
a = np.array([1, 2, np.nan, 4])
mean_val = np.nanmean(a)
a[np.isnan(a)] = mean_val
```

---

### 🔹 4. **Masked Arrays (`numpy.ma`)**

#### 🔸 What is a masked array?

It is an array that **ignores or masks** certain values during computations.

```python
import numpy.ma as ma

data = np.array([1, 2, 3, -999, 5])
masked_data = ma.masked_equal(data, -999)
```

> `-999` is treated as missing and ignored.

#### 🔸 Basic masked operations:

```python
masked_data.mean()    # Ignores -999
masked_data.sum()
masked_data.filled(0) # Replace mask with 0: [1 2 3 0 5]
```

---

### 🔹 5. **Creating Masked Arrays Manually**

```python
data = np.array([1, 2, 3, 4])
mask = np.array([0, 1, 0, 1])   # 1 means masked

m = ma.array(data, mask=mask)
```

---

### 🔹 6. **Masking with Conditions**

```python
a = np.array([1, 5, 10, -1])
masked = ma.masked_where(a < 0, a)   # Mask negatives
```

---

### 🔹 7. **Convert Masked to Regular Arrays**

```python
masked_data.filled(np.nan)   # Fill mask with NaNs
```

---

### 📝 Summary

| Feature              | Function / Example              |
| -------------------- | ------------------------------- |
| Detect NaNs          | `np.isnan(a)`                   |
| Ignore NaNs in stats | `np.nanmean(a)`, `np.nansum(a)` |
| Replace NaNs         | `np.nan_to_num(a, nan=0)`       |
| Mask specific values | `ma.masked_equal(data, -999)`   |
| Fill masked values   | `masked_data.filled(0)`         |
| Mask conditionally   | `ma.masked_where(a < 0, a)`     |
