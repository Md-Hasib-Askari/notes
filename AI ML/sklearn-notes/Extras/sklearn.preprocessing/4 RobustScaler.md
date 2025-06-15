## 🔹 `RobustScaler` – Scaling with Median and IQR

### ✅ Purpose:

* Scales features using **median** and **interquartile range (IQR)**.
* Makes it **robust to outliers**, unlike `StandardScaler` or `MinMaxScaler`.

---

### 📌 Formula:

$$
x' = \frac{x - \text{median}}{\text{IQR}}
$$

Where:

* IQR = Interquartile Range = 75th percentile − 25th percentile

---

### 🔧 How to Use:

```python
from sklearn.preprocessing import RobustScaler
import numpy as np

# Data with outlier
X = np.array([[1, 2], [2, 5], [3, 6], [100, 200]], dtype=float)

# Apply robust scaling
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled)
```

---

### 🔍 Output (approx):

```plaintext
[[-0.5  -0.75 ]
 [-0.25  0.   ]
 [ 0.     0.25]
 [12.25  12.5 ]]
```

* The outlier (`[100, 200]`) doesn’t skew the scale of the rest of the data.
* The majority of values are compressed near 0.

---

### 📌 Use Case:

When your dataset contains:

* Salary data
* Housing prices
* Sensor data with spikes
* Any feature with extreme values

---

### ⚠️ Notes:

* Like all scalers: fit on training data, transform both train and test.
* Still transforms to **centered around median**, not mean.
