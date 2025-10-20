# 📘 Intermediate → Feature Scaling (for Machine Learning)

Feature scaling is the process of **transforming numerical features to a similar scale** so that machine learning models can learn effectively.
Some ML algorithms are sensitive to feature magnitude (e.g., distance-based models), while others are scale-invariant (e.g., tree-based models).

---

## **1. Why Feature Scaling Matters?**

* **Ensures fairness** → Features with large ranges won’t dominate smaller ones.
* **Improves convergence speed** → Gradient descent converges faster with scaled data.
* **Improves performance** → Especially in algorithms that use distances, dot products, or optimization.
* **Required for**:

  * KNN, SVM, K-Means, PCA, Logistic Regression, Neural Networks.
* **Not required for**:

  * Tree-based models (Decision Trees, Random Forests, XGBoost).

---

## **2. Common Feature Scaling Techniques**

### **(a) Min-Max Scaling (Normalization)**

* **Formula**:

  $$
  X' = \frac{X - X_{min}}{X_{max} - X_{min}}
  $$
* **Range**: \[0, 1] (can be adjusted).
* **Best for**: Neural networks (inputs bounded between 0 and 1).
* **Cons**: Sensitive to outliers.

📌 Example:

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[['Age','Salary']])
```

---

### **(b) Standardization (Z-Score Scaling)**

* **Formula**:

  $$
  X' = \frac{X - \mu}{\sigma}
  $$

  where μ = mean, σ = standard deviation.
* **Range**: Centered at 0, with unit variance.
* **Best for**: Algorithms assuming normal distribution (Logistic Regression, Linear Regression, SVM).
* **Cons**: Outliers still affect mean/variance.

📌 Example:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Age','Salary']])
```

---

### **(c) Robust Scaling**

* **Formula**:

  $$
  X' = \frac{X - \text{Median}}{IQR}
  $$

  (IQR = Interquartile Range).
* **Range**: No fixed range, but less sensitive to outliers.
* **Best for**: Data with many outliers (financial transactions, sensor readings).

📌 Example:

```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df_scaled = scaler.fit_transform(df[['Age','Salary']])
```

---

### **(d) Max-Abs Scaling**

* Scales values to range \[-1, 1].
* Keeps **sparsity** in sparse datasets (e.g., TF-IDF).
* Best for high-dimensional text data.

📌 Example:

```python
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df_scaled = scaler.fit_transform(df[['Feature1','Feature2']])
```

---

### **(e) Log Transform**

* Used for **skewed data** (e.g., income, population).
* Reduces variance, makes distribution closer to normal.

📌 Example:

```python
import numpy as np
df['Income_log'] = np.log1p(df['Income'])
```

---

## **3. When to Scale?**

* Always **after train-test split** → avoid data leakage.
* Fit scaler on **training data only**, then transform both training & test.

---

## ✅ Key Takeaways

1. **Normalization (Min-Max)** → Best for NN, bounded inputs.
2. **Standardization (Z-score)** → Default choice, works in most ML models.
3. **Robust Scaling** → Use when outliers are present.
4. **Max-Abs Scaling** → Good for sparse, high-dimensional data.
5. **Log transform** → Fix skewed distributions.

---