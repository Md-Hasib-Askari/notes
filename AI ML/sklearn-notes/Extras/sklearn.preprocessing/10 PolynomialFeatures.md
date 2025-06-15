## 🔹 `PolynomialFeatures` – Generate Polynomial & Interaction Terms

### ✅ Purpose:

* Creates new features by computing **polynomials and interactions** of existing features.
* Helps linear models learn **non-linear patterns**.

---

### 📌 What It Does:

Given input features $x_1, x_2$, it can generate:

#### Degree = 2:

<div align="center">
  <img src="https://github.com/user-attachments/assets/3c8898df-2783-423e-bfe1-f26e9f82a3c0" alt="" />
</div>

#### Degree = 3:

<div align="center">
  <img src="https://github.com/user-attachments/assets/fb7b5e46-6724-43ce-bef7-1bd202b6b607" alt="" />
</div>

---

### 🔧 How to Use:

```python
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Sample data (2 features)
X = np.array([[2, 3]])

# Degree-2 polynomial expansion
poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(X)

print(X_poly)
```

---

### 🔍 Output:

```plaintext
[[1. 2. 3. 4. 6. 9.]]
```

Which corresponds to:

* ```1, x₁, x₂, x₁², x₁·x₂, x₂²```
* ```1, 2, 3, 4, 6, 9```

---

### 🔄 Options:

* `degree`: controls the polynomial degree (default = 2)
* `include_bias=True`: includes the constant 1
* `interaction_only=True`: only includes interaction terms (no powers)

```python
# Only interaction terms (no x^2, just x1*x2)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
```

---

### 📌 Use Cases:

* **Linear regression + polynomial features** → captures **non-linear curves**.
* Works great with **regularization** (e.g., Ridge, Lasso).
* Boosts feature expressiveness in **small to medium datasets**.

---

### 🧠 Example with Linear Regression:

```python
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
model.fit(X_train, y_train)
```

Now the linear regression can fit **cubic curves** or higher!

---

### ⚠️ Notes:

* Quickly **increases dimensionality** — higher degrees can lead to overfitting.
* Combine with **feature selection** or **regularization** if using large degrees.
* Use with **numeric features** only.

