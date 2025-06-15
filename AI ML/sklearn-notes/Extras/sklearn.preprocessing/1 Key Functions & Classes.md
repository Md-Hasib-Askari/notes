## 🔹 Key Functions & Classes in `sklearn.preprocessing`

### 🔸 1. **Scaling and Normalization**

Used to bring features to the same scale or range.

* **`StandardScaler`**
  Standardize features by removing the mean and scaling to unit variance.
  → Useful for most ML algorithms (especially ones using distance).

  ```python
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  ```

* **`MinMaxScaler`**
  Scales features to a given range (default \[0, 1]).

  ```python
  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler()
  X_scaled = scaler.fit_transform(X)
  ```

* **`MaxAbsScaler`**
  Scales features by their maximum absolute value. Keeps zero-centered data.

* **`RobustScaler`**
  Uses median and IQR for scaling. Resistant to outliers.

* **`Normalizer`**
  Scales each **sample** (not feature) to unit norm.
  → Useful for text classification or clustering (e.g., cosine similarity).

---

### 🔸 2. **Encoding Categorical Features**

Convert categorical variables into numerical.

* **`LabelEncoder`**
  Converts labels (target) into numeric form.
  Use only for labels, not input features.

  ```python
  from sklearn.preprocessing import LabelEncoder
  le = LabelEncoder()
  y_encoded = le.fit_transform(y)
  ```

* **`OneHotEncoder`**
  One-hot encodes categorical features (use with `ColumnTransformer` for pipelines).

  ```python
  from sklearn.preprocessing import OneHotEncoder
  enc = OneHotEncoder()
  X_encoded = enc.fit_transform(X)
  ```

* **`OrdinalEncoder`**
  Maps categories to integers for **input features**, unlike `LabelEncoder`.

---

### 🔸 3. **Binarization**

* **`Binarizer`**
  Converts numerical features into 0/1 based on a threshold.

  ```python
  from sklearn.preprocessing import Binarizer
  binarizer = Binarizer(threshold=0.0)
  X_bin = binarizer.fit_transform(X)
  ```

---

### 🔸 4. **Generating Polynomial Features**

* **`PolynomialFeatures`**
  Expands features with polynomial combinations (e.g., $x^2, xy$).

  ```python
  from sklearn.preprocessing import PolynomialFeatures
  poly = PolynomialFeatures(degree=2)
  X_poly = poly.fit_transform(X)
  ```

---

### 🔸 5. **Custom Transformations**

* **`FunctionTransformer`**
  Apply custom or NumPy functions as a transformer.

  ```python
  from sklearn.preprocessing import FunctionTransformer
  import numpy as np
  transformer = FunctionTransformer(np.log1p, validate=True)
  X_trans = transformer.fit_transform(X)
  ```

---

## 🔹 Summary Table

| Task                    | Tool                  |
| ----------------------- | --------------------- |
| Standard Scaling        | `StandardScaler`      |
| Min-Max Scaling         | `MinMaxScaler`        |
| Robust Scaling          | `RobustScaler`        |
| Normalizing Vectors     | `Normalizer`          |
| Label Encoding (target) | `LabelEncoder`        |
| One-Hot Encoding        | `OneHotEncoder`       |
| Ordinal Encoding        | `OrdinalEncoder`      |
| Threshold Binarization  | `Binarizer`           |
| Polynomial Expansion    | `PolynomialFeatures`  |
| Custom Transform        | `FunctionTransformer` |

