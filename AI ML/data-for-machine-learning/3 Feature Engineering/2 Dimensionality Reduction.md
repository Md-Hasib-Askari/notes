# 📘 Feature Engineering → Dimensionality Reduction

Dimensionality reduction is the process of **reducing the number of features** while keeping as much important information as possible. It helps simplify models, reduce overfitting, and improve computation speed.

---

## **1. Why Dimensionality Reduction?**

* **High-dimensional data problems**:

  * “Curse of dimensionality” → too many features = sparse data, poor generalization.
  * Longer training times and higher storage costs.
  * Risk of multicollinearity (features correlated with each other).
* **Benefits**:

  * Reduces noise in data.
  * Makes visualization possible (2D/3D).
  * Improves model performance for small datasets.

---

## **2. Techniques for Dimensionality Reduction**

### **(a) PCA (Principal Component Analysis)**

* **What it does**: Transforms features into new uncorrelated components that capture maximum variance.
* **When to use**:

  * Numerical features with high correlation.
  * Visualization of datasets in 2D/3D.
* **Limitations**: Components are linear combinations → less interpretable.
* **Implementation**:

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
```

---

### **(b) SVD (Singular Value Decomposition)**

* **What it does**: Factorizes a matrix into lower dimensions.
* **Use Case**:

  * Recommendation systems (user-item matrices).
  * Text processing (Latent Semantic Analysis).
* **Implementation**: Often used via **TruncatedSVD** in sklearn.

```python
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=50)
X_svd = svd.fit_transform(X)
```

---

### **(c) Autoencoders (Neural Networks for Compression)**

* **What they are**: Neural nets that learn to compress input into a smaller latent space, then reconstruct it.
* **Use Case**:

  * Non-linear dimensionality reduction.
  * Image/text embeddings.
  * Anomaly detection.
* **Strength**: Can learn **non-linear patterns** unlike PCA/SVD.

---

### **(d) Feature Selection (Alternative Approach)**

Instead of creating new components, select the most **informative features**.

* **Filter Methods**:

  * Correlation, Chi-square test, Mutual Information.
* **Wrapper Methods**:

  * Recursive Feature Elimination (RFE).
* **Embedded Methods**:

  * Lasso (L1 regularization).
  * Tree-based models (feature importance).

📌 Example (RFE in sklearn):

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(model, n_features_to_select=5)
X_rfe = rfe.fit_transform(X, y)
```

---

## **3. When to Use What?**

* **PCA** → For correlated numerical data & visualization.
* **SVD** → For sparse data like text/user-item matrices.
* **Autoencoders** → For non-linear compression, especially in deep learning.
* **Feature Selection** → When you want interpretability & smaller models.

---

## ✅ Key Takeaways

1. Dimensionality reduction **simplifies models** and reduces overfitting.
2. **PCA & SVD** are classical methods for variance preservation.
3. **Autoencoders** capture complex, non-linear structures.
4. **Feature selection** trims irrelevant features while keeping interpretability.

---