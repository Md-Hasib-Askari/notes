

## 🤖 4. Machine Learning Basics – Notes

### 📌 Overview:

Machine Learning (ML) is a subset of AI where systems learn from data to make predictions or decisions without being explicitly programmed for specific tasks.

---

### 🧠 4.1 Types of ML

#### ✅ Supervised Learning:

* Input: Labeled data (features + target)
* Goal: Predict target values
* Algorithms: Linear Regression, Logistic Regression, SVM, Decision Trees

#### ✅ Unsupervised Learning:

* Input: Unlabeled data
* Goal: Find hidden patterns
* Algorithms: K-Means, DBSCAN, PCA

#### ✅ Semi-Supervised & Self-Supervised:

* Limited labeled data + lots of unlabeled
* Common in real-world scenarios (e.g., NLP pretraining)

#### ✅ Reinforcement Learning (basic mention here, deep dive later):

* Agent learns by interacting with an environment and receiving rewards

---

### 📈 4.2 Key Algorithms

#### ✅ Regression:

* Predicts continuous output
* Algorithms: Linear Regression, Ridge/Lasso, Decision Tree Regressor

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, y)
```

#### ✅ Classification:

* Predicts categories
* Algorithms: Logistic Regression, KNN, SVM, Random Forest

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(X_train, y_train)
```

#### ✅ Clustering:

* Groups similar data points
* Algorithms: K-Means, Agglomerative Clustering

---

### 🧪 4.3 Model Evaluation

#### ✅ Regression Metrics:

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* R² Score

#### ✅ Classification Metrics:

* Accuracy, Precision, Recall, F1-Score
* Confusion Matrix
* ROC Curve & AUC Score

```python
from sklearn.metrics import accuracy_score, confusion_matrix
```

---

### 🧰 4.4 Train/Test Split & Cross-Validation

#### ✅ Train-Test Split:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

#### ✅ Cross-Validation:

* K-Fold CV for robust performance estimation

```python
from sklearn.model_selection import cross_val_score
```

---

### 📊 4.5 Bias-Variance Tradeoff

| Concept | Low Bias / High Variance | High Bias / Low Variance |
| ------- | ------------------------ | ------------------------ |
| Model   | Overfitting              | Underfitting             |

* Goal: Find the sweet spot with low bias *and* low variance.
