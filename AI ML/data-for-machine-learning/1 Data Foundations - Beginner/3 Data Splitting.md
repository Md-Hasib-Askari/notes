# 📘 Beginner → Data Splitting (for Machine Learning)

Splitting data is one of the most **fundamental steps** in ML. It ensures that a model is trained, validated, and tested fairly — without “seeing” the answers in advance.

---

## **1. Why Split Data?**

* **Prevent overfitting** → The model may memorize training data if not evaluated properly.
* **Fair evaluation** → We test the model on unseen data to measure real-world performance.
* **Hyperparameter tuning** → Validation sets allow adjusting model parameters without touching the test set.

---

## **2. Train-Test Split**

* **Basic approach**: Divide dataset into two parts.

  * **Training set** → Used to train the model (usually 70–80%).
  * **Test set** → Used to evaluate the final model (usually 20–30%).

📌 Example (scikit-learn):

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

* **Parameters**:

  * `test_size=0.2` → 20% of data is test.
  * `random_state=42` → ensures reproducibility.

---

## **3. Train-Validation-Test Split**

* Sometimes, a separate **validation set** is needed:

  * **Training set** → Model learns patterns.
  * **Validation set** → Used during training to tune hyperparameters.
  * **Test set** → Only used once at the end, for unbiased evaluation.

📌 Common split: **60% train / 20% validation / 20% test**.

---

## **4. Cross-Validation (CV)**

* Instead of a single split, CV divides data into multiple folds:

  * **K-Fold CV**: Split data into *k* folds (e.g., 5). Train on (k−1) folds and test on the remaining one. Repeat k times.
  * **Stratified K-Fold**: Ensures class proportions remain consistent across folds (important for imbalanced datasets).

📌 Example (5-fold CV):

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=5)
print("Average Accuracy:", scores.mean())
```

---

## **5. Time-Series Splitting**

* For **time-series data**, random shuffling is not allowed.
* Instead, use **rolling/expanding window splits** where past data predicts the future.

📌 Example:

* Train: Jan–June → Test: July
* Train: Jan–July → Test: Aug

---

## **6. Best Practices**

* Always keep the **test set untouched** until the very end.
* Use **cross-validation** for small datasets.
* Use **stratified splits** for imbalanced classification problems.
* For **time series**, always respect temporal order.

---

## ✅ Key Takeaways

1. **Train-Test** is the simplest split.
2. **Validation sets** help tune models without leaking test data.
3. **Cross-Validation** provides robust evaluation for small/medium datasets.
4. **Time-series data** needs special splitting techniques.

---