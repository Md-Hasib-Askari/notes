

## 📘 Topic 2.6: Model Evaluation

### 🎯 Goal

Master the key evaluation metrics and diagnostic tools to understand how well your models are doing — and where they’re failing.

---

## 🧠 Core Concepts

### 1. **Confusion Matrix (for classification)**

|                 | Predicted Positive  | Predicted Negative  |
| --------------- | ------------------- | ------------------- |
| Actual Positive | True Positive (TP)  | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN)  |

Helps calculate other metrics.

---

### 2. **Common Metrics**

| Metric      | Formula                                         | Use Case                                       |
| ----------- | ----------------------------------------------- | ---------------------------------------------- |
| Accuracy    | (TP + TN) / (TP + TN + FP + FN)                 | Balanced class distributions                   |
| Precision   | TP / (TP + FP)                                  | Costly false positives (e.g. spam detection)   |
| Recall      | TP / (TP + FN)                                  | Costly false negatives (e.g. cancer detection) |
| F1 Score    | 2 × (Precision × Recall) / (Precision + Recall) | Balanced metric                                |
| Specificity | TN / (TN + FP)                                  | Focus on identifying negatives                 |

---

### 3. **ROC & AUC (Binary Classification)**

* **ROC Curve**: Plots TPR (Recall) vs. FPR
* **AUC (Area Under Curve)**: Closer to 1 is better

> Useful when classes are imbalanced.

---

### 4. **Loss vs. Metric**

| Aspect  | Loss Function             | Evaluation Metric           |
| ------- | ------------------------- | --------------------------- |
| Role    | Used to train the model   | Used to measure performance |
| Example | CrossEntropyLoss, MSELoss | Accuracy, F1, AUC           |

---

### 5. **Evaluation in Regression Tasks**

| Metric                    | Formula                    | Meaning                         |   |                    |
| ------------------------- | -------------------------- | ------------------------------- | - | ------------------ |
| MAE (Mean Absolute Error) | avg(                       \| y\_pred - y\_true               \| ) | Average error size |
| MSE (Mean Squared Error)  | avg((y\_pred - y\_true)^2) | Penalizes larger errors more    |   |                    |
| RMSE (Root MSE)           | sqrt(MSE)                  | Interpretable in original scale |   |                    |
| R² (R-Squared)            | 1 - (RSS/TSS)              | Goodness of fit (1 = perfect)   |   |                    |

---

## 🔧 PyTorch Snippet

```python
from sklearn.metrics import classification_report, confusion_matrix

y_true = [1, 0, 1, 1]
y_pred = [1, 0, 0, 1]

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
```

---

## 🔧 TensorFlow Snippet

```python
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))
```

---

## 🧪 Exercises

### ✅ Conceptual

1. Why is accuracy not a good metric for imbalanced datasets?
2. What’s the trade-off between precision and recall?
3. When should you use F1 score instead of accuracy?

### ✅ Practical

* Train a binary classifier on imbalanced data (e.g. credit fraud).
* Calculate and interpret precision, recall, F1, AUC.
* Plot a confusion matrix and ROC curve.

