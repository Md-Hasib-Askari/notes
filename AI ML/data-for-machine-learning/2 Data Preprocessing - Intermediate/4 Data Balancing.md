# 📘 Intermediate → Data Balancing (for Machine Learning)

In many ML tasks, datasets are **imbalanced**: one class has far more samples than others. This can mislead models into **biasing predictions toward the majority class**, ignoring the minority.
Example:

* Fraud detection → 98% legitimate, 2% fraud.
* Medical diagnosis → 95% healthy, 5% disease.

---

## **1. Why Class Imbalance is a Problem?**

* Standard metrics (like accuracy) become misleading.

  * Example: A model predicting **“no fraud” for all transactions** achieves 98% accuracy but is useless.
* Models fail to **learn minority class patterns**.
* Real-world cost is usually higher for misclassifying minority class (e.g., missing a fraud case).

---

## **2. Detecting Imbalance**

* Class distribution check:

  ```python
  df['label'].value_counts(normalize=True)
  ```
* Visualization: bar plots, pie charts.
* Rule of thumb: If minority class < 10–15%, imbalance needs attention.

---

## **3. Solutions for Data Imbalance**

### **(a) Resampling Techniques**

* **Oversampling** (add more minority samples)

  * Random Oversampling → duplicate minority samples.
  * **SMOTE (Synthetic Minority Oversampling Technique)** → creates synthetic samples by interpolating existing minority samples.
  * Variants: Borderline-SMOTE, ADASYN.
* **Undersampling** (remove some majority samples)

  * Random Undersampling → drop majority class rows.
  * Tomek Links, NearMiss → keep only informative majority samples.
* **Hybrid** → Combine oversampling + undersampling for balance.

📌 Example (SMOTE in Python):

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

---

### **(b) Algorithm-Level Approaches**

* **Class Weights**: Assign higher penalty to minority misclassification.

  * Supported in Logistic Regression, SVM, Random Forest, XGBoost.
* **Ensemble Methods**:

  * Balanced Random Forests, EasyEnsemble.
* **Anomaly Detection Models**:

  * For extreme imbalance (e.g., fraud detection), treat minority class as anomaly detection.

📌 Example (class weights in Logistic Regression):

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced')
```

---

### **(c) Data Augmentation**

* For unstructured data (text, images, audio):

  * **Text**: synonym replacement, back translation.
  * **Images**: rotations, flips, brightness changes.
  * **Audio**: noise injection, pitch shifting.

---

## **4. Evaluation Metrics for Imbalanced Data**

Accuracy is not enough — use metrics sensitive to minority class:

* **Precision** = TP / (TP + FP)
* **Recall (Sensitivity)** = TP / (TP + FN)
* **F1 Score** = Harmonic mean of precision & recall
* **ROC-AUC** (measures class separability)
* **PR-AUC** (better for high imbalance)

📌 Example (sklearn classification report):

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

---

## ✅ Key Takeaways

1. Imbalanced data causes models to ignore minority classes.
2. Fix using **resampling (SMOTE, undersampling)** or **algorithm-level tweaks (class weights, ensembles)**.
3. For text/image/audio, apply **data augmentation**.
4. Always evaluate with **F1, ROC-AUC, PR-AUC**, not just accuracy.

---
