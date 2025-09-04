# 📘 Advanced → Data Quality & Validation (for ML)

High-quality data is essential for reliable ML models. Even the most advanced algorithms cannot perform well if the data is **incomplete, inconsistent, biased, or drifting**. Data validation ensures that training and production data follow expected standards.

---

## **1. Why Data Quality Matters in ML?**

* **Bad data → bad models** ("Garbage in → Garbage out").
* Ensures **consistency** between training and production.
* Prevents **silent model failures** due to data drift.
* Supports **trust & fairness** (avoids bias, discrimination).

---

## **2. Dimensions of Data Quality**

* **Completeness** → No excessive missing values.
* **Consistency** → Same meaning across sources (e.g., “Male/Female” vs “M/F”).
* **Accuracy** → Data reflects reality (no measurement errors).
* **Validity** → Values fall within expected range (e.g., Age > 0).
* **Timeliness** → Data is up-to-date.
* **Uniqueness** → No duplicates.

---

## **3. Data Validation Tools**

* **Great Expectations**

  * Open-source framework for data quality tests.
  * Define expectations (e.g., `Age between 0–120`).
  * Generates validation reports.

📌 Example (Great Expectations rule):

```python
expect_column_values_to_be_between("age", min_value=0, max_value=120)
```

* **Pandera**

  * Python library for data validation with Pandas.
  * Schema-based validation (type checks, constraints).

* **Tfx Data Validation (TFDV)**

  * Google’s tool for validating ML data pipelines.
  * Detects anomalies, schema drift.

---

## **4. Data Bias & Fairness**

* **Sampling bias** → Over/under-represented groups.
* **Label bias** → Incorrect or subjective labels.
* **Algorithmic bias** → Model favors majority group.
* **Mitigation**:

  * Balance datasets (SMOTE, stratified sampling).
  * Fairness metrics (demographic parity, equal opportunity).
  * Bias detection libraries (AIF360, Fairlearn).

---

## **5. Data Drift & Concept Drift**

* **Data Drift** → Input data distribution changes over time.

  * Example: Fraud patterns evolve, making old models obsolete.
* **Concept Drift** → Relationship between features & labels changes.

  * Example: Customer churn drivers may change after new policies.
* **Detection**:

  * Statistical tests (Kolmogorov-Smirnov, Population Stability Index).
  * Monitoring ML metrics (accuracy drop).
* **Mitigation**:

  * Retrain models periodically.
  * Adaptive learning methods.

---

## **6. Best Practices**

* Validate **schema** (columns, types, ranges).
* Set up **automated validation checks** in pipelines.
* Monitor **drift** in production continuously.
* Track & audit **data quality reports**.
* Use **alerting systems** for anomalies.

---

## ✅ Key Takeaways

1. Data quality = foundation of reliable ML.
2. Validate for **completeness, accuracy, consistency, timeliness**.
3. Use tools like **Great Expectations, Pandera, TFDV**.
4. Detect & mitigate **bias and drift** proactively.
5. Ongoing monitoring is as important as one-time validation.

---