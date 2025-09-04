# 📘 Intermediate → Data Cleaning (for Machine Learning)

Data cleaning (also called **data preprocessing**) is the process of **detecting and fixing issues in raw data** so that ML models can learn correctly. Dirty data can mislead models and cause poor performance.

---

## **1. Why Data Cleaning Matters**

* ML models assume data is consistent, complete, and structured.
* “Garbage in → Garbage out”: If data is noisy, the model will produce unreliable predictions.
* Industry reports suggest **60–80% of ML project time** is spent on cleaning and preparation.

---

## **2. Handling Missing Values**

* **Causes**: Human error, sensor failures, incomplete entries.
* **Detection**: `df.isnull().sum()` in Pandas.
* **Solutions**:

  * **Remove** rows/columns with too many missing values.
  * **Impute** values:

    * Numerical → mean, median, mode.
    * Categorical → most frequent value.
    * Advanced → KNN imputation, regression-based imputation.

📌 Example (mean imputation in Pandas):

```python
df['Age'].fillna(df['Age'].mean(), inplace=True)
```

---

## **3. Handling Duplicates**

* **Problem**: Duplicate rows inflate counts and bias ML models.
* **Detection**: `df.duplicated().sum()`
* **Solution**:

  ```python
  df.drop_duplicates(inplace=True)
  ```

---

## **4. Handling Inconsistent Data**

* **Examples**:

  * `"Male", "M", "m"` → should be unified.
  * `"USA", "U.S.A.", "United States"` → same country but different labels.
* **Fix**: Standardize categories using mapping or `.replace()`.

📌 Example:

```python
df['Gender'].replace({'M':'Male', 'm':'Male'}, inplace=True)
```

---

## **5. Handling Outliers**

* **Detection Methods**:

  * **Statistical**: Z-score (>3 or <-3), IQR rule.
  * **Visualization**: Boxplots, scatterplots.
* **Solutions**:

  * Remove extreme values.
  * Cap values at thresholds (Winsorization).
  * Transform (e.g., log scaling for skewed data).

---

## **6. Data Type Conversion**

* Wrong data types can cause errors (e.g., numbers stored as strings).
* Convert as needed:

  ```python
  df['Price'] = df['Price'].astype(float)
  df['Date'] = pd.to_datetime(df['Date'])
  ```

---

## **7. Standardizing Data Formats**

* Dates → unify into `YYYY-MM-DD`.
* Currencies → convert into one standard (USD, EUR).
* Strings → lowercase, strip spaces.

---

## **8. Handling Imbalanced Data (Special Case)**

* If one class dominates (e.g., 98% “no fraud”, 2% “fraud”), models may ignore minority class.
* Solutions:

  * **Resampling**: Oversample minority (SMOTE) or undersample majority.
  * **Class weights**: Tell the model to penalize misclassification of minority class more.

---

## ✅ Key Takeaways

1. Data cleaning ensures **consistency, completeness, and reliability**.
2. Missing values → impute or drop.
3. Duplicates & inconsistencies → remove or standardize.
4. Outliers → detect & handle carefully (don’t remove blindly).
5. Correct data types & formats improve model compatibility.
6. Imbalanced data needs **special handling**.

---