# 📘 Feature Engineering → Feature Creation

Feature creation is about **deriving new variables** from raw data to make hidden patterns more explicit for ML models. Well-engineered features can drastically improve accuracy, even with simple models.

---

## **1. Polynomial Features**

* **What it is**: Extend existing numerical features by adding polynomial terms.

  * Example: If a dataset has `x`, create `x²`, `x³`, etc.
  * Also includes **interaction terms**: `x × y`.
* **Why**: Helps linear models capture **non-linear relationships**.
* **When to use**:

  * When scatter plots show curved patterns.
  * Small/medium datasets (can cause overfitting on small data if too many terms).
* **Implementation**:

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=False)
X_poly = poly.fit_transform(X)
```

---

## **2. Interaction Features**

* **What it is**: Combine two or more features to capture joint effects.

  * Example: In housing prices → `rooms × location_quality` may be more predictive than either alone.
* **Why**: Real-world outcomes often depend on **combinations** of factors.
* **Example Use Case**:

  * Marketing: `(ad_spend × season)` → captures seasonal ad performance.

---

## **3. Aggregated Features (Group-Level Statistics)**

* **What it is**: Summarize groups of data into higher-level statistics.
* **Examples**:

  * Customer transactions → mean purchase value, total spend, purchase frequency.
  * Website logs → average session duration per user.
* **Why**: Aggregations reveal **behavioral patterns** beyond individual records.
* **Implementation**:

```python
df.groupby('customer_id')['purchase_amount'].agg(['mean','max','count'])
```

---

## **4. Time-Based Features**

* **What it is**: Features derived from timestamps or sequences.
* **Examples**:

  * **Lag features**: yesterday’s sales to predict today’s.
  * **Rolling averages**: 7-day moving average of stock prices.
  * **Seasonal features**: day of week, month, quarter, holidays.
* **Why**: Time carries **hidden patterns** like trends, seasonality, and recency.
* **Implementation**:

```python
df['lag_1'] = df['sales'].shift(1)
df['rolling_mean_7'] = df['sales'].rolling(window=7).mean()
```

---

## ✅ Key Takeaways

1. **Polynomial & interaction features** → capture non-linear effects.
2. **Aggregations** → summarize patterns at group level.
3. **Time-based features** → capture trends, seasonality, and dependencies.
4. Always check for **overfitting** (too many generated features can hurt).

---