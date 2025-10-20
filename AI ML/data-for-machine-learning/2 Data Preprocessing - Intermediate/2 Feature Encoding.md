# 📘 Intermediate → Feature Encoding (for Machine Learning)

Most ML models can only work with **numerical inputs**. Since real-world data often contains **categorical, textual, or mixed features**, **feature encoding** is the process of converting them into a numerical form suitable for ML models.

---

## **1. Why Feature Encoding is Important?**

* ML algorithms (e.g., Linear Regression, SVM, Neural Nets) require numeric values.
* Encoding categorical/text features allows models to detect patterns.
* Poor encoding can lead to **information loss** or **high dimensionality**.

---

## **2. Types of Variables That Need Encoding**

* **Categorical**:

  * Nominal (unordered, e.g., color: red, blue, green)
  * Ordinal (ordered, e.g., education: high school < bachelor < master < PhD)
* **Text**:

  * Sentences, reviews, documents
* **Mixed Formats**:

  * Dates/times → day, month, year, weekday

---

## **3. Encoding Techniques for Categorical Features**

### **(a) Label Encoding**

* Assigns a unique integer to each category.
* Example:

  ```
  Color: Red → 0, Blue → 1, Green → 2
  ```
* **Pros**: Simple, no extra dimensions.
* **Cons**: Implies an **order** (Red < Blue < Green), which may mislead models.
* **Best for**: Ordinal features.

📌 Example:

```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['Color'] = encoder.fit_transform(df['Color'])
```

---

### **(b) One-Hot Encoding**

* Creates a new binary column for each category.
* Example:

  ```
  Color: Red → [1,0,0], Blue → [0,1,0], Green → [0,0,1]
  ```
* **Pros**: No false ordering.
* **Cons**: High dimensionality if many categories (curse of dimensionality).
* **Best for**: Nominal features with few unique values.

📌 Example:

```python
pd.get_dummies(df['Color'])
```

---

### **(c) Target / Mean Encoding**

* Replace category with mean of target variable.
* Example (churn dataset):

  ```
  City A → 0.12 churn rate, City B → 0.35, City C → 0.50
  ```
* **Pros**: Captures category–target relationship.
* **Cons**: Can cause **data leakage** (must use only training data statistics).
* **Best for**: High-cardinality categorical features.

---

### **(d) Frequency / Count Encoding**

* Replace category with its frequency/count.
* Example:

  ```
  Color: Red (50), Blue (30), Green (20)
  ```
* **Pros**: Keeps dimensionality low.
* **Cons**: May lose target relationship.
* **Best for**: Large datasets with many unique categories.

---

### **(e) Binary Encoding**

* Convert categories into binary numbers.
* Example: Category “7” → binary `111`.
* **Pros**: Reduces dimensionality compared to one-hot.
* **Cons**: Slight interpretability loss.
* **Best for**: High-cardinality categorical features.

---

## **4. Encoding Techniques for Text Features**

* **Bag of Words (BoW)** → Count occurrences of words.
* **TF-IDF (Term Frequency–Inverse Document Frequency)** → Weighs words based on importance.
* **Embeddings (Word2Vec, GloVe, BERT)** → Capture semantic meaning of text.
* **Best for**: NLP tasks like sentiment analysis, classification, summarization.

---

## **5. Encoding Date/Time Features**

* Extract meaningful features:

  * Year, month, day, weekday, quarter, hour, season.
  * Time difference (e.g., delivery time = delivery\_date − order\_date).
* Cyclical encoding (sin/cos transforms) for periodic features like hours, days.

📌 Example:

```python
df['day_of_week'] = df['date'].dt.dayofweek
df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
```

---

## ✅ Key Takeaways

1. **Label Encoding** → Ordinal data.
2. **One-Hot Encoding** → Small categorical features.
3. **Target/Frequency/Binary Encoding** → High-cardinality features.
4. **Text features** → Use BoW, TF-IDF, or embeddings.
5. **Date/Time** → Extract engineered features (cyclical encoding when needed).

---