# 📘 Beginner → Basic Data Handling (for Machine Learning)

Before applying ML algorithms, you need to know how to **load, explore, and summarize data**. This step ensures you understand the dataset and detect potential issues.

---

## **1. Loading Data**

* **Sources**:

  * **Flat files**: CSV, TXT, Excel
  * **Databases**: SQL queries (PostgreSQL, MySQL, SQLite)
  * **APIs/Web scraping**: JSON, XML feeds
  * **Cloud storage**: AWS S3, Google Drive, BigQuery
* **Tools**:

  * Python: `pandas.read_csv()`, `read_excel()`, `read_sql()`
  * Spark/Dask: for big datasets that don’t fit in memory

📌 **Example (Pandas)**

```python
import pandas as pd
df = pd.read_csv("data.csv")
```

---

## **2. Exploring Data (First Look)**

* **Check dimensions**: rows & columns (`df.shape`)
* **Check data types**: (`df.info()`)
* **View sample records**: (`df.head()`)
* **Check unique values**: (`df['column'].unique()`)
* **Descriptive statistics**: (`df.describe()`)

👉 Helps identify categorical vs numerical features, missing values, and anomalies.

---

## **3. Summary Statistics**

* **Central tendency**: Mean, median, mode
* **Dispersion**: Variance, standard deviation, range, interquartile range (IQR)
* **Distribution shape**: Skewness, kurtosis
* **Correlation**: Pearson, Spearman (to detect feature relationships)

📌 Example: Checking correlation between features

```python
df.corr()
```

---

## **4. Data Visualization (Quick EDA)**

* **Univariate**:

  * Histograms → distribution of a feature
  * Box plots → outliers detection
* **Bivariate**:

  * Scatter plots → relationship between 2 features
  * Heatmaps → correlation matrix
* **Multivariate**:

  * Pair plots (Seaborn)

📌 Example:

```python
import seaborn as sns
sns.heatmap(df.corr(), annot=True)
```

---

## **5. Splitting Data for ML**

* **Why?** To avoid overfitting and fairly evaluate models.
* **Methods**:

  * Train-Test Split (e.g., 80%-20%)
  * Validation Set (train-validation-test)
  * Cross-Validation (K-fold, stratified sampling)

📌 Example:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## **6. Common Challenges**

* **Missing values**: NaN values in data (need imputation or removal)
* **Inconsistent formats**: “Yes/No” vs “1/0”
* **Mixed data types**: Strings inside numeric columns
* **Skewed distributions**: Imbalance in class labels (e.g., fraud detection datasets)

---

## **Key Takeaways**

1. Always **load & explore** data before modeling.
2. Use **summary statistics + visualization** to detect issues.
3. Proper **data splitting** is critical for model fairness.
4. Early data handling prevents problems in preprocessing & ML later.
