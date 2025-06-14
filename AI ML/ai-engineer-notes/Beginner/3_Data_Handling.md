
## 🧩 3. Data Handling – Notes

### 📌 Overview:

Data handling refers to the process of loading, cleaning, transforming, and preparing data before feeding it into machine learning models. High-quality data is often more important than complex models.

---

### 🧹 3.1 Data Preprocessing & Cleaning

#### ✅ Common Issues:

* Missing values
* Duplicate entries
* Inconsistent data formats
* Outliers

#### ✅ Key Techniques:

* **Handling Missing Data**:

  * Drop: `df.dropna()`
  * Impute: `df.fillna(method='ffill')` or `SimpleImputer` from `sklearn`

* **Removing Duplicates**:

  ```python
  df.drop_duplicates(inplace=True)
  ```

* **Type Conversion**:

  ```python
  df['col'] = df['col'].astype(float)
  ```

* **Outlier Detection**:

  * IQR method
  * Z-score
  * Boxplots

---

### 🧰 3.2 Feature Engineering

#### ✅ Techniques:

* **Scaling/Normalization**:

  * `StandardScaler`, `MinMaxScaler` (scikit-learn)

* **Encoding Categorical Data**:

  * One-hot encoding: `pd.get_dummies()`
  * Label encoding: `LabelEncoder()`

* **Feature Extraction**:

  * From date/time: extracting `day`, `month`, `hour`
  * Text vectorization: `TF-IDF`, `Bag of Words`

* **Polynomial Features**:
  Add interaction terms to capture non-linearities.

---

### 📊 3.3 Exploratory Data Analysis (EDA)

#### ✅ Goals:

* Understand distributions
* Identify trends, correlations, anomalies
* Inform feature selection

#### ✅ Tools & Libraries:

* `pandas` for summaries
* `matplotlib` & `seaborn` for plots
* Key plots:

  * Histograms
  * Boxplots
  * Heatmaps (`sns.heatmap(df.corr())`)
  * Pairplots

---

### 📁 3.4 File Handling (CSV, JSON, etc.)

#### ✅ CSV:

```python
import pandas as pd
df = pd.read_csv('data.csv')
```

#### ✅ JSON:

```python
import json
with open('data.json') as f:
    data = json.load(f)
```

#### ✅ Excel:

```python
df = pd.read_excel('file.xlsx')
```

#### ✅ Web Scraping:

* `requests` + `BeautifulSoup`
* Use headers to avoid getting blocked

