# 📘 Intermediate → Exploratory Data Analysis (EDA)

**EDA** is the process of **exploring, visualizing, and summarizing data** to discover patterns, detect anomalies, test assumptions, and guide feature engineering. It acts as a **bridge between raw data and modeling**.

---

## **1. Goals of EDA**

* Understand **data structure**: number of features, data types, distributions.
* Detect **data quality issues**: missing values, outliers, duplicates.
* Discover **relationships** among variables.
* Generate **hypotheses** for feature engineering & model selection.

---

## **2. Steps in EDA**

### **(a) Data Overview**

* Dimensions: `df.shape`
* Column info: `df.info()`
* Descriptive statistics: `df.describe()`
* Missing values: `df.isnull().sum()`

📌 Example:

```python
import pandas as pd
df.describe(include='all')
```

---

### **(b) Univariate Analysis** (one feature at a time)

* **Numerical Features**:

  * Histograms → distribution & skewness.
  * Boxplots → detect outliers.
* **Categorical Features**:

  * Bar charts → frequency counts.
  * Pie charts → category proportions.

📌 Example:

```python
import seaborn as sns
sns.histplot(df['age'], bins=30, kde=True)
```

---

### **(c) Bivariate Analysis** (two features at a time)

* **Numerical vs Numerical**:

  * Scatter plots (detect linear/nonlinear relationships).
  * Correlation heatmap.
* **Categorical vs Numerical**:

  * Boxplots/violin plots (distribution across groups).
* **Categorical vs Categorical**:

  * Crosstab / stacked bar charts.

📌 Example:

```python
sns.scatterplot(x='age', y='salary', data=df)
```

---

### **(d) Multivariate Analysis** (three or more features)

* Pair plots (Seaborn → `sns.pairplot()`)
* Dimensionality reduction (PCA, t-SNE, UMAP) for visualization.
* Cluster heatmaps to detect groupings.

---

## **3. Handling Outliers in EDA**

* Use boxplots, scatterplots to detect anomalies.
* Decide whether to **remove, transform, or keep** depending on domain knowledge.

---

## **4. Correlation & Feature Relationships**

* Compute correlations:

  * Pearson (linear relationships).
  * Spearman (rank-based).
* Watch out for **multicollinearity** (highly correlated features) → can affect regression models.

📌 Example:

```python
import matplotlib.pyplot as plt
import numpy as np

corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
```

---

## **5. Advanced EDA Techniques**

* **Missing data visualization**: `missingno` library.
* **Interactive plots**: Plotly, Bokeh.
* **Feature importance estimation**: RandomForest, XGBoost feature importances (as part of EDA).
* **Target variable analysis**: Class imbalance, regression target distribution.

---

## ✅ Key Takeaways

1. **EDA is mandatory** before modeling — it reveals data quality, structure, and hidden patterns.
2. Use **univariate, bivariate, and multivariate** analysis for deep insights.
3. Visualization (histograms, scatterplots, heatmaps) is key to understanding relationships.
4. EDA guides **feature engineering** and **model selection**.

---