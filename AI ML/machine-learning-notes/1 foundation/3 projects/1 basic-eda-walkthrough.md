# Basic EDA Walkthrough

## Quick Reference
**Goal**: Systematically explore dataset to understand structure, patterns, and quality before modeling.

## Step-by-Step EDA Process

### 1. Data Loading & First Look
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('dataset.csv')

# Basic info
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
df.head()
df.info()
```

### 2. Data Quality Assessment
```python
# Missing values
missing_pct = (df.isnull().sum() / len(df)) * 100
print("Missing Values %:")
print(missing_pct[missing_pct > 0].sort_values(ascending=False))

# Duplicates
print(f"Duplicates: {df.duplicated().sum()}")

# Data types
print("\nData Types:")
print(df.dtypes.value_counts())
```

### 3. Statistical Summary
```python
# Numerical columns
df.describe()

# Categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\n{col} - Unique values: {df[col].nunique()}")
    print(df[col].value_counts().head())
```

### 4. Distribution Analysis
```python
# Numerical distributions
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    df[col].hist(bins=30)
    plt.title(f'{col} - Histogram')
    
    plt.subplot(1, 3, 2)
    df.boxplot(column=col)
    plt.title(f'{col} - Boxplot')
    
    plt.subplot(1, 3, 3)
    df[col].plot(kind='density')
    plt.title(f'{col} - Density')
    
    plt.tight_layout()
    plt.show()
```

### 5. Correlation Analysis
```python
# Correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# High correlations
high_corr = corr_matrix.abs() > 0.7
high_corr_pairs = [(i, j, corr_matrix.iloc[i, j]) 
                   for i in range(len(corr_matrix.columns)) 
                   for j in range(i+1, len(corr_matrix.columns))
                   if high_corr.iloc[i, j]]
print("High Correlations (>0.7):", high_corr_pairs)
```

### 6. Outlier Detection
```python
# IQR method
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

# Check outliers for numerical columns
for col in numerical_cols:
    outliers = detect_outliers_iqr(df, col)
    print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")
```

## Common EDA Patterns to Look For

### Data Quality Issues
- **Missing Values**: Random vs. systematic patterns
- **Duplicates**: Exact vs. partial duplicates
- **Inconsistent Formats**: Date formats, categorical spelling
- **Outliers**: Legitimate extreme values vs. errors

### Distribution Patterns
- **Normal Distribution**: Bell curve, suitable for parametric tests
- **Skewed Distribution**: Right/left skew, may need transformation
- **Bimodal**: Multiple peaks, possible subgroups
- **Uniform**: Equal frequency, may indicate artificial data

### Relationship Patterns
- **Linear Correlations**: Positive/negative relationships
- **Non-linear Relationships**: Curved patterns in scatter plots
- **Categorical Associations**: Chi-square relationships
- **Time Trends**: Seasonal patterns, trends

## Quick EDA Checklist
- [ ] Load data and check shape/columns
- [ ] Assess missing values and duplicates
- [ ] Generate statistical summaries
- [ ] Plot distributions for key variables
- [ ] Create correlation heatmap
- [ ] Identify and investigate outliers
- [ ] Check for class imbalance (classification)
- [ ] Examine relationships with target variable
- [ ] Document key findings and next steps

## Pro Tips
- **Start Simple**: Basic plots before complex analysis
- **Target-Focused**: Always relate findings to your ML objective
- **Document Everything**: Keep notes of insights and decisions
- **Iterate**: EDA is not linear - findings lead to new questions
- **Domain Knowledge**: Subject matter expertise enhances interpretation

## Common Tools
```python
# Quick profiling
import pandas_profiling
profile = df.profile_report()
profile.to_file("eda_report.html")

# Advanced visualization
import plotly.express as px
fig = px.scatter_matrix(df.select_dtypes(include=[np.number]))
fig.show()
```

**Remember**: EDA should tell a story about your data that informs your modeling strategy!