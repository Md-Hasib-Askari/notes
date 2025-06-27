# Data Cleaning and Preprocessing

## Overview
Data cleaning and preprocessing are crucial steps in the machine learning pipeline, often consuming 80% of a data scientist's time. Clean, well-prepared data leads to better model performance and more reliable insights.

## Common Data Quality Issues

### 1. Missing Values
- **Complete case deletion**: Remove rows with any missing values
- **Pairwise deletion**: Use available data for each analysis
- **Imputation**: Fill missing values with statistical measures

```python
import pandas as pd
import numpy as np

# Check for missing values
df.isnull().sum()
df.info()

# Handle missing values
df.dropna()  # Remove rows with missing values
df.fillna(df.mean())  # Fill with mean
df.fillna(method='forward')  # Forward fill
```

### 2. Outliers
- **Statistical outliers**: Beyond 3 standard deviations
- **IQR method**: Outside Q1-1.5*IQR or Q3+1.5*IQR
- **Domain-specific outliers**: Business logic violations

```python
# Detect outliers using IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))

# Remove outliers
df_clean = df[~outliers.any(axis=1)]
```

### 3. Inconsistent Formats
- Date formats (MM/DD/YYYY vs DD-MM-YYYY)
- Text case variations (UPPER, lower, Mixed)
- Units of measurement differences

### 4. Duplicate Records
- Exact duplicates
- Near duplicates (fuzzy matching needed)
- Partial duplicates (same key, different values)

```python
# Handle duplicates
df.duplicated().sum()  # Count duplicates
df.drop_duplicates()   # Remove exact duplicates
df.drop_duplicates(subset=['column1', 'column2'])  # Based on specific columns
```

## Data Preprocessing Techniques

### 1. Feature Scaling
**Standardization (Z-score normalization)**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
```

**Min-Max Normalization**
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df)
```

### 2. Encoding Categorical Variables

**One-Hot Encoding**
```python
pd.get_dummies(df, columns=['category_column'])
# Or using sklearn
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
encoded = encoder.fit_transform(df[['category_column']])
```

**Label Encoding**
```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['encoded_column'] = encoder.fit_transform(df['category_column'])
```

### 3. Feature Engineering
- Create new features from existing ones
- Polynomial features
- Interaction terms
- Date/time feature extraction

```python
# Date feature engineering
df['year'] = pd.to_datetime(df['date']).dt.year
df['month'] = pd.to_datetime(df['date']).dt.month
df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek

# Mathematical transformations
df['log_feature'] = np.log(df['feature'] + 1)
df['sqrt_feature'] = np.sqrt(df['feature'])
```

## Data Validation

### 1. Data Type Validation
```python
# Check data types
df.dtypes

# Convert data types
df['column'] = pd.to_numeric(df['column'], errors='coerce')
df['date_column'] = pd.to_datetime(df['date_column'])
```

### 2. Range Validation
```python
# Check for values outside expected ranges
assert df['age'].between(0, 120).all(), "Invalid age values found"
assert df['percentage'].between(0, 100).all(), "Invalid percentage values"
```

### 3. Consistency Checks
```python
# Check for logical inconsistencies
inconsistent = df[df['start_date'] > df['end_date']]
if not inconsistent.empty:
    print(f"Found {len(inconsistent)} inconsistent date records")
```

## Complete Preprocessing Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    """
    Complete data preprocessing pipeline
    """
    # 1. Handle missing values
    # Numerical columns - impute with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    imputer_num = SimpleImputer(strategy='median')
    df[num_cols] = imputer_num.fit_transform(df[num_cols])
    
    # Categorical columns - impute with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
    
    # 2. Handle outliers (using IQR method)
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)
    
    # 3. Remove duplicates
    df = df.drop_duplicates()
    
    # 4. Encode categorical variables
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # 5. Scale numerical features
    scaler = StandardScaler()
    df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])
    
    return df_encoded

# Usage
cleaned_df = preprocess_data(raw_df)
```

## Best Practices

### 1. Understand Your Data First
- Perform EDA before cleaning
- Understand business context
- Document data quality issues

### 2. Keep Track of Changes
- Log all preprocessing steps
- Save original data
- Document rationale for decisions

### 3. Validate Results
- Check data after each step
- Verify statistical properties
- Test with domain experts

### 4. Consider Impact on ML Models
- Some algorithms handle missing values
- Tree-based models robust to outliers
- Neural networks need scaled features

## Tools and Libraries

### Python Libraries
- **Pandas**: Data manipulation and cleaning
- **NumPy**: Numerical operations
- **Scikit-learn**: Preprocessing utilities
- **Missingno**: Missing data visualization

### Specialized Tools
- **OpenRefine**: Interactive data cleaning
- **Trifacta**: Data wrangling platform
- **Pandas Profiling**: Automated EDA reports

## Learning Objectives
- [ ] Identify common data quality issues
- [ ] Apply appropriate cleaning methods
- [ ] Implement preprocessing pipelines
- [ ] Handle missing values effectively
- [ ] Detect and treat outliers
- [ ] Encode categorical variables
- [ ] Scale features appropriately
- [ ] Validate data quality
- [ ] Document preprocessing steps

## Quick Reference

**Missing Values**: `df.isnull().sum()`, `df.fillna()`, `df.dropna()`
**Outliers**: IQR method, Z-score, domain knowledge
**Scaling**: StandardScaler, MinMaxScaler, RobustScaler
**Encoding**: `pd.get_dummies()`, LabelEncoder, OneHotEncoder
**Validation**: Data type checks, range validation, consistency checks

## Next Steps
- Practice with real datasets
- Learn advanced imputation techniques
- Explore feature selection methods
- Study domain-specific preprocessing