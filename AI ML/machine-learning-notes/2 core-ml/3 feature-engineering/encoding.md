# Categorical Variable Encoding

## Overview
Categorical variable encoding transforms categorical data into numerical format that machine learning algorithms can process. The choice of encoding method significantly impacts model performance.

## Learning Objectives
- Understand different encoding techniques
- Apply appropriate encoding methods based on data characteristics
- Handle high-cardinality categorical variables
- Implement encoding in Python with practical examples

## Core Encoding Methods

### 1. Label Encoding
Assigns integer values to categories. Use for ordinal data only.

```python
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Example data
data = pd.DataFrame({
    'size': ['Small', 'Medium', 'Large', 'Small', 'Large']
})

# Label encoding
le = LabelEncoder()
data['size_encoded'] = le.fit_transform(data['size'])
print(data)
# Output: 0=Large, 1=Medium, 2=Small
```

**When to use**: Ordinal categories with natural order (Low < Medium < High)

### 2. One-Hot Encoding
Creates binary columns for each category. Standard for nominal data.

```python
# Using pandas
data_onehot = pd.get_dummies(data['size'], prefix='size')
print(data_onehot)

# Using sklearn
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False, drop='first')
encoded = ohe.fit_transform(data[['size']])
```

**When to use**: Nominal categories with no inherent order

### 3. Target Encoding (Mean Encoding)
Replaces categories with target variable statistics.

```python
import numpy as np

def target_encode(df, categorical_col, target_col):
    """Target encoding with smoothing"""
    global_mean = df[target_col].mean()
    category_stats = df.groupby(categorical_col)[target_col].agg(['mean', 'count'])
    
    # Add smoothing to prevent overfitting
    smoothing = 10
    category_stats['smoothed_mean'] = (
        (category_stats['mean'] * category_stats['count'] + global_mean * smoothing) /
        (category_stats['count'] + smoothing)
    )
    
    return df[categorical_col].map(category_stats['smoothed_mean'])

# Example
df = pd.DataFrame({
    'category': ['A', 'B', 'A', 'C', 'B', 'A'],
    'target': [1, 0, 1, 1, 0, 0]
})
df['category_encoded'] = target_encode(df, 'category', 'target')
```

**When to use**: High-cardinality categories with strong target correlation

### 4. Binary Encoding
Combines benefits of label and one-hot encoding for high cardinality.

```python
import category_encoders as ce

# Binary encoding
be = ce.BinaryEncoder(cols=['category'])
binary_encoded = be.fit_transform(df)
print(binary_encoded)
```

**When to use**: High-cardinality nominal variables (>50 categories)

## Advanced Techniques

### Count/Frequency Encoding
```python
def count_encode(series):
    """Encode by category frequency"""
    counts = series.value_counts()
    return series.map(counts)

data['size_count'] = count_encode(data['size'])
```

### Hash Encoding
```python
from sklearn.feature_extraction import FeatureHasher

def hash_encode(series, n_features=8):
    """Hash encoding for very high cardinality"""
    hasher = FeatureHasher(n_features=n_features, input_type='string')
    return hasher.transform(series.astype(str)).toarray()
```

### Ordinal Encoding with Custom Order
```python
from sklearn.preprocessing import OrdinalEncoder

# Define custom order
size_order = [['Small', 'Medium', 'Large']]
oe = OrdinalEncoder(categories=size_order)
data['size_ordinal'] = oe.fit_transform(data[['size']])
```

## Handling Special Cases

### Missing Values
```python
# Handle missing values before encoding
data['category'] = data['category'].fillna('Unknown')

# Or use category_encoders with handle_unknown
encoder = ce.OneHotEncoder(handle_unknown='return_nan')
```

### High Cardinality
```python
def reduce_cardinality(series, min_freq=5):
    """Group rare categories"""
    counts = series.value_counts()
    rare_categories = counts[counts < min_freq].index
    return series.replace(rare_categories, 'Other')

data['category_reduced'] = reduce_cardinality(data['category'])
```

### Time-Based Features
```python
# Extract features from datetime
df['date'] = pd.to_datetime(df['date'])
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
```

## Complete Encoding Pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline

# Define transformers
categorical_features = ['color', 'size', 'brand']
ordinal_features = ['grade']
ordinal_categories = [['F', 'D', 'C', 'B', 'A']]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
        ('ord', OrdinalEncoder(categories=ordinal_categories), ordinal_features)
    ]
)

# Create pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
```

## Best Practices

### 1. Choose Appropriate Method
- **Ordinal data**: Label/Ordinal encoding
- **Nominal, low cardinality (<10)**: One-hot encoding
- **Nominal, high cardinality (>50)**: Target/Binary/Hash encoding
- **Tree-based models**: Can handle label encoding for nominal data

### 2. Prevent Data Leakage
```python
# Use cross-validation for target encoding
from sklearn.model_selection import KFold

def cv_target_encode(X, y, categorical_col, cv=5):
    """Cross-validated target encoding"""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    encoded = np.zeros(len(X))
    
    for train_idx, val_idx in kf.split(X):
        # Encode validation set using training statistics
        train_stats = X.iloc[train_idx].groupby(categorical_col)[y.iloc[train_idx]].mean()
        encoded[val_idx] = X.iloc[val_idx][categorical_col].map(train_stats)
    
    return encoded
```

### 3. Handle Unknown Categories
```python
# Save encoder for new data
import joblib

# Fit and save
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(train_data[['category']])
joblib.dump(encoder, 'category_encoder.pkl')

# Load and transform new data
encoder = joblib.load('category_encoder.pkl')
new_encoded = encoder.transform(new_data[['category']])
```

### 4. Monitor Encoding Quality
```python
def encoding_quality_check(original, encoded, target):
    """Check encoding effectiveness"""
    from scipy.stats import chi2_contingency
    
    # Chi-square test for association
    chi2, p_value, _, _ = chi2_contingency(pd.crosstab(original, target))
    
    # Mutual information
    from sklearn.feature_selection import mutual_info_classif
    mi_score = mutual_info_classif(encoded.reshape(-1, 1), target)[0]
    
    print(f"Chi-square p-value: {p_value:.4f}")
    print(f"Mutual Information: {mi_score:.4f}")
```

## Common Pitfalls

1. **Using label encoding for nominal data**: Creates false ordinal relationships
2. **One-hot encoding high cardinality**: Creates sparse, high-dimensional data
3. **Target encoding without cross-validation**: Causes overfitting
4. **Not handling unknown categories**: Breaks model in production
5. **Inconsistent encoding across train/test**: Creates data leakage

## Summary

Categorical encoding is crucial for ML success. Choose methods based on:
- **Data type**: Ordinal vs nominal
- **Cardinality**: Number of unique categories
- **Model type**: Tree-based vs linear models
- **Target relationship**: Correlation strength

Always validate encoding choices through cross-validation and monitor for overfitting, especially with target-based encodings.
