# First Machine Learning Model

A practical guide to building your very first machine learning model from start to finish.

## Overview
Your first ML model should be simple, interpretable, and solve a clear problem. This guide walks through the complete process using a binary classification example.

## Complete ML Workflow

### 1. Problem Definition
```python
# Example: Predict if a house will sell above median price
# Type: Binary Classification
# Target: 0 (below median), 1 (above median)
# Success metric: Accuracy > 80%
```

### 2. Data Preparation
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load and explore data
df = pd.read_csv('house_prices.csv')
print(f"Dataset shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")

# Simple feature selection (numerical only)
features = ['sqft', 'bedrooms', 'bathrooms', 'age']
X = df[features]
y = (df['price'] > df['price'].median()).astype(int)  # Binary target

# Handle missing values
X = X.fillna(X.median())

print(f"Features: {features}")
print(f"Target distribution:\n{y.value_counts()}")
```

### 3. Train-Test Split
```python
# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
```

### 4. Feature Scaling
```python
# Scale features for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 5. Model Selection & Training
```python
# Start with Logistic Regression (simple and interpretable)
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

print("Model trained successfully!")
```

### 6. Model Evaluation
```python
# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'coefficient': model.coef_[0],
    'abs_coefficient': np.abs(model.coef_[0])
}).sort_values('abs_coefficient', ascending=False)

print("\nFeature Importance:")
print(feature_importance)
```

## Model Interpretation
```python
# Understanding predictions
sample_idx = 0
sample_features = X_test.iloc[sample_idx]
prediction = model.predict(X_test_scaled[sample_idx:sample_idx+1])[0]
probability = model.predict_proba(X_test_scaled[sample_idx:sample_idx+1])[0, 1]

print(f"Sample features: {sample_features.values}")
print(f"Prediction: {'Above median' if prediction == 1 else 'Below median'}")
print(f"Probability: {probability:.3f}")
```

## Common First Model Mistakes
1. **Not splitting data properly** → Data leakage
2. **Skipping data exploration** → Poor feature selection
3. **Using complex models first** → Hard to debug
4. **Ignoring class imbalance** → Misleading accuracy
5. **Not validating assumptions** → Overfitting

## Simple Validation
```python
from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
```

## Improvement Strategies
1. **More features**: Add relevant variables
2. **Feature engineering**: Create new features
3. **Different algorithms**: Try Random Forest, SVM
4. **Hyperparameter tuning**: Optimize model parameters
5. **Data quality**: Better cleaning and preprocessing

## Complete Example Script
```python
# first_ml_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def build_first_model(data_path):
    """Build and evaluate your first ML model."""
    
    # 1. Load and explore data
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape}")
    
    # 2. Prepare features and target
    features = ['sqft', 'bedrooms', 'bathrooms', 'age']
    X = df[features].fillna(df[features].median())
    y = (df['price'] > df['price'].median()).astype(int)
    
    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # 6. Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 7. Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return model, scaler

# Usage
# model, scaler = build_first_model('house_prices.csv')
```

## Key Takeaways
- **Start simple**: Linear models are interpretable and fast
- **Validate properly**: Use train/test split and cross-validation
- **Understand your data**: EDA before modeling
- **Check assumptions**: Ensure your approach fits the problem
- **Iterate gradually**: Improve step by step

## Next Steps
1. Try different algorithms (Random Forest, SVM)
2. Implement feature engineering
3. Learn hyperparameter tuning
4. Practice with different datasets
5. Study model evaluation metrics

## Learning Objectives
- [x] Understand the complete ML workflow
- [x] Build and evaluate a simple model
- [x] Interpret model results
- [x] Validate model performance
- [x] Identify common pitfalls