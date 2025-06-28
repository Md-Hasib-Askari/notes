# XGBoost and LightGBM

## Overview
XGBoost and LightGBM are popular gradient boosting frameworks used for structured data tasks. They are known for their speed, accuracy, and scalability.

## Key Features

### XGBoost
- **Regularization**: L1 and L2 regularization to prevent overfitting
- **Handling Missing Values**: Built-in support for missing values
- **Parallel Processing**: Tree construction parallelization
- **Cross-Validation**: Built-in cross-validation support

### LightGBM
- **Leaf-wise Growth**: More efficient than level-wise growth
- **Categorical Feature Support**: Native handling of categorical features
- **Memory Efficiency**: Lower memory usage than XGBoost
- **Speed**: Faster training than XGBoost

## Installation
```bash
# Install XGBoost
pip install xgboost

# Install LightGBM
pip install lightgbm
```

## Basic Workflow

### XGBoost
```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Parameters
params = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'logloss'
}

# Train
model = xgb.train(params, dtrain, num_boost_round=100)

# Predict
predictions = model.predict(dtest)
```

### LightGBM
```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Parameters
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.8
}

# Train
model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100)

# Predict
predictions = model.predict(X_test)
```

## Advanced Features

### Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

# XGBoost
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300]
}
model = xgb.XGBClassifier()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

# LightGBM
param_grid = {
    'num_leaves': [15, 31, 63],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300]
}
model = lgb.LGBMClassifier()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)
```

### Feature Importance
```python
# XGBoost
importance = model.get_score(importance_type='weight')
print(importance)

# LightGBM
importance = model.feature_importance()
print(importance)
```

## Best Practices
1. Use early stopping to prevent overfitting.
2. Use cross-validation for reliable model evaluation.
3. Experiment with hyperparameter tuning for optimal performance.
4. Use feature importance to understand model behavior.
5. Use LightGBM for large datasets with categorical features.

## Common Pitfalls
- **Overfitting**: Use regularization and early stopping.
- **Memory Issues**: Use LightGBM for memory efficiency.
- **Feature Mismatch**: Ensure consistent preprocessing for training and testing.

## Applications
- **Classification**: Fraud detection, sentiment analysis
- **Regression**: House price prediction, stock market forecasting
- **Ranking**: Search engine optimization, recommendation systems

## Resources
- **XGBoost Documentation**: Official guide and tutorials
- **LightGBM Documentation**: Official guide and tutorials
- **Kaggle Competitions**: Many winning solutions use XGBoost and LightGBM
