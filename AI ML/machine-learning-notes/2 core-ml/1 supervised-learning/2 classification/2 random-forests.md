# Random Forests

## Overview
Random Forest is an ensemble learning method that combines multiple decision trees to create a more robust and accurate model. It uses bagging (bootstrap aggregating) and random feature selection to reduce overfitting and improve generalization.

## Key Concepts

### Ensemble Method
- **Bootstrap Aggregating (Bagging)**: Creates multiple subsets of training data through random sampling with replacement
- **Random Feature Selection**: Each tree uses a random subset of features at each split
- **Majority Voting**: Final prediction is made by aggregating predictions from all trees
- **Forest Structure**: Typically uses 100-1000 trees for optimal performance

### How Random Forests Work
1. **Bootstrap Sampling**: Create n bootstrap samples from training data
2. **Tree Building**: Train decision tree on each sample using random feature subset
3. **Feature Randomness**: At each node, consider only √p features (p = total features)
4. **Prediction**: Aggregate predictions (majority vote for classification, average for regression)
5. **Out-of-Bag (OOB) Error**: Use samples not in bootstrap for validation

## Advantages
- **Reduced Overfitting**: Multiple trees with randomness prevent memorizing training data
- **Feature Importance**: Calculates importance scores for feature selection
- **Handles Missing Values**: Built-in methods for dealing with incomplete data
- **No Feature Scaling Required**: Tree-based methods are scale-invariant
- **Out-of-Bag Error**: Built-in validation without separate test set
- **Parallelizable**: Trees can be trained independently

## Disadvantages
- **Less Interpretable**: Harder to understand than single decision tree
- **Memory Intensive**: Stores multiple trees
- **Can Overfit with Noisy Data**: Still susceptible with very noisy datasets
- **Biased Towards Categorical Features**: May favor features with more levels

## Implementation

### Basic Random Forest
```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd

# Classification Example
X_class, y_class = make_classification(n_samples=1000, n_features=10, 
                                      n_informative=5, n_redundant=3, 
                                      random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, 
                                                    test_size=0.2, random_state=42)

# Create and train Random Forest classifier
rf_classifier = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Maximum depth of trees
    min_samples_split=5,   # Minimum samples to split node
    min_samples_leaf=2,    # Minimum samples in leaf
    max_features='sqrt',   # Number of features per split
    random_state=42
)

rf_classifier.fit(X_train, y_train)

# Predictions
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

### Feature Importance Analysis
```python
# Feature importance
feature_importance = rf_classifier.feature_importances_
feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]

# Create importance DataFrame
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("Feature Importance:")
print(importance_df)

# Plot feature importance
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('Importance')
plt.title('Random Forest Feature Importance')
plt.gca().invert_yaxis()
plt.show()
```

### Out-of-Bag (OOB) Error
```python
# Enable OOB score calculation
rf_oob = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,  # Calculate OOB error
    random_state=42
)

rf_oob.fit(X_train, y_train)
print(f"OOB Score: {rf_oob.oob_score_:.3f}")
```

### Regression Example
```python
# Regression Example
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, 
                              noise=0.1, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)

# Random Forest Regressor
rf_regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

rf_regressor.fit(X_train_reg, y_train_reg)

# Predictions and evaluation
y_pred_reg = rf_regressor.predict(X_test_reg)
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"MSE: {mse:.3f}")
print(f"R² Score: {r2:.3f}")
```

## Parameter Tuning

### Grid Search for Hyperparameters
```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Grid search
rf_grid = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_grid, param_grid, cv=5, 
                          scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Use best model
best_rf = grid_search.best_estimator_
```

### Random Search (More Efficient)
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Random parameter distributions
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [None] + list(range(5, 21)),
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2', None]
}

# Random search
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50,  # Number of parameter settings sampled
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
print("Best parameters:", random_search.best_params_)
```

## Advanced Techniques

### Handling Imbalanced Data
```python
# Class weight balancing
rf_balanced = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # Automatically adjust weights
    random_state=42
)

# Manual class weights
class_weights = {0: 1, 1: 3}  # Give more weight to minority class
rf_weighted = RandomForestClassifier(
    n_estimators=100,
    class_weight=class_weights,
    random_state=42
)
```

### Feature Selection with Random Forest
```python
from sklearn.feature_selection import SelectFromModel

# Feature selection based on importance
selector = SelectFromModel(rf_classifier, threshold='median')
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

print(f"Original features: {X_train.shape[1]}")
print(f"Selected features: {X_train_selected.shape[1]}")

# Train on selected features
rf_selected = RandomForestClassifier(random_state=42)
rf_selected.fit(X_train_selected, y_train)
```

## Key Parameters

### Essential Parameters
- **n_estimators**: Number of trees (default: 100)
  - More trees = better performance but slower training
  - Typical range: 100-1000

- **max_depth**: Maximum depth of trees (default: None)
  - Controls overfitting
  - None = trees expanded until leaves are pure

- **min_samples_split**: Minimum samples to split internal node (default: 2)
  - Higher values prevent overfitting
  - Typical range: 2-10

- **min_samples_leaf**: Minimum samples in leaf node (default: 1)
  - Smooths model, prevents overfitting
  - Typical range: 1-5

- **max_features**: Features considered for best split (default: 'sqrt')
  - 'sqrt': √(total features)
  - 'log2': log₂(total features)
  - None: All features

## Best Practices

### Model Training
1. **Start with Defaults**: Begin with default parameters
2. **Increase Trees Gradually**: Start with 100, increase if performance improves
3. **Control Depth**: Use max_depth to prevent overfitting
4. **Use OOB Score**: Quick validation without separate test set
5. **Monitor Training Time**: Balance performance vs. computation time

### Feature Engineering
1. **Feature Importance**: Use for feature selection
2. **Handle Categorical Variables**: Use proper encoding
3. **Missing Values**: Random Forests handle missing values well
4. **Feature Scaling**: Not required for tree-based methods

### Validation Strategy
```python
# Cross-validation with Random Forest
from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(rf_classifier, X_train, y_train, cv=5)
print(f"CV Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
```

## Common Issues and Solutions

### Overfitting
- **Solution**: Reduce max_depth, increase min_samples_split/leaf
- **Use OOB Error**: Monitor for overfitting during training

### Poor Performance
- **Insufficient Trees**: Increase n_estimators
- **Poor Feature Quality**: Feature engineering and selection
- **Imbalanced Data**: Use class_weight parameter

### Slow Training
- **Reduce Trees**: Lower n_estimators
- **Limit Depth**: Set max_depth
- **Parallel Processing**: Use n_jobs=-1

## Comparison with Other Algorithms

| Aspect | Random Forest | Decision Tree | Gradient Boosting | SVM |
|--------|---------------|---------------|-------------------|-----|
| Overfitting | Low | High | Medium | Low-Medium |
| Training Speed | Medium | Fast | Slow | Medium |
| Interpretability | Low | High | Low | Low |
| Feature Scaling | Not Required | Not Required | Not Required | Required |
| Missing Values | Handles Well | Handles Well | Handles Well | Requires Imputation |

## Learning Objectives
- [ ] Understand ensemble learning principles
- [ ] Implement Random Forest for classification and regression
- [ ] Analyze feature importance and selection
- [ ] Tune hyperparameters effectively
- [ ] Handle imbalanced datasets
- [ ] Use OOB error for validation
- [ ] Compare with other ensemble methods
- [ ] Apply to real-world problems

## Practice Projects
1. **Customer Churn Prediction**: Binary classification with feature importance
2. **House Price Prediction**: Regression with hyperparameter tuning
3. **Image Classification**: High-dimensional data handling
4. **Feature Selection Pipeline**: Automated feature selection workflow

## Next Steps
- Study Gradient Boosting (XGBoost, LightGBM)
- Learn about Stacking and Voting ensembles
- Explore Advanced ensemble techniques
- Practice with competition datasets (Kaggle)