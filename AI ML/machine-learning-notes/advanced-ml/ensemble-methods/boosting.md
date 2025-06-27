# Boosting (XGBoost, LightGBM)

## Overview
Boosting is an ensemble method that combines weak learners sequentially, where each learner corrects the mistakes of previous ones. XGBoost and LightGBM are the most popular gradient boosting implementations.

## Core Concepts

### Gradient Boosting Fundamentals
- **Sequential Learning**: Models are added one at a time
- **Error Correction**: Each new model focuses on previous model's errors
- **Gradient Descent**: Optimizes loss function by fitting to negative gradients
- **Weak Learners**: Usually decision trees with limited depth

### Mathematical Foundation
```python
# Gradient Boosting Algorithm
# F_0(x) = initial prediction (usually mean for regression)
# For m = 1 to M:
#   1. Compute pseudo-residuals: r_i = -∂L(y_i, F_{m-1}(x_i))/∂F_{m-1}(x_i)
#   2. Fit weak learner h_m(x) to residuals
#   3. Find optimal step size: γ_m = argmin Σ L(y_i, F_{m-1}(x_i) + γ*h_m(x_i))
#   4. Update: F_m(x) = F_{m-1}(x) + γ_m * h_m(x)
```

## XGBoost (Extreme Gradient Boosting)

### Key Features
- **Regularization**: L1 and L2 regularization to prevent overfitting
- **Handling Missing Values**: Built-in missing value handling
- **Parallel Processing**: Tree construction parallelization
- **Cross-Validation**: Built-in cross-validation support

### Basic Implementation
```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd
import numpy as np

# Classification
def xgboost_classification(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Parameters
    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'logloss',
        'seed': 42
    }
    
    # Train with early stopping
    model = xgb.train(
        params, 
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    # Predictions
    y_pred = model.predict(dtest)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    return model, accuracy_score(y_test, y_pred_binary)

# Regression
def xgboost_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # Fit with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    y_pred = model.predict(X_test)
    return model, mean_squared_error(y_test, y_pred)
```

### Advanced XGBoost Features
```python
# Custom Objective Function
def custom_objective(y_pred, y_true):
    """Custom objective function example"""
    grad = y_pred - y_true.get_label()
    hess = np.ones(len(y_true.get_label()))
    return grad, hess

# Custom Evaluation Metric
def custom_eval(y_pred, y_true):
    """Custom evaluation metric"""
    labels = y_true.get_label()
    error = np.mean(np.abs(y_pred - labels))
    return 'custom_mae', error

# Multi-class Classification
def xgboost_multiclass(X, y, num_classes):
    params = {
        'objective': 'multi:softprob',
        'num_class': num_classes,
        'max_depth': 6,
        'learning_rate': 0.1,
        'eval_metric': 'mlogloss'
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(params, dtrain, num_boost_round=100)
    return model
```

## LightGBM (Light Gradient Boosting Machine)

### Key Features
- **Leaf-wise Growth**: More efficient than level-wise growth
- **Categorical Feature Support**: Native categorical feature handling
- **Memory Efficiency**: Lower memory usage than XGBoost
- **Speed**: Faster training than XGBoost

### Basic Implementation
```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Classification
def lightgbm_classification(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Parameters
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }
    
    # Train with validation
    model = lgb.train(
        params,
        train_data,
        valid_sets=[test_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    return model, accuracy_score(y_test, y_pred_binary)

# Regression
def lightgbm_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        num_leaves=31,
        learning_rate=0.1,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50)]
    )
    
    y_pred = model.predict(X_test)
    return model, mean_squared_error(y_test, y_pred)
```

### Categorical Features in LightGBM
```python
def lightgbm_with_categorical(X, y, categorical_features):
    """Handle categorical features natively in LightGBM"""
    
    # Mark categorical features
    train_data = lgb.Dataset(
        X_train, 
        label=y_train,
        categorical_feature=categorical_features
    )
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'cat_smooth': 10,  # Categorical smoothing
        'cat_l2': 10       # L2 regularization for categorical
    }
    
    model = lgb.train(params, train_data, num_boost_round=100)
    return model
```

## Hyperparameter Tuning

### Grid Search with Cross-Validation
```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

def tune_xgboost(X, y):
    """Hyperparameter tuning for XGBoost"""
    
    # Define parameter grid
    param_grid = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Create model
    xgb_model = xgb.XGBClassifier(random_state=42)
    
    # Grid search
    grid_search = GridSearchCV(
        xgb_model,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    return grid_search.best_params_, grid_search.best_score_

def tune_lightgbm(X, y):
    """Hyperparameter tuning for LightGBM"""
    
    param_grid = {
        'num_leaves': [15, 31, 63],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'feature_fraction': [0.8, 0.9, 1.0],
        'bagging_fraction': [0.8, 0.9, 1.0]
    }
    
    lgb_model = lgb.LGBMClassifier(random_state=42)
    
    grid_search = GridSearchCV(
        lgb_model,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    
    return grid_search.best_params_, grid_search.best_score_
```

### Bayesian Optimization
```python
from skopt import BayesSearchCV
from skopt.space import Real, Integer

def bayesian_tune_xgboost(X, y):
    """Bayesian optimization for XGBoost"""
    
    # Define search space
    search_space = {
        'max_depth': Integer(3, 8),
        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'n_estimators': Integer(100, 1000),
        'subsample': Real(0.7, 1.0),
        'colsample_bytree': Real(0.7, 1.0),
        'reg_alpha': Real(0, 10),
        'reg_lambda': Real(0, 10)
    }
    
    xgb_model = xgb.XGBClassifier(random_state=42)
    
    bayes_search = BayesSearchCV(
        xgb_model,
        search_space,
        n_iter=50,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )
    
    bayes_search.fit(X, y)
    
    return bayes_search.best_params_, bayes_search.best_score_
```

## Feature Importance and Interpretation

### Feature Importance
```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(model, feature_names, model_type='xgboost'):
    """Plot feature importance"""
    
    if model_type == 'xgboost':
        importance = model.get_score(importance_type='weight')
        importance_df = pd.DataFrame.from_dict(importance, orient='index', columns=['importance'])
    elif model_type == 'lightgbm':
        importance = model.feature_importance(importance_type='split')
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
    
    # Sort and plot
    importance_df = importance_df.sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(importance_df)), importance_df['importance'])
    plt.yticks(range(len(importance_df)), importance_df.index)
    plt.xlabel('Feature Importance')
    plt.title(f'{model_type.upper()} Feature Importance')
    plt.tight_layout()
    plt.show()

# SHAP values for model interpretation
def explain_with_shap(model, X_test, model_type='xgboost'):
    """Use SHAP for model interpretation"""
    import shap
    
    if model_type == 'xgboost':
        explainer = shap.TreeExplainer(model)
    elif model_type == 'lightgbm':
        explainer = shap.TreeExplainer(model)
    
    shap_values = explainer.shap_values(X_test)
    
    # Summary plot
    shap.summary_plot(shap_values, X_test)
    
    # Waterfall plot for single prediction
    shap.waterfall_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
    
    return shap_values
```

## Model Comparison and Evaluation

### Comprehensive Comparison
```python
def compare_boosting_models(X, y, task_type='classification'):
    """Compare XGBoost and LightGBM performance"""
    
    models = {}
    results = {}
    
    # XGBoost
    if task_type == 'classification':
        xgb_model = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        lgb_model = lgb.LGBMClassifier(
            n_estimators=1000,
            num_leaves=31,
            learning_rate=0.1,
            random_state=42
        )
        scoring = 'accuracy'
    else:
        xgb_model = xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        lgb_model = lgb.LGBMRegressor(
            n_estimators=1000,
            num_leaves=31,
            learning_rate=0.1,
            random_state=42
        )
        scoring = 'neg_mean_squared_error'
    
    # Cross-validation comparison
    from sklearn.model_selection import cross_val_score
    import time
    
    models_dict = {'XGBoost': xgb_model, 'LightGBM': lgb_model}
    
    for name, model in models_dict.items():
        start_time = time.time()
        cv_scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
        training_time = time.time() - start_time
        
        results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_time': training_time
        }
        
        models[name] = model
    
    return models, results

# Performance metrics
def detailed_evaluation(y_true, y_pred, task_type='classification'):
    """Detailed model evaluation"""
    
    if task_type == 'classification':
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        
        print("Classification Report:")
        print(classification_report(y_true, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        
        if len(np.unique(y_true)) == 2:  # Binary classification
            auc = roc_auc_score(y_true, y_pred)
            print(f"\nAUC Score: {auc:.4f}")
    
    else:  # Regression
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
```

## Advanced Techniques

### Stacking with Boosting Models
```python
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

def create_boosting_stack(X, y, task_type='classification'):
    """Create stacked ensemble with boosting models"""
    
    # Base models
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
    
    base_models = [
        ('xgb', xgb_model),
        ('lgb', lgb_model)
    ]
    
    if task_type == 'classification':
        meta_model = LogisticRegression()
        stacking_model = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5
        )
    else:
        meta_model = LinearRegression()
        stacking_model = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5
        )
    
    stacking_model.fit(X, y)
    return stacking_model
```

### Multi-target Learning
```python
def multi_target_boosting(X, y_multi):
    """Boosting for multi-target problems"""
    
    from sklearn.multioutput import MultiOutputRegressor
    
    # XGBoost for multi-target
    xgb_multi = MultiOutputRegressor(
        xgb.XGBRegressor(n_estimators=100, random_state=42)
    )
    
    # LightGBM for multi-target
    lgb_multi = MultiOutputRegressor(
        lgb.LGBMRegressor(n_estimators=100, random_state=42)
    )
    
    xgb_multi.fit(X, y_multi)
    lgb_multi.fit(X, y_multi)
    
    return xgb_multi, lgb_multi
```

## Best Practices

### Data Preprocessing for Boosting
```python
def preprocess_for_boosting(X, y, handle_missing=True, encode_categorical=True):
    """Preprocessing specifically for boosting algorithms"""
    
    X_processed = X.copy()
    
    # Handle missing values (boosting algorithms can handle them, but preprocessing might help)
    if handle_missing:
        from sklearn.impute import SimpleImputer
        
        # Numerical features
        num_features = X_processed.select_dtypes(include=[np.number]).columns
        if len(num_features) > 0:
            num_imputer = SimpleImputer(strategy='median')
            X_processed[num_features] = num_imputer.fit_transform(X_processed[num_features])
        
        # Categorical features
        cat_features = X_processed.select_dtypes(include=['object']).columns
        if len(cat_features) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X_processed[cat_features] = cat_imputer.fit_transform(X_processed[cat_features])
    
    # Encode categorical variables (for XGBoost)
    if encode_categorical:
        from sklearn.preprocessing import LabelEncoder
        
        cat_features = X_processed.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for feature in cat_features:
            le = LabelEncoder()
            X_processed[feature] = le.fit_transform(X_processed[feature].astype(str))
            label_encoders[feature] = le
    
    return X_processed, label_encoders

# Feature engineering for boosting
def create_boosting_features(X):
    """Create features that work well with boosting"""
    
    X_enhanced = X.copy()
    
    # Polynomial features for numerical columns
    num_cols = X.select_dtypes(include=[np.number]).columns
    
    # Interaction features (limited to avoid explosion)
    for i, col1 in enumerate(num_cols[:5]):  # Limit to first 5 numerical columns
        for col2 in num_cols[i+1:6]:
            X_enhanced[f'{col1}_{col2}_interaction'] = X[col1] * X[col2]
    
    # Binning continuous variables
    for col in num_cols:
        X_enhanced[f'{col}_binned'] = pd.cut(X[col], bins=5, labels=False)
    
    return X_enhanced
```

### Production Considerations
```python
def production_boosting_pipeline(X_train, y_train, X_test, model_type='lightgbm'):
    """Production-ready boosting pipeline"""
    
    # 1. Data validation
    assert X_train.shape[1] == X_test.shape[1], "Feature mismatch between train and test"
    
    # 2. Preprocessing
    X_train_processed, encoders = preprocess_for_boosting(X_train, y_train)
    X_test_processed = X_test.copy()
    
    # Apply same preprocessing to test set
    for feature, encoder in encoders.items():
        if feature in X_test_processed.columns:
            X_test_processed[feature] = encoder.transform(X_test_processed[feature].astype(str))
    
    # 3. Model training with validation
    if model_type == 'lightgbm':
        model = lgb.LGBMClassifier(
            n_estimators=1000,
            num_leaves=31,
            learning_rate=0.1,
            random_state=42
        )
    else:
        model = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    
    # 4. Fit with early stopping
    model.fit(
        X_train_processed, y_train,
        eval_set=[(X_train_processed, y_train)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    # 5. Predictions
    predictions = model.predict(X_test_processed)
    probabilities = model.predict_proba(X_test_processed)
    
    return {
        'model': model,
        'predictions': predictions,
        'probabilities': probabilities,
        'encoders': encoders,
        'feature_names': X_train_processed.columns.tolist()
    }
```

## XGBoost vs LightGBM Comparison

| Aspect | XGBoost | LightGBM |
|--------|---------|----------|
| **Speed** | Moderate | Fast |
| **Memory Usage** | Higher | Lower |
| **Tree Growth** | Level-wise | Leaf-wise |
| **Categorical Features** | Requires encoding | Native support |
| **Overfitting** | More robust | Can overfit easier |
| **Documentation** | Extensive | Good |
| **Community** | Large | Growing |
| **GPU Support** | Yes | Yes (better) |

## Common Pitfalls and Solutions

### 1. Overfitting
```python
# Solutions for overfitting
overfitting_solutions = {
    'early_stopping': {'early_stopping_rounds': 50},
    'regularization': {'reg_alpha': 1, 'reg_lambda': 1},
    'subsampling': {'subsample': 0.8, 'colsample_bytree': 0.8},
    'learning_rate': {'learning_rate': 0.05, 'n_estimators': 2000},
    'max_depth': {'max_depth': 4}  # Reduce tree depth
}
```

### 2. Memory Issues
```python
# Memory optimization techniques
memory_optimization = {
    'reduce_precision': 'Use float32 instead of float64',
    'feature_selection': 'Remove irrelevant features',
    'batch_processing': 'Process data in chunks',
    'garbage_collection': 'Explicit memory cleanup'
}

# Example of memory-efficient training
def memory_efficient_training(X, y, chunk_size=10000):
    """Train model with limited memory"""
    
    model = lgb.LGBMClassifier(n_estimators=100)
    
    # Train in chunks
    for i in range(0, len(X), chunk_size):
        X_chunk = X[i:i+chunk_size]
        y_chunk = y[i:i+chunk_size]
        
        if i == 0:
            model.fit(X_chunk, y_chunk)
        else:
            # Incremental learning (if supported)
            model.fit(X_chunk, y_chunk, init_model=model)
    
    return model
```

## Key Takeaways

1. **XGBoost**: More stable, better documentation, handles overfitting well
2. **LightGBM**: Faster, memory-efficient, better for large datasets
3. **Hyperparameter tuning** is crucial for optimal performance
4. **Early stopping** prevents overfitting and saves training time
5. **Feature engineering** can significantly improve boosting performance
6. **Cross-validation** is essential for reliable model evaluation
7. **SHAP values** provide excellent model interpretability
8. **Categorical features** are handled better by LightGBM natively

## Resources
- **XGBoost Documentation**: Official XGBoost guide and tutorials
- **LightGBM Documentation**: Microsoft's LightGBM documentation
- **Kaggle Competitions**: Many winning solutions use boosting
- **Papers**: Original XGBoost and LightGBM research papers
