# Automated Feature Engineering

## Learning Objectives
- Understand automated feature engineering concepts and benefits
- Learn about feature generation and selection algorithms
- Implement automated feature engineering pipelines
- Use popular AutoML libraries for feature engineering
- Apply best practices for automated feature creation

## Introduction

Automated feature engineering (AutoFE) automatically creates, selects, and transforms features from raw data to improve model performance while reducing manual effort.

### Key Benefits
- **Efficiency**: Reduces time and effort in feature creation
- **Discovery**: Finds non-obvious feature combinations
- **Scalability**: Handles large feature spaces systematically
- **Consistency**: Applies transformations uniformly
- **Performance**: Often improves model accuracy

## Core Concepts

### 1. Feature Generation
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations

# Polynomial feature generation
def generate_polynomial_features(df, degree=2):
    """Generate polynomial features automatically"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(df[numeric_cols])
    
    # Get feature names
    feature_names = poly.get_feature_names_out(numeric_cols)
    
    return pd.DataFrame(poly_features, columns=feature_names, index=df.index)

# Mathematical transformations
def generate_math_features(df):
    """Generate mathematical transformations"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    new_features = df.copy()
    
    for col in numeric_cols:
        # Log transformation (handle zeros)
        new_features[f'{col}_log'] = np.log1p(df[col].abs())
        
        # Square root
        new_features[f'{col}_sqrt'] = np.sqrt(df[col].abs())
        
        # Square
        new_features[f'{col}_squared'] = df[col] ** 2
        
        # Reciprocal (handle zeros)
        new_features[f'{col}_reciprocal'] = 1 / (df[col] + 1e-8)
    
    return new_features
```

### 2. Feature Interactions
```python
def generate_feature_interactions(df, max_combinations=2):
    """Generate feature interaction terms"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    new_features = df.copy()
    
    # Pairwise interactions
    for col1, col2 in combinations(numeric_cols, 2):
        # Multiplication
        new_features[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        
        # Division (handle zeros)
        new_features[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
        
        # Addition
        new_features[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
        
        # Subtraction
        new_features[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
    
    return new_features

# Time-based features for datetime columns
def generate_temporal_features(df, datetime_col):
    """Generate temporal features from datetime column"""
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    # Extract temporal components
    df[f'{datetime_col}_year'] = df[datetime_col].dt.year
    df[f'{datetime_col}_month'] = df[datetime_col].dt.month
    df[f'{datetime_col}_day'] = df[datetime_col].dt.day
    df[f'{datetime_col}_hour'] = df[datetime_col].dt.hour
    df[f'{datetime_col}_dayofweek'] = df[datetime_col].dt.dayofweek
    df[f'{datetime_col}_quarter'] = df[datetime_col].dt.quarter
    df[f'{datetime_col}_is_weekend'] = df[datetime_col].dt.dayofweek.isin([5, 6])
    
    return df
```

### 3. Statistical Features
```python
def generate_statistical_features(df, window_size=5):
    """Generate rolling statistical features"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    new_features = df.copy()
    
    for col in numeric_cols:
        # Rolling statistics
        new_features[f'{col}_rolling_mean'] = df[col].rolling(window=window_size).mean()
        new_features[f'{col}_rolling_std'] = df[col].rolling(window=window_size).std()
        new_features[f'{col}_rolling_min'] = df[col].rolling(window=window_size).min()
        new_features[f'{col}_rolling_max'] = df[col].rolling(window=window_size).max()
        
        # Lag features
        new_features[f'{col}_lag_1'] = df[col].shift(1)
        new_features[f'{col}_lag_2'] = df[col].shift(2)
        
        # Difference features
        new_features[f'{col}_diff_1'] = df[col].diff(1)
        new_features[f'{col}_diff_2'] = df[col].diff(2)
    
    return new_features
```

## Popular AutoML Libraries

### 1. FeatureTools
```python
import featuretools as ft
import pandas as pd

# Example dataset
def create_sample_data():
    customers = pd.DataFrame({
        'customer_id': [1, 2, 3],
        'age': [25, 30, 35],
        'income': [50000, 75000, 100000]
    })
    
    transactions = pd.DataFrame({
        'transaction_id': [1, 2, 3, 4, 5],
        'customer_id': [1, 1, 2, 2, 3],
        'amount': [100, 200, 150, 300, 250],
        'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', 
                                   '2023-01-01', '2023-01-03', '2023-01-02'])
    })
    
    return customers, transactions

def automated_feature_engineering_featuretools():
    """Automated feature engineering with FeatureTools"""
    customers, transactions = create_sample_data()
    
    # Create EntitySet
    es = ft.EntitySet(id='customer_data')
    
    # Add entities
    es = es.add_dataframe(
        dataframe_name='customers',
        dataframe=customers,
        index='customer_id'
    )
    
    es = es.add_dataframe(
        dataframe_name='transactions',
        dataframe=transactions,
        index='transaction_id',
        time_index='timestamp'
    )
    
    # Add relationship
    relationship = ft.Relationship(
        parent_dataframe_name='customers',
        parent_column_name='customer_id',
        child_dataframe_name='transactions',
        child_column_name='customer_id'
    )
    es = es.add_relationship(relationship)
    
    # Deep Feature Synthesis (DFS)
    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name='customers',
        max_depth=2,
        verbose=True
    )
    
    return feature_matrix, feature_defs

# Run automated feature engineering
features, definitions = automated_feature_engineering_featuretools()
print("Generated Features:")
print(features.head())
print("\nFeature Definitions:")
for def_ in definitions[:5]:  # Show first 5
    print(f"- {def_}")
```

### 2. TPOT (Tree-based Pipeline Optimization Tool)
```python
from tpot import TPOTClassifier, TPOTRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

def automated_pipeline_tpot_classification():
    """Automated ML pipeline with TPOT for classification"""
    # Generate sample data
    X, y = make_classification(
        n_samples=1000, 
        n_features=10, 
        n_informative=5, 
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize TPOT
    tpot = TPOTClassifier(
        generations=5,
        population_size=20,
        verbosity=2,
        random_state=42,
        config_dict='TPOT light'  # Faster configuration
    )
    
    # Fit TPOT
    tpot.fit(X_train, y_train)
    
    # Make predictions
    y_pred = tpot.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Export optimized pipeline
    tpot.export('tpot_optimized_pipeline.py')
    
    return tpot

def automated_pipeline_tpot_regression():
    """Automated ML pipeline with TPOT for regression"""
    # Generate sample data
    X, y = make_regression(
        n_samples=1000, 
        n_features=10, 
        noise=0.1, 
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize TPOT
    tpot = TPOTRegressor(
        generations=5,
        population_size=20,
        verbosity=2,
        random_state=42
    )
    
    # Fit TPOT
    tpot.fit(X_train, y_train)
    
    # Make predictions
    y_pred = tpot.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE: {mse:.4f}")
    
    return tpot
```

### 3. AutoFeat
```python
# Install: pip install autofeat
try:
    from autofeat import AutoFeatClassifier, AutoFeatRegressor
    
    def automated_feature_generation_autofeat():
        """Automated feature generation with AutoFeat"""
        from sklearn.datasets import make_classification
        
        # Generate sample data
        X, y = make_classification(
            n_samples=1000, 
            n_features=5, 
            n_informative=3, 
            random_state=42
        )
        
        # Convert to DataFrame
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        # Initialize AutoFeat
        afc = AutoFeatClassifier(
            feateng_steps=2,
            max_gb=1,  # Memory limit in GB
            verbose=1
        )
        
        # Fit and transform
        X_transformed = afc.fit_transform(X_df, y)
        
        print(f"Original features: {X.shape[1]}")
        print(f"Generated features: {X_transformed.shape[1]}")
        print(f"Selected features: {len(afc.good_cols_)}")
        
        return X_transformed, afc
        
except ImportError:
    print("AutoFeat not installed. Install with: pip install autofeat")
```

## Custom Automated Feature Engineering Pipeline

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AutomatedFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom automated feature engineering pipeline"""
    
    def __init__(self, 
                 generate_polynomial=True, 
                 polynomial_degree=2,
                 generate_interactions=True,
                 generate_math_transforms=True,
                 select_k_best=None,
                 scale_features=True):
        self.generate_polynomial = generate_polynomial
        self.polynomial_degree = polynomial_degree
        self.generate_interactions = generate_interactions
        self.generate_math_transforms = generate_math_transforms
        self.select_k_best = select_k_best
        self.scale_features = scale_features
        
    def fit(self, X, y=None):
        """Fit the feature engineer"""
        self.feature_names_ = X.columns.tolist()
        self.numeric_features_ = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Fit transformations
        X_transformed = self._transform_features(X)
        
        # Fit feature selector
        if self.select_k_best and y is not None:
            self.selector_ = SelectKBest(
                score_func=f_classif, 
                k=min(self.select_k_best, X_transformed.shape[1])
            )
            self.selector_.fit(X_transformed, y)
        
        # Fit scaler
        if self.scale_features:
            if hasattr(self, 'selector_'):
                X_selected = self.selector_.transform(X_transformed)
            else:
                X_selected = X_transformed
                
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X_selected)
        
        return self
    
    def transform(self, X):
        """Transform features"""
        X_transformed = self._transform_features(X)
        
        # Apply feature selection
        if hasattr(self, 'selector_'):
            X_transformed = self.selector_.transform(X_transformed)
        
        # Apply scaling
        if hasattr(self, 'scaler_'):
            X_transformed = self.scaler_.transform(X_transformed)
        
        return X_transformed
    
    def _transform_features(self, X):
        """Apply feature transformations"""
        result = X.copy()
        
        # Generate polynomial features
        if self.generate_polynomial and self.numeric_features_:
            poly_features = generate_polynomial_features(
                X[self.numeric_features_], 
                degree=self.polynomial_degree
            )
            # Remove original features to avoid duplication
            poly_features = poly_features.drop(columns=self.numeric_features_)
            result = pd.concat([result, poly_features], axis=1)
        
        # Generate mathematical transformations
        if self.generate_math_transforms and self.numeric_features_:
            math_features = generate_math_features(X[self.numeric_features_])
            # Remove original features
            math_features = math_features.drop(columns=self.numeric_features_)
            result = pd.concat([result, math_features], axis=1)
        
        # Generate feature interactions
        if self.generate_interactions and len(self.numeric_features_) > 1:
            interaction_features = generate_feature_interactions(
                X[self.numeric_features_]
            )
            # Remove original features
            interaction_features = interaction_features.drop(columns=self.numeric_features_)
            result = pd.concat([result, interaction_features], axis=1)
        
        # Handle infinite and NaN values
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.fillna(0)
        
        return result

# Example usage
def example_automated_feature_engineering():
    """Example of using custom automated feature engineering"""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000, 
        n_features=5, 
        n_informative=3, 
        random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
    
    # Initialize automated feature engineer
    auto_fe = AutomatedFeatureEngineer(
        generate_polynomial=True,
        polynomial_degree=2,
        generate_interactions=True,
        generate_math_transforms=True,
        select_k_best=20,
        scale_features=True
    )
    
    # Fit and transform
    X_train_transformed = auto_fe.fit_transform(X_train, y_train)
    X_test_transformed = auto_fe.transform(X_test)
    
    print(f"Original features: {X_train.shape[1]}")
    print(f"Generated features: {X_train_transformed.shape[1]}")
    
    # Train model with original features
    rf_original = RandomForestClassifier(random_state=42)
    rf_original.fit(X_train, y_train)
    y_pred_original = rf_original.predict(X_test)
    acc_original = accuracy_score(y_test, y_pred_original)
    
    # Train model with engineered features
    rf_engineered = RandomForestClassifier(random_state=42)
    rf_engineered.fit(X_train_transformed, y_train)
    y_pred_engineered = rf_engineered.predict(X_test_transformed)
    acc_engineered = accuracy_score(y_test, y_pred_engineered)
    
    print(f"Accuracy with original features: {acc_original:.4f}")
    print(f"Accuracy with engineered features: {acc_engineered:.4f}")
    print(f"Improvement: {acc_engineered - acc_original:.4f}")
    
    return auto_fe

# Run example
feature_engineer = example_automated_feature_engineering()
```

## Best Practices

### 1. Feature Generation Strategy
```python
def strategic_feature_generation(df, target_col=None):
    """Strategic approach to feature generation"""
    
    # 1. Understand data types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    print(f"Datetime columns: {len(datetime_cols)}")
    
    # 2. Generate features based on data types
    new_features = df.copy()
    
    # Numeric features
    if numeric_cols:
        # Statistical features
        new_features = generate_statistical_features(new_features)
        
        # Mathematical transformations
        new_features = generate_math_features(new_features[numeric_cols])
        new_features = pd.concat([df, new_features.drop(columns=numeric_cols)], axis=1)
    
    # Categorical features
    for col in categorical_cols:
        # Frequency encoding
        freq_map = df[col].value_counts().to_dict()
        new_features[f'{col}_frequency'] = df[col].map(freq_map)
        
        # Target encoding (if target is provided)
        if target_col and target_col in df.columns:
            target_mean = df.groupby(col)[target_col].mean()
            new_features[f'{col}_target_encoded'] = df[col].map(target_mean)
    
    # Datetime features
    for col in datetime_cols:
        new_features = generate_temporal_features(new_features, col)
    
    return new_features

# Feature validation
def validate_generated_features(X_original, X_generated, y=None):
    """Validate generated features"""
    print(f"Original shape: {X_original.shape}")
    print(f"Generated shape: {X_generated.shape}")
    print(f"Features added: {X_generated.shape[1] - X_original.shape[1]}")
    
    # Check for infinite values
    inf_count = np.isinf(X_generated).sum().sum()
    print(f"Infinite values: {inf_count}")
    
    # Check for NaN values
    nan_count = X_generated.isnull().sum().sum()
    print(f"NaN values: {nan_count}")
    
    # Memory usage
    memory_mb = X_generated.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory usage: {memory_mb:.2f} MB")
    
    # Feature correlation with target (if provided)
    if y is not None:
        correlations = X_generated.corrwith(pd.Series(y)).abs().sort_values(ascending=False)
        print(f"\nTop 10 features by correlation with target:")
        print(correlations.head(10))
    
    return {
        'inf_count': inf_count,
        'nan_count': nan_count,
        'memory_mb': memory_mb
    }
```

### 2. Feature Selection Integration
```python
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, RFE, 
    SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier

def comprehensive_feature_selection(X, y, method='auto'):
    """Comprehensive feature selection after generation"""
    
    # Remove low variance features
    variance_selector = VarianceThreshold(threshold=0.01)
    X_variance = variance_selector.fit_transform(X)
    
    print(f"After variance threshold: {X_variance.shape[1]} features")
    
    if method == 'auto':
        # Automatic method selection based on data size
        if X.shape[1] < 100:
            method = 'rfe'
        elif X.shape[1] < 1000:
            method = 'selectkbest'
        else:
            method = 'selectfrommodel'
    
    if method == 'selectkbest':
        # Select K best features
        k = min(50, X_variance.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X_variance, y)
        
    elif method == 'rfe':
        # Recursive Feature Elimination
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        n_features = min(20, X_variance.shape[1])
        selector = RFE(estimator=estimator, n_features_to_select=n_features)
        X_selected = selector.fit_transform(X_variance, y)
        
    elif method == 'selectfrommodel':
        # Select from model importance
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        selector = SelectFromModel(estimator=estimator)
        X_selected = selector.fit_transform(X_variance, y)
    
    print(f"Final selected features: {X_selected.shape[1]}")
    
    return X_selected, selector
```

## Performance Monitoring

```python
import time
import psutil
import os

class FeatureEngineeringProfiler:
    """Profile feature engineering performance"""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.operations = []
    
    def start_profiling(self):
        """Start profiling"""
        self.start_time = time.time()
        self.start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        
    def log_operation(self, operation_name, features_before, features_after):
        """Log a feature engineering operation"""
        current_time = time.time()
        current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        
        self.operations.append({
            'operation': operation_name,
            'time': current_time - self.start_time,
            'memory_mb': current_memory,
            'features_before': features_before,
            'features_after': features_after,
            'features_added': features_after - features_before
        })
    
    def get_report(self):
        """Get profiling report"""
        if not self.operations:
            return "No operations logged"
        
        total_time = self.operations[-1]['time']
        total_memory = self.operations[-1]['memory_mb'] - self.start_memory
        total_features = sum(op['features_added'] for op in self.operations)
        
        report = f"""
Feature Engineering Profiling Report
=====================================
Total time: {total_time:.2f} seconds
Memory increase: {total_memory:.2f} MB
Total features generated: {total_features}
Features per second: {total_features/total_time:.2f}

Operations breakdown:
"""
        
        for op in self.operations:
            report += f"- {op['operation']}: +{op['features_added']} features in {op['time']:.2f}s\n"
        
        return report

# Example usage
def profiled_feature_engineering_example():
    """Example with profiling"""
    from sklearn.datasets import make_classification
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    # Initialize profiler
    profiler = FeatureEngineeringProfiler()
    profiler.start_profiling()
    
    # Feature engineering with profiling
    current_features = X_df.shape[1]
    
    # Polynomial features
    poly_features = generate_polynomial_features(X_df)
    profiler.log_operation('Polynomial Features', current_features, poly_features.shape[1])
    current_features = poly_features.shape[1]
    
    # Math transformations
    math_features = generate_math_features(poly_features)
    profiler.log_operation('Math Transformations', current_features, math_features.shape[1])
    current_features = math_features.shape[1]
    
    # Interactions
    interaction_features = generate_feature_interactions(math_features)
    profiler.log_operation('Feature Interactions', current_features, interaction_features.shape[1])
    
    # Print report
    print(profiler.get_report())
    
    return interaction_features

# Run profiled example
result = profiled_feature_engineering_example()
```

## Common Pitfalls and Solutions

### 1. Overfitting Prevention
```python
def prevent_overfitting_features(X_train, X_val, y_train, y_val, max_features=100):
    """Prevent overfitting in automated feature engineering"""
    
    # Start with basic features
    auto_fe = AutomatedFeatureEngineer(
        generate_polynomial=False,
        generate_interactions=False,
        generate_math_transforms=True,
        select_k_best=None,
        scale_features=True
    )
    
    # Incrementally add feature types
    feature_types = [
        ('math_transforms', True, False, False),
        ('polynomial', True, True, False),
        ('interactions', True, True, True)
    ]
    
    best_score = 0
    best_config = None
    
    for name, math, poly, interact in feature_types:
        auto_fe.generate_math_transforms = math
        auto_fe.generate_polynomial = poly
        auto_fe.generate_interactions = interact
        
        # Fit and transform
        X_train_transformed = auto_fe.fit_transform(X_train, y_train)
        X_val_transformed = auto_fe.transform(X_val)
        
        # Limit features if too many
        if X_train_transformed.shape[1] > max_features:
            auto_fe.select_k_best = max_features
            X_train_transformed = auto_fe.fit_transform(X_train, y_train)
            X_val_transformed = auto_fe.transform(X_val)
        
        # Train simple model for validation
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42)
        model.fit(X_train_transformed, y_train)
        score = model.score(X_val_transformed, y_val)
        
        print(f"{name}: {X_train_transformed.shape[1]} features, score: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_config = (name, math, poly, interact)
    
    print(f"\nBest configuration: {best_config[0]} (score: {best_score:.4f})")
    return best_config
```

### 2. Memory Management
```python
def memory_efficient_feature_engineering(df, chunk_size=1000):
    """Memory-efficient feature engineering for large datasets"""
    
    # Process in chunks
    feature_chunks = []
    
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size].copy()
        
        # Generate features for chunk
        chunk_features = generate_math_features(chunk)
        
        # Append to list
        feature_chunks.append(chunk_features)
        
        # Clear memory
        del chunk
        
        if i % (chunk_size * 10) == 0:
            print(f"Processed {min(i + chunk_size, len(df))}/{len(df)} rows")
    
    # Combine chunks
    result = pd.concat(feature_chunks, ignore_index=True)
    
    return result
```

## Summary

Automated feature engineering accelerates the ML pipeline by:

1. **Systematic Generation**: Creates comprehensive feature sets automatically
2. **Pattern Discovery**: Finds complex relationships in data
3. **Efficiency**: Reduces manual feature engineering time
4. **Scalability**: Handles large feature spaces effectively
5. **Performance**: Often improves model accuracy

### Key Takeaways
- Use domain knowledge to guide automated processes
- Monitor for overfitting and computational constraints
- Combine multiple AutoFE techniques for best results
- Always validate generated features
- Consider memory and time limitations

### Next Steps
- Experiment with different AutoML libraries
- Develop custom feature generation strategies
- Integrate with MLOps pipelines
- Explore domain-specific feature engineering patterns