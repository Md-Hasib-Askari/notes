# SHAP (SHapley Additive exPlanations)

## Learning Objectives
- Understand SHAP values and their theoretical foundation
- Implement SHAP for different types of models (tree-based, linear, deep learning)
- Create various SHAP visualizations for model interpretation
- Apply SHAP for feature importance and model debugging
- Use SHAP for global and local explanations
- Integrate SHAP into ML workflows for production interpretability

## Introduction

SHAP (SHapley Additive exPlanations) is a unified framework for explaining machine learning model predictions based on cooperative game theory. It provides consistent and theoretically grounded explanations by computing the contribution of each feature to individual predictions.

### Key Concepts
- **Shapley Values**: Fair allocation of contributions from cooperative game theory
- **Additive Feature Attribution**: Explanation model where feature contributions sum to prediction
- **Efficiency**: Sum of SHAP values equals difference between prediction and expected value
- **Symmetry**: Features with equal marginal contributions get equal SHAP values
- **Dummy**: Features that don't affect predictions have zero SHAP values

## Theoretical Foundation

### 1. Shapley Values from Game Theory
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import shap
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression, load_boston
import warnings
warnings.filterwarnings('ignore')

# Set up visualization
shap.initjs()

def calculate_shapley_value_manual(model, X_instance, X_background, feature_idx):
    """
    Calculate Shapley value manually for educational purposes
    """
    n_features = len(X_instance)
    feature_indices = list(range(n_features))
    feature_indices.remove(feature_idx)
    
    shapley_value = 0
    
    # Iterate over all possible coalitions
    for r in range(n_features):
        # All combinations of size r not including the target feature
        for coalition in combinations(feature_indices, r):
            coalition = list(coalition)
            
            # Coalition with target feature
            coalition_with_feature = coalition + [feature_idx]
            
            # Calculate marginal contribution
            # Prediction with coalition including target feature
            X_with = X_instance.copy()
            X_without = X_instance.copy()
            
            # Set non-coalition features to background values
            for i in range(n_features):
                if i not in coalition_with_feature:
                    X_with[i] = X_background[i]
                if i not in coalition:
                    X_without[i] = X_background[i]
            
            # Marginal contribution
            contribution = model.predict([X_with])[0] - model.predict([X_without])[0]
            
            # Weight by coalition size
            weight = 1.0 / (n_features * np.math.comb(n_features - 1, r))
            
            shapley_value += weight * contribution
    
    return shapley_value

def demonstrate_shapley_properties():
    """Demonstrate key properties of Shapley values"""
    
    # Create simple dataset
    X, y = make_regression(n_samples=1000, n_features=4, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Create SHAP explainer
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test[:5])
    
    print("SHAP Properties Demonstration:")
    print("=" * 40)
    
    for i in range(5):
        instance = X_test[i]
        prediction = model.predict([instance])[0]
        expected_value = explainer.expected_value
        shap_sum = np.sum(shap_values[i].values)
        
        print(f"\nInstance {i+1}:")
        print(f"Prediction: {prediction:.4f}")
        print(f"Expected value: {expected_value:.4f}")
        print(f"Sum of SHAP values: {shap_sum:.4f}")
        print(f"Expected + SHAP sum: {expected_value + shap_sum:.4f}")
        print(f"Efficiency check (should be ~0): {abs(prediction - (expected_value + shap_sum)):.6f}")

# Run demonstration
demonstrate_shapley_properties()
```

### 2. SHAP Explainer Types
```python
class SHAPExplainerManager:
    """Manage different types of SHAP explainers"""
    
    def __init__(self):
        self.explainers = {}
        self.explanations = {}
    
    def create_tree_explainer(self, model, X_background=None):
        """Create TreeExplainer for tree-based models"""
        try:
            explainer = shap.TreeExplainer(model, X_background)
            self.explainers['tree'] = explainer
            print("TreeExplainer created successfully")
            return explainer
        except Exception as e:
            print(f"Error creating TreeExplainer: {e}")
            return None
    
    def create_linear_explainer(self, model, X_background):
        """Create LinearExplainer for linear models"""
        try:
            explainer = shap.LinearExplainer(model, X_background)
            self.explainers['linear'] = explainer
            print("LinearExplainer created successfully")
            return explainer
        except Exception as e:
            print(f"Error creating LinearExplainer: {e}")
            return None
    
    def create_kernel_explainer(self, model_predict_func, X_background):
        """Create KernelExplainer for any model"""
        try:
            explainer = shap.KernelExplainer(model_predict_func, X_background)
            self.explainers['kernel'] = explainer
            print("KernelExplainer created successfully")
            return explainer
        except Exception as e:
            print(f"Error creating KernelExplainer: {e}")
            return None
    
    def create_deep_explainer(self, model, X_background):
        """Create DeepExplainer for deep learning models"""
        try:
            explainer = shap.DeepExplainer(model, X_background)
            self.explainers['deep'] = explainer
            print("DeepExplainer created successfully")
            return explainer
        except Exception as e:
            print(f"Error creating DeepExplainer: {e}")
            return None
    
    def create_permutation_explainer(self, model_predict_func, X_background):
        """Create PermutationExplainer"""
        try:
            explainer = shap.PermutationExplainer(model_predict_func, X_background)
            self.explainers['permutation'] = explainer
            print("PermutationExplainer created successfully")
            return explainer
        except Exception as e:
            print(f"Error creating PermutationExplainer: {e}")
            return None
    
    def explain_instance(self, explainer_type, X_instance, **kwargs):
        """Generate SHAP values for an instance"""
        if explainer_type not in self.explainers:
            print(f"Explainer {explainer_type} not found")
            return None
        
        explainer = self.explainers[explainer_type]
        
        try:
            shap_values = explainer(X_instance, **kwargs)
            self.explanations[explainer_type] = shap_values
            return shap_values
        except Exception as e:
            print(f"Error generating explanations: {e}")
            return None

# Example usage for different model types
def demonstrate_explainer_types():
    """Demonstrate different SHAP explainer types"""
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    manager = SHAPExplainerManager()
    
    # 1. Tree-based model with TreeExplainer
    print("1. Tree-based Model (Random Forest)")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    tree_explainer = manager.create_tree_explainer(rf_model)
    if tree_explainer:
        tree_shap_values = manager.explain_instance('tree', X_test[:5])
        print(f"Tree SHAP values shape: {tree_shap_values.values.shape}")
    
    # 2. Linear model with LinearExplainer
    print("\n2. Linear Model (Logistic Regression)")
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)
    
    linear_explainer = manager.create_linear_explainer(lr_model, X_train)
    if linear_explainer:
        linear_shap_values = manager.explain_instance('linear', X_test[:5])
        print(f"Linear SHAP values shape: {linear_shap_values.values.shape}")
    
    # 3. Any model with KernelExplainer (slower but universal)
    print("\n3. Universal Model (Kernel SHAP)")
    def model_predict(X):
        return rf_model.predict_proba(X)[:, 1]
    
    # Use smaller background for kernel explainer (for speed)
    kernel_explainer = manager.create_kernel_explainer(model_predict, X_train[:50])
    if kernel_explainer:
        kernel_shap_values = manager.explain_instance('kernel', X_test[:2])  # Fewer instances for speed
        print(f"Kernel SHAP values shape: {kernel_shap_values.values.shape}")
    
    return manager

# Run demonstration
manager = demonstrate_explainer_types()
```

## SHAP for Different Model Types

### 1. Tree-Based Models
```python
def comprehensive_tree_analysis():
    """Comprehensive SHAP analysis for tree-based models"""
    
    # Load dataset
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train different tree models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nAnalyzing {model_name}")
        print("=" * 40)
        
        # Train model
        if model_name == 'Gradient Boosting':
            # Convert to regression problem for demonstration
            model.fit(X_train, y_train.astype(float))
        else:
            model.fit(X_train, y_train)
        
        # Create TreeExplainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)
        
        # Store results
        results[model_name] = {
            'model': model,
            'explainer': explainer,
            'shap_values': shap_values,
            'feature_names': feature_names
        }
        
        # Basic statistics
        print(f"Expected value: {explainer.expected_value}")
        print(f"SHAP values shape: {shap_values.values.shape}")
        
        # Feature importance
        feature_importance = np.abs(shap_values.values).mean(0)
        top_features = np.argsort(feature_importance)[-5:]
        
        print("\nTop 5 most important features:")
        for idx in reversed(top_features):
            print(f"  {feature_names[idx]}: {feature_importance[idx]:.4f}")
    
    return results

def analyze_tree_interactions():
    """Analyze feature interactions in tree models"""
    
    # Generate data with known interactions
    np.random.seed(42)
    n_samples = 1000
    
    X1 = np.random.normal(0, 1, n_samples)
    X2 = np.random.normal(0, 1, n_samples)
    X3 = np.random.normal(0, 1, n_samples)
    
    # Create interaction: y depends on X1*X2 interaction
    y = X1 * X2 + 0.5 * X3 + np.random.normal(0, 0.1, n_samples)
    
    X = np.column_stack([X1, X2, X3])
    feature_names = ['X1', 'X2', 'X3']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train tree model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # SHAP analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    
    # Interaction values
    shap_interaction_values = explainer.shap_interaction_values(X_test)
    
    print("Feature Interaction Analysis")
    print("=" * 30)
    
    # Main effects
    main_effects = np.abs(shap_values.values).mean(0)
    print("Main effects (average |SHAP value|):")
    for i, name in enumerate(feature_names):
        print(f"  {name}: {main_effects[i]:.4f}")
    
    # Interaction effects
    print("\nInteraction effects:")
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            interaction_strength = np.abs(shap_interaction_values[:, i, j]).mean()
            print(f"  {feature_names[i]} × {feature_names[j]}: {interaction_strength:.4f}")
    
    return {
        'model': model,
        'shap_values': shap_values,
        'interaction_values': shap_interaction_values,
        'feature_names': feature_names,
        'X_test': X_test
    }

# Run tree analyses
tree_results = comprehensive_tree_analysis()
interaction_results = analyze_tree_interactions()
```

### 2. Linear Models
```python
def analyze_linear_models():
    """SHAP analysis for linear models"""
    
    # Generate dataset with correlated features
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    # Create correlated features
    X = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=0.3 * np.ones((n_features, n_features)) + 0.7 * np.eye(n_features),
        size=n_samples
    )
    
    # True coefficients
    true_coef = np.array([2.0, -1.5, 0.8, 0.0, 1.2])
    y = X @ true_coef + np.random.normal(0, 0.1, n_samples)
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train linear model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # SHAP analysis
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer(X_test)
    
    print("Linear Model SHAP Analysis")
    print("=" * 30)
    
    print("True coefficients:", true_coef)
    print("Model coefficients:", model.coef_)
    print("Expected value:", explainer.expected_value)
    
    # Compare SHAP values with coefficients
    print("\nSHAP vs Coefficients comparison:")
    mean_shap = np.abs(shap_values.values).mean(0)
    
    for i, name in enumerate(feature_names):
        print(f"{name}:")
        print(f"  True coef: {true_coef[i]:.3f}")
        print(f"  Model coef: {model.coef_[i]:.3f}")
        print(f"  Mean |SHAP|: {mean_shap[i]:.3f}")
    
    # Demonstrate linearity property
    print("\nLinearity Check (SHAP values should be linear combinations of features):")
    for i in range(3):  # Check first 3 instances
        instance = X_test[i]
        shap_manual = (instance - X_train.mean(axis=0)) * model.coef_
        shap_computed = shap_values[i].values
        
        print(f"\nInstance {i+1}:")
        print(f"Manual SHAP: {shap_manual}")
        print(f"Computed SHAP: {shap_computed}")
        print(f"Difference: {np.abs(shap_manual - shap_computed).max():.6f}")
    
    return {
        'model': model,
        'explainer': explainer,
        'shap_values': shap_values,
        'true_coef': true_coef,
        'X_test': X_test,
        'feature_names': feature_names
    }

def analyze_regularized_models():
    """SHAP analysis for regularized linear models"""
    
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.preprocessing import StandardScaler
    
    # Generate high-dimensional data
    X, y = make_regression(n_samples=500, n_features=50, n_informative=10, 
                          noise=0.1, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train different regularized models
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{model_name} Analysis")
        print("=" * 20)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # SHAP analysis
        explainer = shap.LinearExplainer(model, X_train_scaled)
        shap_values = explainer(X_test_scaled)
        
        # Feature importance
        feature_importance = np.abs(shap_values.values).mean(0)
        active_features = np.sum(np.abs(model.coef_) > 1e-6)
        
        print(f"Active features: {active_features}")
        print(f"R² score: {model.score(X_test_scaled, y_test):.4f}")
        
        # Top features
        top_indices = np.argsort(feature_importance)[-5:]
        print("Top 5 features by SHAP importance:")
        for idx in reversed(top_indices):
            print(f"  Feature {idx}: {feature_importance[idx]:.4f}")
        
        results[model_name] = {
            'model': model,
            'shap_values': shap_values,
            'feature_importance': feature_importance
        }
    
    return results

# Run linear model analyses
linear_results = analyze_linear_models()
regularized_results = analyze_regularized_models()
```

### 3. Deep Learning Models
```python
try:
    import tensorflow as tf
    from tensorflow import keras
    
    def create_neural_network_shap():
        """SHAP analysis for neural networks"""
        
        # Generate dataset
        X, y = make_classification(n_samples=2000, n_features=20, n_informative=10,
                                 n_redundant=5, random_state=42)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Build neural network
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(20,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train model
        history = model.fit(X_train_scaled, y_train, 
                          epochs=50, batch_size=32, 
                          validation_split=0.2, 
                          verbose=0)
        
        # SHAP analysis
        # Use a subset of training data as background
        background = X_train_scaled[:100]
        
        # Create DeepExplainer
        explainer = shap.DeepExplainer(model, background)
        
        # Generate SHAP values for test samples
        shap_values = explainer.shap_values(X_test_scaled[:20])
        
        print("Neural Network SHAP Analysis")
        print("=" * 30)
        print(f"Model accuracy: {model.evaluate(X_test_scaled, y_test, verbose=0)[1]:.4f}")
        print(f"SHAP values shape: {np.array(shap_values).shape}")
        
        # Feature importance
        if isinstance(shap_values, list):
            shap_array = shap_values[0]  # For binary classification
        else:
            shap_array = shap_values
        
        feature_importance = np.abs(shap_array).mean(0)
        top_features = np.argsort(feature_importance)[-5:]
        
        print("\nTop 5 features by SHAP importance:")
        for idx in reversed(top_features):
            print(f"  Feature {idx}: {feature_importance[idx]:.4f}")
        
        return {
            'model': model,
            'explainer': explainer,
            'shap_values': shap_values,
            'X_test': X_test_scaled,
            'y_test': y_test,
            'scaler': scaler,
            'feature_importance': feature_importance
        }
    
    # Alternative: Using GradientExplainer for TensorFlow/Keras
    def gradient_explainer_example():
        """Example using GradientExplainer"""
        
        # Simpler model for gradient explanation
        X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Build model
        model = keras.Sequential([
            keras.layers.Dense(32, activation='relu', input_shape=(10,)),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train_scaled, y_train, epochs=100, verbose=0)
        
        # GradientExplainer
        explainer = shap.GradientExplainer(model, X_train_scaled[:50])
        shap_values = explainer.shap_values(X_test_scaled[:10])
        
        print("Gradient Explainer Results")
        print("=" * 25)
        print(f"SHAP values shape: {np.array(shap_values).shape}")
        
        return {
            'model': model,
            'explainer': explainer,
            'shap_values': shap_values
        }
    
    # Run neural network analysis
    nn_results = create_neural_network_shap()
    gradient_results = gradient_explainer_example()
    
except ImportError:
    print("TensorFlow not available. Skipping deep learning SHAP examples.")
    nn_results = None
    gradient_results = None
```

## SHAP Visualizations

### 1. Summary Plots
```python
def create_comprehensive_visualizations():
    """Create comprehensive SHAP visualizations"""
    
    # Use the tree results from earlier
    if 'tree_results' in globals():
        rf_data = tree_results['Random Forest']
        shap_values = rf_data['shap_values']
        feature_names = rf_data['feature_names']
    else:
        # Recreate if needed
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X, y = data.data, data.target
        feature_names = data.feature_names
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)
    
    # 1. Summary Plot (Beeswarm)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, features=X_test, feature_names=feature_names, show=False)
    plt.title("SHAP Summary Plot (Beeswarm)")
    plt.tight_layout()
    plt.show()
    
    # 2. Summary Plot (Bar)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, features=X_test, feature_names=feature_names, 
                     plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Bar)")
    plt.tight_layout()
    plt.show()
    
    # 3. Waterfall Plot for single prediction
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap_values[0], show=False)
    plt.title("SHAP Waterfall Plot - Single Prediction")
    plt.tight_layout()
    plt.show()
    
    # 4. Force Plot (as matplotlib)
    plt.figure(figsize=(14, 4))
    shap.force_plot(explainer.expected_value[1], shap_values[0].values, 
                   features=X_test[0], feature_names=feature_names, 
                   matplotlib=True, show=False)
    plt.title("SHAP Force Plot - Single Prediction")
    plt.tight_layout()
    plt.show()
    
    # 5. Partial Dependence Plot
    feature_idx = 0  # First feature
    plt.figure(figsize=(10, 6))
    shap.partial_dependence_plot(
        feature_idx, model.predict_proba, X_test, ice=False,
        model_expected_value=True, feature_expected_value=True, show=False)
    plt.title(f"Partial Dependence Plot - {feature_names[feature_idx]}")
    plt.tight_layout()
    plt.show()
    
    return shap_values

def create_comparison_plots():
    """Create comparison plots between different instances"""
    
    # Generate sample data for demonstration
    X, y = make_classification(n_samples=1000, n_features=8, n_informative=5, random_state=42)
    feature_names = [f'Feature_{i}' for i in range(8)]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # SHAP analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    
    # 1. Compare two instances
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 1, 1)
    shap.waterfall_plot(shap_values[0], show=False)
    plt.title("Instance 1 - SHAP Explanation")
    
    plt.subplot(2, 1, 2)
    shap.waterfall_plot(shap_values[1], show=False)
    plt.title("Instance 2 - SHAP Explanation")
    
    plt.tight_layout()
    plt.show()
    
    # 2. Decision plot for multiple instances
    plt.figure(figsize=(10, 8))
    shap.decision_plot(explainer.expected_value[1], shap_values[:5].values, 
                      features=X_test[:5], feature_names=feature_names, show=False)
    plt.title("SHAP Decision Plot - Multiple Instances")
    plt.tight_layout()
    plt.show()
    
    return model, explainer, shap_values

# Create visualizations
shap_viz_values = create_comprehensive_visualizations()
comparison_model, comparison_explainer, comparison_shap = create_comparison_plots()
```

### 2. Interactive Visualizations
```python
def create_interactive_visualizations():
    """Create interactive SHAP visualizations"""
    
    # Generate dataset
    X, y = make_classification(n_samples=500, n_features=10, n_informative=7, random_state=42)
    feature_names = [f'Feature_{i}' for i in range(10)]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # SHAP analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    
    print("Interactive SHAP Visualizations")
    print("=" * 30)
    
    # 1. Interactive Force Plot
    print("1. Creating interactive force plot...")
    force_plot = shap.force_plot(
        explainer.expected_value[1], 
        shap_values[0].values, 
        features=X_test[0], 
        feature_names=feature_names
    )
    
    # 2. Interactive Force Plot for multiple instances
    print("2. Creating interactive force plot for multiple instances...")
    force_plot_multi = shap.force_plot(
        explainer.expected_value[1], 
        shap_values[:10].values, 
        features=X_test[:10], 
        feature_names=feature_names
    )
    
    # 3. Interactive Embedding Plot
    print("3. Creating embedding plot...")
    try:
        embedding_plot = shap.embedding_plot(
            "pca", 
            shap_values.values, 
            feature_names=feature_names
        )
    except Exception as e:
        print(f"Embedding plot error: {e}")
        embedding_plot = None
    
    return {
        'force_plot': force_plot,
        'force_plot_multi': force_plot_multi,
        'embedding_plot': embedding_plot,
        'shap_values': shap_values,
        'feature_names': feature_names
    }

def create_monitoring_dashboard():
    """Create a monitoring dashboard with SHAP values"""
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Generate sample data
    X, y = make_classification(n_samples=200, n_features=6, n_informative=4, random_state=42)
    feature_names = [f'Feature_{i}' for i in range(6)]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # SHAP analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Feature Importance', 'SHAP Values Distribution', 
                       'Prediction vs SHAP Sum', 'Feature Correlations'),
        specs=[[{"type": "bar"}, {"type": "box"}],
               [{"type": "scatter"}, {"type": "heatmap"}]]
    )
    
    # 1. Feature Importance
    feature_importance = np.abs(shap_values.values).mean(0)
    fig.add_trace(
        go.Bar(x=feature_names, y=feature_importance, name="Feature Importance"),
        row=1, col=1
    )
    
    # 2. SHAP Values Distribution
    for i, feature in enumerate(feature_names[:4]):  # Show top 4 features
        fig.add_trace(
            go.Box(y=shap_values.values[:, i], name=feature),
            row=1, col=2
        )
    
    # 3. Prediction vs SHAP Sum
    predictions = model.predict_proba(X_test)[:, 1]
    shap_sum = shap_values.values.sum(axis=1) + explainer.expected_value[1]
    
    fig.add_trace(
        go.Scatter(x=predictions, y=shap_sum, mode='markers', 
                  name="Prediction vs SHAP", opacity=0.7),
        row=2, col=1
    )
    
    # Add perfect correlation line
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                  name="Perfect Correlation", line=dict(dash='dash')),
        row=2, col=1
    )
    
    # 4. Feature Correlations
    correlation_matrix = np.corrcoef(X_test.T)
    fig.add_trace(
        go.Heatmap(z=correlation_matrix, x=feature_names, y=feature_names,
                  colorscale='RdBu', zmid=0),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="SHAP Monitoring Dashboard",
        height=800,
        showlegend=True
    )
    
    fig.show()
    
    return fig

# Create interactive visualizations
interactive_viz = create_interactive_visualizations()
dashboard = create_monitoring_dashboard()
```

## Advanced SHAP Applications

### 1. Model Debugging and Validation
```python
def debug_model_with_shap():
    """Use SHAP for model debugging and validation"""
    
    # Create dataset with known bias
    np.random.seed(42)
    n_samples = 1000
    
    # Features
    age = np.random.normal(40, 15, n_samples)
    income = np.random.normal(50000, 20000, n_samples)
    credit_score = np.random.normal(700, 100, n_samples)
    
    # Introduce bias: gender affects outcome unfairly
    gender = np.random.binomial(1, 0.5, n_samples)  # 0: female, 1: male
    
    # Target: loan approval (biased by gender)
    loan_approval = (
        0.3 * (age - 40) / 15 +
        0.4 * (income - 50000) / 20000 +
        0.5 * (credit_score - 700) / 100 +
        0.3 * gender +  # Unfair bias
        np.random.normal(0, 0.1, n_samples)
    ) > 0
    
    X = np.column_stack([age, income, credit_score, gender])
    feature_names = ['Age', 'Income', 'Credit_Score', 'Gender']
    y = loan_approval.astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("Model Debugging with SHAP")
    print("=" * 25)
    print(f"Model accuracy: {model.score(X_test, y_test):.4f}")
    
    # SHAP analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    
    # 1. Overall feature importance
    feature_importance = np.abs(shap_values.values).mean(0)
    print("\nFeature Importance (mean |SHAP|):")
    for i, name in enumerate(feature_names):
        print(f"  {name}: {feature_importance[i]:.4f}")
    
    # 2. Check for bias - compare SHAP values by gender
    male_indices = X_test[:, 3] == 1
    female_indices = X_test[:, 3] == 0
    
    male_shap_mean = shap_values.values[male_indices].mean(axis=0)
    female_shap_mean = shap_values.values[female_indices].mean(axis=0)
    
    print("\nBias Analysis - Average SHAP values by group:")
    print("Feature\t\tMale\t\tFemale\t\tDifference")
    print("-" * 50)
    for i, name in enumerate(feature_names):
        diff = male_shap_mean[i] - female_shap_mean[i]
        print(f"{name:<12}\t{male_shap_mean[i]:.4f}\t\t{female_shap_mean[i]:.4f}\t\t{diff:.4f}")
    
    # 3. Identify problematic predictions
    predictions = model.predict_proba(X_test)[:, 1]
    
    # Find cases where gender has high impact
    gender_impact = np.abs(shap_values.values[:, 3])
    high_gender_impact = gender_impact > np.percentile(gender_impact, 90)
    
    print(f"\nHigh gender impact cases: {np.sum(high_gender_impact)}")
    print("Sample problematic cases:")
    
    problematic_indices = np.where(high_gender_impact)[0][:5]
    for idx in problematic_indices:
        print(f"\nCase {idx}:")
        print(f"  Features: Age={X_test[idx, 0]:.1f}, Income=${X_test[idx, 1]:.0f}, "
              f"Credit={X_test[idx, 2]:.0f}, Gender={'M' if X_test[idx, 3] else 'F'}")
        print(f"  Prediction: {predictions[idx]:.3f}")
        print(f"  Gender SHAP: {shap_values.values[idx, 3]:.4f}")
    
    return {
        'model': model,
        'shap_values': shap_values,
        'X_test': X_test,
        'feature_names': feature_names,
        'bias_analysis': {
            'male_shap_mean': male_shap_mean,
            'female_shap_mean': female_shap_mean
        }
    }

def validate_model_consistency():
    """Validate model consistency using SHAP"""
    
    # Generate dataset
    X, y = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)
    feature_names = [f'Feature_{i}' for i in range(5)]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train multiple models for comparison
    models = {
        'Linear': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    shap_results = {}
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        
        if model_name == 'Linear':
            explainer = shap.LinearExplainer(model, X_train)
        else:
            explainer = shap.TreeExplainer(model)
        
        shap_values = explainer(X_test)
        shap_results[model_name] = {
            'model': model,
            'shap_values': shap_values,
            'score': model.score(X_test, y_test)
        }
    
    print("Model Consistency Validation")
    print("=" * 28)
    
    # Compare feature importance across models
    print("Feature importance comparison:")
    print("Feature\t\t", end="")
    for model_name in models.keys():
        print(f"{model_name:<15}", end="")
    print()
    
    for i, feature in enumerate(feature_names):
        print(f"{feature:<12}\t", end="")
        for model_name in models.keys():
            importance = np.abs(shap_results[model_name]['shap_values'].values).mean(0)[i]
            print(f"{importance:<15.4f}", end="")
        print()
    
    # Check agreement between models
    linear_shap = shap_results['Linear']['shap_values'].values
    rf_shap = shap_results['Random Forest']['shap_values'].values
    
    # Correlation of SHAP values
    correlations = []
    for i in range(len(feature_names)):
        corr = np.corrcoef(linear_shap[:, i], rf_shap[:, i])[0, 1]
        correlations.append(corr)
        print(f"\nSHAP correlation (Linear vs RF) for {feature_names[i]}: {corr:.4f}")
    
    return shap_results, correlations

# Run debugging and validation
debug_results = debug_model_with_shap()
validation_results, correlations = validate_model_consistency()
```

### 2. Production Integration
```python
def create_shap_monitoring_service():
    """Create a SHAP monitoring service for production"""
    
    import json
    from datetime import datetime
    import pickle
    
    class SHAPMonitoringService:
        """Service for monitoring model explanations in production"""
        
        def __init__(self, model, explainer, feature_names, 
                     explanation_threshold=0.1, storage_path="shap_logs/"):
            self.model = model
            self.explainer = explainer
            self.feature_names = feature_names
            self.explanation_threshold = explanation_threshold
            self.storage_path = storage_path
            self.explanation_history = []
            
            # Create storage directory
            import os
            os.makedirs(storage_path, exist_ok=True)
        
        def explain_prediction(self, X_instance, prediction_id=None):
            """Generate explanation for a single prediction"""
            
            # Ensure X_instance is 2D
            if X_instance.ndim == 1:
                X_instance = X_instance.reshape(1, -1)
            
            # Generate SHAP values
            shap_values = self.explainer(X_instance)
            
            # Get prediction
            if hasattr(self.model, 'predict_proba'):
                prediction = self.model.predict_proba(X_instance)[0]
                predicted_class = self.model.predict(X_instance)[0]
            else:
                prediction = self.model.predict(X_instance)[0]
                predicted_class = None
            
            # Create explanation record
            explanation = {
                'prediction_id': prediction_id or f"pred_{len(self.explanation_history)}",
                'timestamp': datetime.now().isoformat(),
                'features': X_instance[0].tolist(),
                'feature_names': self.feature_names,
                'prediction': float(prediction[1]) if predicted_class is not None else float(prediction),
                'predicted_class': int(predicted_class) if predicted_class is not None else None,
                'shap_values': shap_values[0].values.tolist(),
                'shap_expected_value': float(self.explainer.expected_value[1] if hasattr(self.explainer.expected_value, '__len__') else self.explainer.expected_value),
                'top_features': self._get_top_features(shap_values[0].values),
                'explanation_summary': self._create_explanation_summary(shap_values[0].values)
            }
            
            # Store explanation
            self.explanation_history.append(explanation)
            self._save_explanation(explanation)
            
            # Check for anomalies
            anomalies = self._detect_explanation_anomalies(explanation)
            
            return {
                'explanation': explanation,
                'anomalies': anomalies
            }
        
        def _get_top_features(self, shap_values, top_k=5):
            """Get top contributing features"""
            abs_shap = np.abs(shap_values)
            top_indices = np.argsort(abs_shap)[-top_k:][::-1]
            
            return [
                {
                    'feature': self.feature_names[idx],
                    'shap_value': float(shap_values[idx]),
                    'contribution': float(abs_shap[idx])
                }
                for idx in top_indices
            ]
        
        def _create_explanation_summary(self, shap_values):
            """Create human-readable explanation summary"""
            top_positive = []
            top_negative = []
            
            for i, (feature, shap_val) in enumerate(zip(self.feature_names, shap_values)):
                if shap_val > self.explanation_threshold:
                    top_positive.append((feature, shap_val))
                elif shap_val < -self.explanation_threshold:
                    top_negative.append((feature, shap_val))
            
            # Sort by absolute value
            top_positive.sort(key=lambda x: abs(x[1]), reverse=True)
            top_negative.sort(key=lambda x: abs(x[1]), reverse=True)
            
            summary = "Prediction factors:\n"
            
            if top_positive:
                summary += "Factors increasing prediction:\n"
                for feature, value in top_positive[:3]:
                    summary += f"  - {feature}: +{value:.4f}\n"
            
            if top_negative:
                summary += "Factors decreasing prediction:\n"
                for feature, value in top_negative[:3]:
                    summary += f"  - {feature}: {value:.4f}\n"
            
            return summary
        
        def _detect_explanation_anomalies(self, explanation):
            """Detect anomalies in explanations"""
            anomalies = []
            shap_values = np.array(explanation['shap_values'])
            
            # Check for extreme SHAP values
            extreme_threshold = 3 * np.std([np.abs(exp['shap_values']).max() 
                                          for exp in self.explanation_history[-100:]])
            
            if np.abs(shap_values).max() > extreme_threshold:
                anomalies.append({
                    'type': 'extreme_shap_value',
                    'message': f"Extreme SHAP value detected: {np.abs(shap_values).max():.4f}",
                    'threshold': extreme_threshold
                })
            
            # Check for unexpected feature importance
            current_importance = np.abs(shap_values)
            if len(self.explanation_history) > 10:
                historical_importance = np.mean([
                    np.abs(exp['shap_values']) for exp in self.explanation_history[-100:]
                ], axis=0)
                
                importance_change = np.abs(current_importance - historical_importance)
                if np.max(importance_change) > 0.5 * np.mean(historical_importance):
                    anomalies.append({
                        'type': 'importance_shift',
                        'message': "Significant shift in feature importance detected",
                        'max_change': float(np.max(importance_change))
                    })
            
            return anomalies
        
        def _save_explanation(self, explanation):
            """Save explanation to file"""
            filename = f"{self.storage_path}/explanation_{explanation['prediction_id']}.json"
            with open(filename, 'w') as f:
                json.dump(explanation, f, indent=2)
        
        def get_explanation_statistics(self, time_window_hours=24):
            """Get statistics for recent explanations"""
            from datetime import timedelta
            
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            
            recent_explanations = [
                exp for exp in self.explanation_history
                if datetime.fromisoformat(exp['timestamp']) > cutoff_time
            ]
            
            if not recent_explanations:
                return {"message": "No recent explanations found"}
            
            # Calculate statistics
            all_shap_values = np.array([exp['shap_values'] for exp in recent_explanations])
            
            stats = {
                'total_explanations': len(recent_explanations),
                'average_prediction': np.mean([exp['prediction'] for exp in recent_explanations]),
                'feature_importance_avg': np.mean(np.abs(all_shap_values), axis=0).tolist(),
                'feature_importance_std': np.std(np.abs(all_shap_values), axis=0).tolist(),
                'most_important_features': [
                    self.feature_names[i] for i in 
                    np.argsort(np.mean(np.abs(all_shap_values), axis=0))[-5:][::-1]
                ]
            }
            
            return stats
    
    # Example usage
    # Create sample model and data
    X, y = make_classification(n_samples=1000, n_features=8, n_informative=5, random_state=42)
    feature_names = [f'Feature_{i}' for i in range(8)]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    explainer = shap.TreeExplainer(model)
    
    # Create monitoring service
    monitoring_service = SHAPMonitoringService(
        model=model,
        explainer=explainer,
        feature_names=feature_names
    )
    
    print("SHAP Monitoring Service Demo")
    print("=" * 30)
    
    # Simulate production predictions
    for i in range(10):
        instance = X_test[i]
        result = monitoring_service.explain_prediction(instance, f"prod_pred_{i}")
        
        print(f"\nPrediction {i+1}:")
        print(f"Predicted probability: {result['explanation']['prediction']:.4f}")
        print(f"Top feature: {result['explanation']['top_features'][0]['feature']}")
        print(f"Anomalies detected: {len(result['anomalies'])}")
        
        if result['anomalies']:
            for anomaly in result['anomalies']:
                print(f"  - {anomaly['type']}: {anomaly['message']}")
    
    # Get statistics
    stats = monitoring_service.get_explanation_statistics()
    print(f"\nExplanation Statistics:")
    print(f"Total explanations: {stats['total_explanations']}")
    print(f"Average prediction: {stats['average_prediction']:.4f}")
    print(f"Most important features: {stats['most_important_features'][:3]}")
    
    return monitoring_service

# Create monitoring service
monitoring_service = create_shap_monitoring_service()
```

## Summary

SHAP provides a unified framework for machine learning interpretability through:

1. **Theoretical Foundation**: Based on cooperative game theory and Shapley values
2. **Model Agnostic**: Works with any machine learning model
3. **Consistent Explanations**: Satisfies efficiency, symmetry, and dummy properties
4. **Multiple Explainer Types**: TreeExplainer, LinearExplainer, KernelExplainer, DeepExplainer
5. **Rich Visualizations**: Summary plots, waterfall plots, force plots, decision plots
6. **Production Ready**: Can be integrated into monitoring and debugging workflows

### Key Takeaways
- Choose the right explainer type for your model to ensure efficiency
- Use SHAP for both global model understanding and local prediction explanations
- Leverage visualizations to communicate insights to stakeholders
- Implement SHAP monitoring in production for model debugging and validation
- Combine SHAP with other interpretability methods for comprehensive understanding

### Next Steps
- Explore SHAP for specific domains (NLP, computer vision, time series)
- Integrate SHAP with model governance and compliance frameworks
- Implement automated explanation quality monitoring
- Use SHAP for feature selection and model improvement
- Combine with other XAI methods like LIME and anchors