# Model Interpretability

## Learning Objectives
- Understand different types of model interpretability approaches
- Learn about global vs local interpretability methods
- Implement various interpretability techniques for different model types
- Apply interpretability methods to improve model trust and debugging
- Design interpretable ML systems for production environments

## 1. Introduction to Model Interpretability

### What is Model Interpretability?
Model interpretability is the degree to which a human can understand the cause of a decision made by a machine learning model. It encompasses both the ability to predict a model's result and understand why the model made that specific decision.

### Types of Interpretability

#### Intrinsic vs Post-hoc Interpretability
- **Intrinsic**: Models that are interpretable by design (linear regression, decision trees)
- **Post-hoc**: Methods applied after training to explain black-box models

#### Global vs Local Interpretability
- **Global**: Understanding the entire model behavior
- **Local**: Understanding individual predictions

#### Model-Specific vs Model-Agnostic
- **Model-Specific**: Techniques designed for specific algorithms
- **Model-Agnostic**: Techniques that work with any model

## 2. Intrinsically Interpretable Models

### Linear Models

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Linear Regression Example
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
feature_names = diabetes.feature_names

# Train linear regression
linear_model = LinearRegression()
linear_model.fit(X, y)

# Interpretability through coefficients
coefficients_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': linear_model.coef_,
    'Abs_Coefficient': np.abs(linear_model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print("Linear Model Coefficients:")
print(coefficients_df)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=coefficients_df, x='Coefficient', y='Feature')
plt.title('Linear Model Feature Importance')
plt.xlabel('Coefficient Value')
plt.show()

# Feature importance analysis
def interpret_linear_model(model, feature_names, X, y):
    """Comprehensive linear model interpretation"""
    
    # Coefficients
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_,
        'Abs_Coefficient': np.abs(model.coef_)
    })
    
    # Statistical significance (simplified)
    from scipy import stats
    predictions = model.predict(X)
    residuals = y - predictions
    mse = np.mean(residuals**2)
    
    # Standard errors (simplified calculation)
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
    var_beta = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept).diagonal()
    std_errors = np.sqrt(var_beta[1:])  # Exclude intercept
    
    coef_df['Std_Error'] = std_errors
    coef_df['t_value'] = coef_df['Coefficient'] / coef_df['Std_Error']
    coef_df['p_value'] = 2 * (1 - stats.t.cdf(np.abs(coef_df['t_value']), df=X.shape[0]-X.shape[1]-1))
    
    return coef_df

interpretation = interpret_linear_model(linear_model, feature_names, X, y)
print("\nDetailed Linear Model Interpretation:")
print(interpretation)
```

### Decision Trees

```python
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.datasets import load_iris

# Decision Tree Example
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train decision tree
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)

# Text representation of tree
tree_rules = export_text(tree_model, feature_names=iris.feature_names)
print("Decision Tree Rules:")
print(tree_rules)

# Visualize tree
plt.figure(figsize=(15, 10))
plot_tree(tree_model, 
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,
          rounded=True)
plt.title('Decision Tree Visualization')
plt.show()

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': iris.feature_names,
    'Importance': tree_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Decision Tree Feature Importance:")
print(feature_importance)

# Path explanation for single prediction
def explain_tree_prediction(model, feature_names, instance):
    """Explain a single tree prediction"""
    tree = model.tree_
    feature = tree.feature
    threshold = tree.threshold
    
    def recurse(node, depth=0):
        indent = "  " * depth
        if tree.feature[node] != -2:  # Not a leaf
            name = feature_names[feature[node]]
            threshold_val = threshold[node]
            print(f"{indent}if {name} <= {threshold_val:.2f}:")
            recurse(tree.children_left[node], depth + 1)
            print(f"{indent}else:  # if {name} > {threshold_val:.2f}")
            recurse(tree.children_right[node], depth + 1)
        else:
            print(f"{indent}return class {np.argmax(tree.value[node])}")
    
    print("Decision path for prediction:")
    recurse(0)
    
    # Get actual path for instance
    decision_path = model.decision_path([instance])
    leaf_id = model.apply([instance])
    
    print(f"\nActual path for instance: {instance}")
    print(f"Predicted class: {model.predict([instance])[0]}")
    print(f"Prediction probability: {model.predict_proba([instance])[0]}")

# Explain specific prediction
explain_tree_prediction(tree_model, iris.feature_names, X_test[0])
```

### Rule-Based Models

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

class RuleExtractor:
    """Extract human-readable rules from tree-based models"""
    
    def __init__(self, model, feature_names, class_names):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
    
    def extract_rules(self, max_rules=10):
        """Extract rules from decision tree or random forest"""
        rules = []
        
        if hasattr(self.model, 'tree_'):
            # Single decision tree
            rules.extend(self._extract_tree_rules(self.model.tree_, 0, []))
        elif hasattr(self.model, 'estimators_'):
            # Random forest
            for i, tree in enumerate(self.model.estimators_[:5]):  # Limit to first 5 trees
                tree_rules = self._extract_tree_rules(tree.tree_, 0, [])
                rules.extend(tree_rules)
        
        # Sort rules by confidence/support
        rules.sort(key=lambda x: x['confidence'], reverse=True)
        return rules[:max_rules]
    
    def _extract_tree_rules(self, tree, node, conditions):
        """Recursively extract rules from tree"""
        rules = []
        
        if tree.feature[node] != -2:  # Not a leaf
            feature_name = self.feature_names[tree.feature[node]]
            threshold = tree.threshold[node]
            
            # Left child (<=)
            left_conditions = conditions + [f"{feature_name} <= {threshold:.2f}"]
            rules.extend(self._extract_tree_rules(tree, tree.children_left[node], left_conditions))
            
            # Right child (>)
            right_conditions = conditions + [f"{feature_name} > {threshold:.2f}"]
            rules.extend(self._extract_tree_rules(tree, tree.children_right[node], right_conditions))
        else:
            # Leaf node
            class_counts = tree.value[node][0]
            predicted_class = np.argmax(class_counts)
            confidence = class_counts[predicted_class] / np.sum(class_counts)
            support = np.sum(class_counts)
            
            rule = {
                'conditions': ' AND '.join(conditions),
                'prediction': self.class_names[predicted_class],
                'confidence': confidence,
                'support': support
            }
            rules.append(rule)
        
        return rules

# Extract rules from decision tree
rule_extractor = RuleExtractor(tree_model, iris.feature_names, iris.target_names)
rules = rule_extractor.extract_rules()

print("Extracted Rules:")
for i, rule in enumerate(rules):
    print(f"Rule {i+1}:")
    print(f"  If: {rule['conditions']}")
    print(f"  Then: {rule['prediction']}")
    print(f"  Confidence: {rule['confidence']:.3f}")
    print(f"  Support: {rule['support']}")
    print()
```

## 3. Global Interpretability Methods

### Permutation Feature Importance

```python
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Train a more complex model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Built-in feature importance
builtin_importance = pd.DataFrame({
    'Feature': iris.feature_names,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Built-in Feature Importance:")
print(builtin_importance)

# Permutation importance
perm_importance = permutation_importance(
    rf_model, X_test, y_test, 
    n_repeats=10, random_state=42, 
    scoring='accuracy'
)

perm_importance_df = pd.DataFrame({
    'Feature': iris.feature_names,
    'Importance_Mean': perm_importance.importances_mean,
    'Importance_Std': perm_importance.importances_std
}).sort_values('Importance_Mean', ascending=False)

print("Permutation Feature Importance:")
print(perm_importance_df)

# Compare different importance measures
comparison_df = pd.merge(
    builtin_importance.rename(columns={'Importance': 'Builtin_Importance'}),
    perm_importance_df[['Feature', 'Importance_Mean']].rename(columns={'Importance_Mean': 'Permutation_Importance'}),
    on='Feature'
)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.barh(comparison_df['Feature'], comparison_df['Builtin_Importance'])
plt.title('Built-in Feature Importance')
plt.xlabel('Importance')

plt.subplot(1, 2, 2)
plt.barh(comparison_df['Feature'], comparison_df['Permutation_Importance'])
plt.title('Permutation Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()
```

### Partial Dependence Plots

```python
from sklearn.inspection import partial_dependence, PartialDependenceDisplay

# Partial dependence plots
features = [0, 1, 2, 3]  # All features
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, feature in enumerate(features):
    pd_result = partial_dependence(
        rf_model, X_train, features=[feature], 
        percentiles=(0.05, 0.95), grid_resolution=50
    )
    
    axes[i].plot(pd_result['values'][0], pd_result['average'][0])
    axes[i].set_xlabel(iris.feature_names[feature])
    axes[i].set_ylabel('Partial Dependence')
    axes[i].set_title(f'PDP: {iris.feature_names[feature]}')
    axes[i].grid(True)

plt.tight_layout()
plt.show()

# 2D Partial dependence
display = PartialDependenceDisplay.from_estimator(
    rf_model, X_train, features=[(0, 1), (2, 3)],
    feature_names=iris.feature_names
)
display.plot()
plt.show()
```

### Accumulated Local Effects (ALE)

```python
def ale_plot(model, X, feature_idx, bins=20):
    """Compute Accumulated Local Effects plot"""
    
    # Get feature values and create bins
    feature_values = X[:, feature_idx]
    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.quantile(feature_values, quantiles)
    
    # Calculate local effects
    ale_values = []
    bin_centers = []
    
    for i in range(len(bin_edges) - 1):
        # Find instances in this bin
        lower_bound = bin_edges[i]
        upper_bound = bin_edges[i + 1]
        
        mask = (feature_values >= lower_bound) & (feature_values <= upper_bound)
        if not np.any(mask):
            continue
            
        X_bin = X[mask].copy()
        
        # Calculate local effect
        X_lower = X_bin.copy()
        X_upper = X_bin.copy()
        X_lower[:, feature_idx] = lower_bound
        X_upper[:, feature_idx] = upper_bound
        
        if hasattr(model, 'predict_proba'):
            pred_lower = model.predict_proba(X_lower)[:, 1]  # For binary classification
            pred_upper = model.predict_proba(X_upper)[:, 1]
        else:
            pred_lower = model.predict(X_lower)
            pred_upper = model.predict(X_upper)
        
        local_effect = np.mean(pred_upper - pred_lower)
        ale_values.append(local_effect)
        bin_centers.append((lower_bound + upper_bound) / 2)
    
    # Accumulate effects
    accumulated_effects = np.cumsum([0] + ale_values)
    bin_centers = [bin_edges[0]] + bin_centers
    
    return bin_centers, accumulated_effects

# Example with binary classification
from sklearn.datasets import make_classification

# Create binary classification dataset
X_binary, y_binary = make_classification(
    n_samples=1000, n_features=4, n_redundant=0, 
    n_informative=4, random_state=42
)

# Train binary classifier
binary_model = RandomForestClassifier(n_estimators=100, random_state=42)
binary_model.fit(X_binary, y_binary)

# Create ALE plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i in range(4):
    bin_centers, ale_values = ale_plot(binary_model, X_binary, i)
    axes[i].plot(bin_centers, ale_values, marker='o')
    axes[i].set_xlabel(f'Feature {i}')
    axes[i].set_ylabel('Accumulated Local Effect')
    axes[i].set_title(f'ALE Plot: Feature {i}')
    axes[i].grid(True)

plt.tight_layout()
plt.show()
```

## 4. Local Interpretability Methods

### Individual Conditional Expectation (ICE)

```python
from sklearn.inspection import partial_dependence

def ice_plot(model, X, feature_idx, sample_size=100):
    """Create Individual Conditional Expectation plot"""
    
    # Sample instances
    if len(X) > sample_size:
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    # Get feature range
    feature_values = X[:, feature_idx]
    feature_range = np.linspace(
        np.percentile(feature_values, 5),
        np.percentile(feature_values, 95),
        50
    )
    
    # Calculate ICE curves
    ice_curves = []
    for i in range(len(X_sample)):
        instance = X_sample[i].copy()
        predictions = []
        
        for value in feature_range:
            instance_modified = instance.copy()
            instance_modified[feature_idx] = value
            
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba([instance_modified])[0, 1]
            else:
                pred = model.predict([instance_modified])[0]
            predictions.append(pred)
        
        ice_curves.append(predictions)
    
    return feature_range, ice_curves

# Create ICE plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

feature_names = [f'Feature {i}' for i in range(4)]

for i in range(4):
    feature_range, ice_curves = ice_plot(binary_model, X_binary, i, sample_size=50)
    
    # Plot individual curves
    for curve in ice_curves:
        axes[i].plot(feature_range, curve, alpha=0.3, color='blue', linewidth=0.5)
    
    # Plot average (PDP)
    average_curve = np.mean(ice_curves, axis=0)
    axes[i].plot(feature_range, average_curve, color='red', linewidth=2, label='Average (PDP)')
    
    axes[i].set_xlabel(feature_names[i])
    axes[i].set_ylabel('Prediction')
    axes[i].set_title(f'ICE Plot: {feature_names[i]}')
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
plt.show()
```

### Counterfactual Explanations

```python
class CounterfactualExplainer:
    """Generate counterfactual explanations"""
    
    def __init__(self, model, feature_ranges):
        self.model = model
        self.feature_ranges = feature_ranges
    
    def generate_counterfactual(self, instance, target_class, max_iterations=1000, step_size=0.01):
        """Generate counterfactual explanation using gradient descent"""
        
        instance = instance.copy()
        original_instance = instance.copy()
        
        for iteration in range(max_iterations):
            # Get current prediction
            if hasattr(self.model, 'predict_proba'):
                current_pred = self.model.predict_proba([instance])[0]
                current_class = np.argmax(current_pred)
            else:
                current_pred = self.model.predict([instance])[0]
                current_class = current_pred
            
            # Check if we've reached target class
            if current_class == target_class:
                break
            
            # Simple perturbation strategy
            for feature_idx in range(len(instance)):
                # Try small perturbations
                for direction in [-1, 1]:
                    test_instance = instance.copy()
                    perturbation = direction * step_size * (
                        self.feature_ranges[feature_idx][1] - 
                        self.feature_ranges[feature_idx][0]
                    )
                    test_instance[feature_idx] += perturbation
                    
                    # Clip to feature range
                    test_instance[feature_idx] = np.clip(
                        test_instance[feature_idx],
                        self.feature_ranges[feature_idx][0],
                        self.feature_ranges[feature_idx][1]
                    )
                    
                    # Check if this improves prediction
                    if hasattr(self.model, 'predict_proba'):
                        test_pred = self.model.predict_proba([test_instance])[0]
                        if test_pred[target_class] > current_pred[target_class]:
                            instance = test_instance
                            break
                    else:
                        test_pred = self.model.predict([test_instance])[0]
                        if test_pred == target_class:
                            instance = test_instance
                            break
        
        # Calculate changes
        changes = {}
        for i, (original, modified) in enumerate(zip(original_instance, instance)):
            if abs(original - modified) > 1e-6:
                changes[f'Feature_{i}'] = {
                    'original': original,
                    'counterfactual': modified,
                    'change': modified - original
                }
        
        return instance, changes

# Example usage
feature_ranges = [(X_binary[:, i].min(), X_binary[:, i].max()) for i in range(X_binary.shape[1])]
cf_explainer = CounterfactualExplainer(binary_model, feature_ranges)

# Get a misclassified instance
test_instance = X_binary[0]
original_pred = binary_model.predict([test_instance])[0]
target_class = 1 - original_pred  # Flip the class

counterfactual, changes = cf_explainer.generate_counterfactual(
    test_instance, target_class
)

print(f"Original prediction: {original_pred}")
print(f"Counterfactual prediction: {binary_model.predict([counterfactual])[0]}")
print("Changes needed:")
for feature, change_info in changes.items():
    print(f"  {feature}: {change_info['original']:.3f} → {change_info['counterfactual']:.3f} "
          f"(change: {change_info['change']:.3f})")
```

## 5. Model-Specific Interpretability

### Neural Network Interpretability

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class SimpleNN(nn.Module):
    """Simple neural network for interpretation"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Train neural network
X_tensor = torch.FloatTensor(X_binary)
y_tensor = torch.LongTensor(y_binary)

model_nn = SimpleNN(X_binary.shape[1], 32, 2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_nn.parameters(), lr=0.01)

# Simple training loop
model_nn.train()
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model_nn(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

model_nn.eval()

# Gradient-based feature importance
def gradient_importance(model, instance):
    """Calculate gradient-based feature importance"""
    instance_tensor = torch.FloatTensor(instance).unsqueeze(0)
    instance_tensor.requires_grad_(True)
    
    output = model(instance_tensor)
    predicted_class = torch.argmax(output)
    
    # Backpropagate to get gradients
    output[0, predicted_class].backward()
    
    # Get gradients
    gradients = instance_tensor.grad.squeeze().detach().numpy()
    return gradients

# Calculate importance for test instance
test_instance = X_binary[0]
importance = gradient_importance(model_nn, test_instance)

print("Gradient-based feature importance:")
for i, imp in enumerate(importance):
    print(f"Feature {i}: {imp:.4f}")

# Visualize
plt.figure(figsize=(8, 6))
plt.bar(range(len(importance)), importance)
plt.xlabel('Feature Index')
plt.ylabel('Gradient Importance')
plt.title('Gradient-based Feature Importance')
plt.show()
```

### Attention Mechanisms

```python
class AttentionNN(nn.Module):
    """Neural network with attention for interpretability"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(AttentionNN, self).__init__()
        self.feature_attention = nn.Linear(input_size, input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Calculate attention weights
        attention_weights = torch.softmax(self.feature_attention(x), dim=1)
        
        # Apply attention
        attended_features = x * attention_weights
        
        # Forward pass
        hidden = F.relu(self.fc1(attended_features))
        output = self.fc2(hidden)
        
        return output, attention_weights

# Train attention model
attention_model = AttentionNN(X_binary.shape[1], 32, 2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(attention_model.parameters(), lr=0.01)

attention_model.train()
for epoch in range(100):
    optimizer.zero_grad()
    outputs, _ = attention_model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

attention_model.eval()

# Get attention weights for interpretation
with torch.no_grad():
    test_tensor = torch.FloatTensor(X_binary[:5])
    predictions, attention_weights = attention_model(test_tensor)
    
    print("Attention weights for first 5 instances:")
    for i in range(5):
        print(f"Instance {i}: {attention_weights[i].numpy()}")
```

## 6. Evaluation of Interpretability

### Fidelity Metrics

```python
def fidelity_score(original_model, surrogate_model, X_test):
    """Calculate fidelity between original and surrogate model"""
    original_preds = original_model.predict(X_test)
    surrogate_preds = surrogate_model.predict(X_test)
    
    agreement = np.mean(original_preds == surrogate_preds)
    return agreement

# Example: Compare tree surrogate to random forest
from sklearn.tree import DecisionTreeClassifier

# Train surrogate tree
surrogate_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
surrogate_tree.fit(X_train, rf_model.predict(X_train))

# Calculate fidelity
fidelity = fidelity_score(rf_model, surrogate_tree, X_test)
print(f"Surrogate model fidelity: {fidelity:.3f}")
```

### Stability Metrics

```python
def explanation_stability(explainer, instance, model, n_trials=10):
    """Measure stability of explanations across multiple runs"""
    explanations = []
    
    for _ in range(n_trials):
        if hasattr(explainer, 'explain_instance'):
            # LIME-style explainer
            exp = explainer.explain_instance(
                instance, model.predict_proba, num_samples=500
            )
            exp_dict = dict(exp.as_list())
        else:
            # Custom explainer
            exp_dict = explainer.explain(instance)
        
        explanations.append(exp_dict)
    
    # Calculate stability as correlation between explanations
    features = list(explanations[0].keys())
    correlations = []
    
    for i in range(len(explanations)):
        for j in range(i+1, len(explanations)):
            values_i = [explanations[i].get(f, 0) for f in features]
            values_j = [explanations[j].get(f, 0) for f in features]
            corr = np.corrcoef(values_i, values_j)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
    
    return np.mean(correlations) if correlations else 0

# Example stability measurement (would need LIME installed)
# stability = explanation_stability(explainer, X_test[0], rf_model)
# print(f"Explanation stability: {stability:.3f}")
```

## 7. Production Interpretability Systems

### Model Interpretability Pipeline

```python
class InterpretabilityPipeline:
    """Comprehensive interpretability pipeline"""
    
    def __init__(self, model, X_train, feature_names=None):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
        self.global_explanations = {}
        self.local_explainers = {}
    
    def compute_global_interpretability(self):
        """Compute global interpretability metrics"""
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.global_explanations['feature_importance'] = {
                'features': self.feature_names,
                'importance': self.model.feature_importances_
            }
        
        # Permutation importance
        from sklearn.inspection import permutation_importance
        perm_imp = permutation_importance(
            self.model, self.X_train, self.model.predict(self.X_train),
            n_repeats=5, random_state=42
        )
        
        self.global_explanations['permutation_importance'] = {
            'features': self.feature_names,
            'importance': perm_imp.importances_mean,
            'std': perm_imp.importances_std
        }
        
        return self.global_explanations
    
    def setup_local_explainers(self):
        """Setup local explanation methods"""
        try:
            import lime.lime_tabular
            self.local_explainers['lime'] = lime.lime_tabular.LimeTabularExplainer(
                self.X_train,
                feature_names=self.feature_names,
                mode='classification'
            )
        except ImportError:
            print("LIME not available")
        
        try:
            import shap
            if hasattr(self.model, 'predict_proba'):
                self.local_explainers['shap'] = shap.Explainer(self.model.predict_proba, self.X_train)
        except ImportError:
            print("SHAP not available")
    
    def explain_instance(self, instance, methods=['permutation']):
        """Explain a single instance using specified methods"""
        explanations = {}
        
        if 'permutation' in methods:
            explanations['permutation'] = self._permutation_explanation(instance)
        
        if 'lime' in methods and 'lime' in self.local_explainers:
            exp = self.local_explainers['lime'].explain_instance(
                instance, self.model.predict_proba, num_features=len(self.feature_names)
            )
            explanations['lime'] = dict(exp.as_list())
        
        if 'shap' in methods and 'shap' in self.local_explainers:
            shap_values = self.local_explainers['shap']([instance])
            explanations['shap'] = dict(zip(self.feature_names, shap_values.values[0]))
        
        return explanations
    
    def _permutation_explanation(self, instance):
        """Simple permutation-based local explanation"""
        baseline_pred = self.model.predict_proba([instance])[0]
        explanations = {}
        
        for i, feature_name in enumerate(self.feature_names):
            # Permute feature
            perturbed_instance = instance.copy()
            # Use random value from training set
            random_value = np.random.choice(self.X_train[:, i])
            perturbed_instance[i] = random_value
            
            perturbed_pred = self.model.predict_proba([perturbed_instance])[0]
            importance = np.max(baseline_pred) - np.max(perturbed_pred)
            explanations[feature_name] = importance
        
        return explanations
    
    def generate_report(self, instances=None, output_file='interpretability_report.html'):
        """Generate comprehensive interpretability report"""
        
        report = []
        report.append("<html><head><title>Model Interpretability Report</title></head><body>")
        report.append("<h1>Model Interpretability Report</h1>")
        
        # Global explanations
        report.append("<h2>Global Model Interpretability</h2>")
        if 'feature_importance' in self.global_explanations:
            report.append("<h3>Feature Importance</h3>")
            imp_data = self.global_explanations['feature_importance']
            for feature, importance in zip(imp_data['features'], imp_data['importance']):
                report.append(f"<p>{feature}: {importance:.4f}</p>")
        
        # Local explanations
        if instances is not None:
            report.append("<h2>Local Explanations</h2>")
            for i, instance in enumerate(instances[:5]):  # Limit to 5 instances
                report.append(f"<h3>Instance {i+1}</h3>")
                explanations = self.explain_instance(instance)
                
                for method, explanation in explanations.items():
                    report.append(f"<h4>{method.upper()}</h4>")
                    if isinstance(explanation, dict):
                        for feature, importance in explanation.items():
                            report.append(f"<p>{feature}: {importance:.4f}</p>")
        
        report.append("</body></html>")
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Report saved to {output_file}")

# Usage example
pipeline = InterpretabilityPipeline(rf_model, X_train, iris.feature_names)
global_explanations = pipeline.compute_global_interpretability()
pipeline.setup_local_explainers()

# Generate explanations for test instances
for i in range(3):
    explanations = pipeline.explain_instance(X_test[i])
    print(f"Instance {i} explanations:")
    for method, explanation in explanations.items():
        print(f"  {method}: {explanation}")
    print()
```

## 8. Best Practices and Guidelines

### Choosing Interpretability Methods

```python
def choose_interpretability_method(model_type, data_type, use_case):
    """Recommend interpretability methods based on context"""
    
    recommendations = {
        'methods': [],
        'rationale': []
    }
    
    # Model-specific recommendations
    if model_type in ['linear_regression', 'logistic_regression']:
        recommendations['methods'].append('coefficient_analysis')
        recommendations['rationale'].append('Linear models have inherent interpretability through coefficients')
    
    elif model_type in ['decision_tree', 'random_forest']:
        recommendations['methods'].extend(['feature_importance', 'tree_visualization'])
        recommendations['rationale'].append('Tree-based models provide natural feature importance and rules')
    
    elif model_type in ['neural_network', 'deep_learning']:
        recommendations['methods'].extend(['gradient_methods', 'attention_weights'])
        recommendations['rationale'].append('Neural networks require specialized gradient-based methods')
    
    else:  # Black-box models
        recommendations['methods'].extend(['lime', 'shap', 'permutation_importance'])
        recommendations['rationale'].append('Black-box models need model-agnostic explanation methods')
    
    # Use case considerations
    if use_case == 'debugging':
        recommendations['methods'].extend(['ice_plots', 'partial_dependence'])
        recommendations['rationale'].append('Debugging requires understanding feature effects and interactions')
    
    elif use_case == 'compliance':
        recommendations['methods'].extend(['counterfactuals', 'feature_attribution'])
        recommendations['rationale'].append('Compliance requires explainable individual decisions')
    
    elif use_case == 'trust_building':
        recommendations['methods'].extend(['global_surrogate', 'rule_extraction'])
        recommendations['rationale'].append('Trust building requires understandable model approximations')
    
    # Data type considerations
    if data_type == 'tabular':
        recommendations['methods'].extend(['feature_importance', 'partial_dependence'])
    elif data_type == 'text':
        recommendations['methods'].extend(['attention_weights', 'word_importance'])
    elif data_type == 'image':
        recommendations['methods'].extend(['saliency_maps', 'grad_cam'])
    
    return recommendations

# Example usage
recommendations = choose_interpretability_method(
    model_type='random_forest',
    data_type='tabular',
    use_case='debugging'
)

print("Recommended interpretability methods:")
for method in recommendations['methods']:
    print(f"- {method}")

print("\nRationale:")
for rationale in recommendations['rationale']:
    print(f"- {rationale}")
```

### Interpretability Checklist

```python
class InterpretabilityChecklist:
    """Checklist for ensuring comprehensive model interpretability"""
    
    def __init__(self):
        self.checklist = {
            'global_understanding': [
                'Feature importance computed',
                'Model behavior understood across feature ranges',
                'Feature interactions identified',
                'Model assumptions validated'
            ],
            'local_understanding': [
                'Individual predictions explainable',
                'Counterfactual explanations available',
                'Local feature importance computed',
                'Prediction confidence assessed'
            ],
            'validation': [
                'Explanation fidelity measured',
                'Explanation stability assessed',
                'Human evaluation conducted',
                'Explanation consistency verified'
            ],
            'documentation': [
                'Interpretability methods documented',
                'Limitations clearly stated',
                'Use cases defined',
                'Maintenance procedures established'
            ],
            'production_readiness': [
                'Real-time explanation capability',
                'Scalable explanation generation',
                'Monitoring for explanation drift',
                'User interface for explanations'
            ]
        }
    
    def evaluate_model(self, model_info):
        """Evaluate model against interpretability checklist"""
        results = {}
        
        for category, items in self.checklist.items():
            category_score = 0
            category_results = {}
            
            for item in items:
                # This would normally check actual implementation
                # For demo, we'll use model_info
                completed = item.lower().replace(' ', '_') in model_info.get(category, [])
                category_results[item] = completed
                if completed:
                    category_score += 1
            
            results[category] = {
                'items': category_results,
                'score': category_score / len(items),
                'total_items': len(items)
            }
        
        return results
    
    def generate_report(self, evaluation_results):
        """Generate interpretability assessment report"""
        print("=== Model Interpretability Assessment ===\n")
        
        overall_score = 0
        total_categories = len(evaluation_results)
        
        for category, results in evaluation_results.items():
            score = results['score']
            overall_score += score
            
            print(f"{category.replace('_', ' ').title()}: {score:.1%}")
            print(f"  Completed: {sum(results['items'].values())}/{results['total_items']}")
            
            # Show incomplete items
            incomplete = [item for item, completed in results['items'].items() if not completed]
            if incomplete:
                print("  Missing:")
                for item in incomplete:
                    print(f"    - {item}")
            print()
        
        overall_score /= total_categories
        print(f"Overall Interpretability Score: {overall_score:.1%}")
        
        if overall_score < 0.7:
            print("\n⚠️  Warning: Model interpretability needs improvement")
        elif overall_score < 0.9:
            print("\n✅ Good: Model has adequate interpretability")
        else:
            print("\n🌟 Excellent: Model has comprehensive interpretability")

# Example usage
checklist = InterpretabilityChecklist()

# Mock model information
model_info = {
    'global_understanding': ['feature_importance_computed', 'model_behavior_understood'],
    'local_understanding': ['individual_predictions_explainable'],
    'validation': ['explanation_fidelity_measured'],
    'documentation': ['interpretability_methods_documented', 'limitations_clearly_stated'],
    'production_readiness': ['real_time_explanation_capability']
}

evaluation = checklist.evaluate_model(model_info)
checklist.generate_report(evaluation)
```

## Summary

Model interpretability is crucial for building trustworthy and accountable AI systems. Key approaches include:

### Intrinsic Interpretability
- **Linear Models**: Direct coefficient interpretation
- **Decision Trees**: Rule-based explanations
- **Rule-Based Models**: Human-readable if-then rules

### Post-hoc Methods
- **Global**: Feature importance, partial dependence, surrogate models
- **Local**: LIME, SHAP, counterfactuals, ICE plots
- **Model-Specific**: Attention weights, gradient methods

### Best Practices
1. **Choose appropriate methods** based on model type and use case
2. **Validate explanations** for fidelity and stability
3. **Document limitations** and assumptions
4. **Design for production** with scalable explanation systems
5. **Consider human factors** in explanation design

### Key Considerations
- **Trade-offs**: Accuracy vs interpretability
- **Context dependency**: Different stakeholders need different explanations
- **Validation**: Explanations should be tested and validated
- **Maintenance**: Interpretability systems need ongoing monitoring

Model interpretability is not one-size-fits-all - the choice of methods should align with specific requirements, constraints, and stakeholder needs.