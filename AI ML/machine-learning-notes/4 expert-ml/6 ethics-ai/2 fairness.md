# Fairness in Machine Learning

## Learning Objectives
- Understand different definitions and types of fairness in ML
- Learn fairness metrics and measurement techniques
- Implement fairness-aware algorithms and mitigation strategies
- Apply fairness constraints in model training and evaluation
- Design fair ML systems for production environments

## 1. Introduction to ML Fairness

### What is Fairness?
Fairness in machine learning refers to the absence of bias or discrimination against individuals or groups based on sensitive attributes like race, gender, age, or religion. However, fairness is not a single concept but encompasses multiple definitions that can sometimes conflict.

### Types of Fairness

#### Individual Fairness
- **Definition**: Similar individuals should receive similar treatment
- **Principle**: "Treat similar people similarly"
- **Challenge**: Defining similarity metrics

#### Group Fairness
- **Definition**: Statistical parity across different groups
- **Principle**: Equal outcomes for different demographic groups
- **Types**: Statistical parity, equal opportunity, equalized odds

#### Counterfactual Fairness
- **Definition**: Decisions should be the same in a counterfactual world
- **Principle**: Outcomes shouldn't change if sensitive attributes were different
- **Application**: "What would happen if this person were a different race/gender?"

## 2. Fairness Metrics

### Implementation of Key Fairness Metrics

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class FairnessMetrics:
    """Comprehensive fairness metrics implementation"""
    
    def __init__(self, y_true, y_pred, y_prob, sensitive_attr):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.sensitive_attr = sensitive_attr
        self.groups = np.unique(sensitive_attr)
    
    def statistical_parity(self):
        """Statistical Parity: P(Y_hat = 1 | A = a) = P(Y_hat = 1 | A = b)"""
        group_rates = {}
        for group in self.groups:
            mask = self.sensitive_attr == group
            positive_rate = np.mean(self.y_pred[mask])
            group_rates[group] = positive_rate
        
        # Calculate difference and ratio
        rates = list(group_rates.values())
        sp_difference = max(rates) - min(rates)
        sp_ratio = min(rates) / max(rates) if max(rates) > 0 else 0
        
        return {
            'group_rates': group_rates,
            'statistical_parity_difference': sp_difference,
            'disparate_impact_ratio': sp_ratio
        }
    
    def equal_opportunity(self):
        """Equal Opportunity: P(Y_hat = 1 | Y = 1, A = a) = P(Y_hat = 1 | Y = 1, A = b)"""
        group_tpr = {}
        for group in self.groups:
            mask = (self.sensitive_attr == group) & (self.y_true == 1)
            if np.sum(mask) > 0:
                tpr = np.mean(self.y_pred[mask])
                group_tpr[group] = tpr
            else:
                group_tpr[group] = 0
        
        tpr_values = list(group_tpr.values())
        eo_difference = max(tpr_values) - min(tpr_values)
        
        return {
            'group_tpr': group_tpr,
            'equal_opportunity_difference': eo_difference
        }
    
    def equalized_odds(self):
        """Equalized Odds: Equal TPR and FPR across groups"""
        group_metrics = {}
        
        for group in self.groups:
            group_mask = self.sensitive_attr == group
            
            # True Positive Rate
            tp_mask = group_mask & (self.y_true == 1)
            tpr = np.mean(self.y_pred[tp_mask]) if np.sum(tp_mask) > 0 else 0
            
            # False Positive Rate
            tn_mask = group_mask & (self.y_true == 0)
            fpr = np.mean(self.y_pred[tn_mask]) if np.sum(tn_mask) > 0 else 0
            
            group_metrics[group] = {'tpr': tpr, 'fpr': fpr}
        
        # Calculate differences
        tpr_values = [metrics['tpr'] for metrics in group_metrics.values()]
        fpr_values = [metrics['fpr'] for metrics in group_metrics.values()]
        
        tpr_difference = max(tpr_values) - min(tpr_values)
        fpr_difference = max(fpr_values) - min(fpr_values)
        
        return {
            'group_metrics': group_metrics,
            'tpr_difference': tpr_difference,
            'fpr_difference': fpr_difference,
            'max_equalized_odds_difference': max(tpr_difference, fpr_difference)
        }
    
    def calibration(self):
        """Calibration: P(Y = 1 | Y_hat = s, A = a) = P(Y = 1 | Y_hat = s, A = b)"""
        calibration_metrics = {}
        
        # Bin predictions
        bins = np.linspace(0, 1, 11)
        
        for group in self.groups:
            group_mask = self.sensitive_attr == group
            group_probs = self.y_prob[group_mask]
            group_true = self.y_true[group_mask]
            
            bin_calibration = []
            for i in range(len(bins) - 1):
                bin_mask = (group_probs >= bins[i]) & (group_probs < bins[i + 1])
                if np.sum(bin_mask) > 0:
                    actual_rate = np.mean(group_true[bin_mask])
                    predicted_rate = np.mean(group_probs[bin_mask])
                    bin_calibration.append({
                        'bin_range': (bins[i], bins[i + 1]),
                        'actual_rate': actual_rate,
                        'predicted_rate': predicted_rate,
                        'calibration_error': abs(actual_rate - predicted_rate)
                    })
            
            calibration_metrics[group] = bin_calibration
        
        return calibration_metrics
    
    def individual_fairness_metric(self, distance_function=None):
        """Individual Fairness: Similar individuals get similar predictions"""
        if distance_function is None:
            # Use L2 distance as default (simplified)
            distance_function = lambda x, y: np.linalg.norm(x - y)
        
        # For demonstration, calculate prediction consistency
        # In practice, this requires feature similarity calculation
        consistency_scores = []
        
        for group in self.groups:
            group_mask = self.sensitive_attr == group
            group_probs = self.y_prob[group_mask]
            
            if len(group_probs) > 1:
                # Calculate variance within group (lower is better)
                consistency = 1 / (1 + np.var(group_probs))
                consistency_scores.append(consistency)
        
        return {
            'group_consistency': dict(zip(self.groups, consistency_scores)),
            'overall_consistency': np.mean(consistency_scores) if consistency_scores else 0
        }
    
    def comprehensive_report(self):
        """Generate comprehensive fairness report"""
        sp = self.statistical_parity()
        eo = self.equal_opportunity()
        eqo = self.equalized_odds()
        cal = self.calibration()
        ind = self.individual_fairness_metric()
        
        return {
            'statistical_parity': sp,
            'equal_opportunity': eo,
            'equalized_odds': eqo,
            'calibration': cal,
            'individual_fairness': ind
        }

# Create example dataset with bias
def create_fair_demo_dataset(n_samples=5000):
    """Create dataset for fairness demonstration"""
    X, y = make_classification(
        n_samples=n_samples, n_features=8, n_informative=5,
        n_redundant=1, n_clusters_per_class=1, random_state=42
    )
    
    # Add sensitive attribute (gender)
    np.random.seed(42)
    gender = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.6, 0.4])
    
    # Introduce bias
    bias_factor = np.where(gender == 'Female', 0.3, 0)
    biased_prob = 1 / (1 + np.exp(-(X[:, 0] + bias_factor)))
    y_biased = (np.random.random(n_samples) < biased_prob).astype(int)
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['gender'] = gender
    df['target'] = y_biased
    
    return df

# Demonstrate fairness metrics
data = create_fair_demo_dataset()
X = data.drop(['target', 'gender'], axis=1)
y = data['target']
gender = data['gender']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
gender_train, gender_test = train_test_split(gender, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Get predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate fairness metrics
fairness_calc = FairnessMetrics(
    y_true=y_test.values,
    y_pred=y_pred,
    y_prob=y_prob,
    sensitive_attr=gender_test.values
)

# Generate comprehensive report
fairness_report = fairness_calc.comprehensive_report()

print("=== FAIRNESS METRICS REPORT ===")
print(f"\nStatistical Parity:")
sp = fairness_report['statistical_parity']
for group, rate in sp['group_rates'].items():
    print(f"  {group}: {rate:.3f}")
print(f"  Difference: {sp['statistical_parity_difference']:.3f}")
print(f"  Disparate Impact Ratio: {sp['disparate_impact_ratio']:.3f}")

print(f"\nEqual Opportunity:")
eo = fairness_report['equal_opportunity']
for group, tpr in eo['group_tpr'].items():
    print(f"  {group} TPR: {tpr:.3f}")
print(f"  Difference: {eo['equal_opportunity_difference']:.3f}")

print(f"\nEqualized Odds:")
eqo = fairness_report['equalized_odds']
for group, metrics in eqo['group_metrics'].items():
    print(f"  {group}: TPR={metrics['tpr']:.3f}, FPR={metrics['fpr']:.3f}")
print(f"  Max Difference: {eqo['max_equalized_odds_difference']:.3f}")
```

## 3. Fairness-Aware Algorithms

### Pre-processing Methods

```python
class FairPreprocessing:
    """Pre-processing methods for fairness"""
    
    def __init__(self, sensitive_attr):
        self.sensitive_attr = sensitive_attr
    
    def reweighting(self, X, y, sensitive_col):
        """Reweight training samples to achieve fairness"""
        weights = np.ones(len(X))
        
        # Calculate weights to balance representation
        for group in np.unique(X[sensitive_col]):
            group_mask = X[sensitive_col] == group
            
            for label in [0, 1]:
                label_mask = y == label
                combined_mask = group_mask & label_mask
                
                if np.sum(combined_mask) > 0:
                    # Weight inversely proportional to frequency
                    group_label_freq = np.sum(combined_mask) / len(X)
                    overall_label_freq = np.sum(label_mask) / len(X)
                    
                    weight = overall_label_freq / (group_label_freq + 1e-8)
                    weights[combined_mask] = weight
        
        return weights
    
    def disparate_impact_remover(self, X, sensitive_col, repair_level=1.0):
        """Remove disparate impact from features"""
        X_repaired = X.copy()
        
        for col in X.columns:
            if col != sensitive_col:
                # Calculate repair amount
                overall_mean = X[col].mean()
                
                for group in X[sensitive_col].unique():
                    group_mask = X[sensitive_col] == group
                    group_mean = X.loc[group_mask, col].mean()
                    
                    # Repair towards overall mean
                    repair_amount = repair_level * (overall_mean - group_mean)
                    X_repaired.loc[group_mask, col] += repair_amount
        
        return X_repaired

# Demonstrate pre-processing
preprocessor = FairPreprocessing(sensitive_attr='gender')

# Reweighting
sample_weights = preprocessor.reweighting(
    X=data[['gender'] + [col for col in data.columns if col not in ['target', 'gender']]],
    y=data['target'],
    sensitive_col='gender'
)

print(f"\nSample weights distribution:")
for gender in ['Male', 'Female']:
    mask = data['gender'] == gender
    avg_weight = np.mean(sample_weights[mask])
    print(f"  {gender}: {avg_weight:.3f}")

# Disparate impact removal
X_repaired = preprocessor.disparate_impact_remover(
    X=data[['gender'] + [col for col in data.columns if col not in ['target']]],
    sensitive_col='gender',
    repair_level=0.5
)

print(f"\nFeature means before/after repair:")
for col in ['feature_0', 'feature_1']:
    print(f"\n{col}:")
    for gender in ['Male', 'Female']:
        mask = data['gender'] == gender
        before = data.loc[mask, col].mean()
        after = X_repaired.loc[mask, col].mean()
        print(f"  {gender}: {before:.3f} → {after:.3f}")
```

### In-processing Methods

```python
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array

class FairLogisticRegression(BaseEstimator, ClassifierMixin):
    """Logistic regression with fairness constraints"""
    
    def __init__(self, fairness_penalty=1.0, max_iter=1000, learning_rate=0.01):
        self.fairness_penalty = fairness_penalty
        self.max_iter = max_iter
        self.learning_rate = learning_rate
    
    def fit(self, X, y, sensitive_features):
        """Fit model with fairness constraints"""
        X, y = check_X_y(X, y)
        n_features = X.shape[1]
        
        # Initialize weights
        self.coef_ = np.random.normal(0, 0.01, n_features)
        self.intercept_ = 0
        
        # Gradient descent with fairness penalty
        for iteration in range(self.max_iter):
            # Predictions
            z = X @ self.coef_ + self.intercept_
            predictions = 1 / (1 + np.exp(-z))
            
            # Standard logistic loss gradient
            loss_grad_coef = X.T @ (predictions - y) / len(X)
            loss_grad_intercept = np.mean(predictions - y)
            
            # Fairness penalty gradient
            fairness_grad_coef, fairness_grad_intercept = self._fairness_gradient(
                X, y, predictions, sensitive_features
            )
            
            # Update weights
            self.coef_ -= self.learning_rate * (
                loss_grad_coef + self.fairness_penalty * fairness_grad_coef
            )
            self.intercept_ -= self.learning_rate * (
                loss_grad_intercept + self.fairness_penalty * fairness_grad_intercept
            )
        
        return self
    
    def _fairness_gradient(self, X, y, predictions, sensitive_features):
        """Calculate gradient of fairness penalty"""
        # Simplified fairness penalty based on statistical parity
        fairness_grad_coef = np.zeros_like(self.coef_)
        fairness_grad_intercept = 0
        
        groups = np.unique(sensitive_features)
        if len(groups) >= 2:
            # Calculate group-wise prediction rates
            group_rates = []
            group_masks = []
            
            for group in groups:
                mask = sensitive_features == group
                group_masks.append(mask)
                if np.sum(mask) > 0:
                    group_rate = np.mean(predictions[mask])
                    group_rates.append(group_rate)
                else:
                    group_rates.append(0)
            
            # Penalty for difference in group rates
            if len(group_rates) >= 2:
                rate_diff = group_rates[0] - group_rates[1]
                
                # Gradient of penalty w.r.t. coefficients
                for i, (mask1, mask2) in enumerate([(group_masks[0], group_masks[1])]):
                    if np.sum(mask1) > 0 and np.sum(mask2) > 0:
                        grad_contribution = 2 * rate_diff * (
                            np.mean(X[mask1] * predictions[mask1].reshape(-1, 1) * (1 - predictions[mask1]).reshape(-1, 1), axis=0) / np.sum(mask1) -
                            np.mean(X[mask2] * predictions[mask2].reshape(-1, 1) * (1 - predictions[mask2]).reshape(-1, 1), axis=0) / np.sum(mask2)
                        )
                        fairness_grad_coef += grad_contribution
        
        return fairness_grad_coef, fairness_grad_intercept
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        X = check_array(X)
        z = X @ self.coef_ + self.intercept_
        prob_1 = 1 / (1 + np.exp(-z))
        prob_0 = 1 - prob_1
        return np.column_stack([prob_0, prob_1])
    
    def predict(self, X):
        """Predict class labels"""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# Demonstrate fair classifier
print("\n=== FAIR CLASSIFIER COMPARISON ===")

# Prepare data
X_train_fair = X_train.values
X_test_fair = X_test.values
gender_train_encoded = (gender_train == 'Female').astype(int).values
gender_test_encoded = (gender_test == 'Female').astype(int).values

# Train regular classifier
regular_model = LogisticRegression(random_state=42, max_iter=1000)
regular_model.fit(X_train_fair, y_train)

# Train fair classifier
fair_model = FairLogisticRegression(fairness_penalty=2.0, max_iter=500)
fair_model.fit(X_train_fair, y_train.values, gender_train_encoded)

# Compare fairness metrics
models = {
    'Regular': regular_model,
    'Fair': fair_model
}

for model_name, model in models.items():
    print(f"\n{model_name} Model:")
    
    y_pred_model = model.predict(X_test_fair)
    y_prob_model = model.predict_proba(X_test_fair)[:, 1]
    
    fairness_calc_model = FairnessMetrics(
        y_true=y_test.values,
        y_pred=y_pred_model,
        y_prob=y_prob_model,
        sensitive_attr=gender_test.values
    )
    
    sp_model = fairness_calc_model.statistical_parity()
    eo_model = fairness_calc_model.equal_opportunity()
    
    print(f"  Statistical Parity Difference: {sp_model['statistical_parity_difference']:.3f}")
    print(f"  Disparate Impact Ratio: {sp_model['disparate_impact_ratio']:.3f}")
    print(f"  Equal Opportunity Difference: {eo_model['equal_opportunity_difference']:.3f}")
```

## 4. Post-processing Methods

### Calibration and Threshold Optimization

```python
class PostProcessingFairness:
    """Post-processing methods for fairness"""
    
    def __init__(self):
        self.group_thresholds = {}
        self.calibration_params = {}
    
    def equalized_odds_postprocessing(self, y_true, y_prob, sensitive_attr, target_tpr=None, target_fpr=None):
        """Post-process to achieve equalized odds"""
        groups = np.unique(sensitive_attr)
        optimal_thresholds = {}
        
        if target_tpr is None:
            # Use average TPR as target
            overall_tpr = np.mean(y_prob[y_true == 1])
            target_tpr = overall_tpr
        
        if target_fpr is None:
            # Use average FPR as target
            overall_fpr = np.mean(y_prob[y_true == 0])
            target_fpr = overall_fpr
        
        for group in groups:
            group_mask = sensitive_attr == group
            group_y_true = y_true[group_mask]
            group_y_prob = y_prob[group_mask]
            
            # Find threshold that achieves target TPR and FPR
            best_threshold = 0.5
            best_score = float('inf')
            
            for threshold in np.arange(0.01, 1.0, 0.01):
                group_y_pred = (group_y_prob >= threshold).astype(int)
                
                if np.sum(group_y_true == 1) > 0:
                    tpr = np.mean(group_y_pred[group_y_true == 1])
                else:
                    tpr = 0
                
                if np.sum(group_y_true == 0) > 0:
                    fpr = np.mean(group_y_pred[group_y_true == 0])
                else:
                    fpr = 0
                
                # Score based on distance to target TPR and FPR
                score = abs(tpr - target_tpr) + abs(fpr - target_fpr)
                
                if score < best_score:
                    best_score = score
                    best_threshold = threshold
            
            optimal_thresholds[group] = best_threshold
        
        self.group_thresholds = optimal_thresholds
        return optimal_thresholds
    
    def calibration_postprocessing(self, y_true, y_prob, sensitive_attr):
        """Post-process for calibration fairness"""
        groups = np.unique(sensitive_attr)
        calibration_params = {}
        
        for group in groups:
            group_mask = sensitive_attr == group
            group_y_true = y_true[group_mask]
            group_y_prob = y_prob[group_mask]
            
            # Fit calibration using Platt scaling (simplified)
            # In practice, use sklearn.calibration.CalibratedClassifierCV
            from sklearn.linear_model import LogisticRegression
            
            if len(group_y_prob) > 10:  # Minimum samples for calibration
                calibrator = LogisticRegression()
                calibrator.fit(group_y_prob.reshape(-1, 1), group_y_true)
                calibration_params[group] = calibrator
            else:
                calibration_params[group] = None
        
        self.calibration_params = calibration_params
        return calibration_params
    
    def apply_thresholds(self, y_prob, sensitive_attr):
        """Apply group-specific thresholds"""
        y_pred_fair = np.zeros_like(y_prob)
        
        for group, threshold in self.group_thresholds.items():
            group_mask = sensitive_attr == group
            y_pred_fair[group_mask] = (y_prob[group_mask] >= threshold).astype(int)
        
        return y_pred_fair
    
    def apply_calibration(self, y_prob, sensitive_attr):
        """Apply group-specific calibration"""
        y_prob_calibrated = y_prob.copy()
        
        for group, calibrator in self.calibration_params.items():
            if calibrator is not None:
                group_mask = sensitive_attr == group
                if np.sum(group_mask) > 0:
                    calibrated_probs = calibrator.predict_proba(
                        y_prob[group_mask].reshape(-1, 1)
                    )[:, 1]
                    y_prob_calibrated[group_mask] = calibrated_probs
        
        return y_prob_calibrated

# Demonstrate post-processing
print("\n=== POST-PROCESSING FAIRNESS ===")

# Use regular model predictions
y_prob_regular = regular_model.predict_proba(X_test_fair)[:, 1]

post_processor = PostProcessingFairness()

# Equalized odds post-processing
optimal_thresholds = post_processor.equalized_odds_postprocessing(
    y_true=y_test.values,
    y_prob=y_prob_regular,
    sensitive_attr=gender_test.values
)

print("Optimal thresholds for equalized odds:")
for group, threshold in optimal_thresholds.items():
    print(f"  {group}: {threshold:.3f}")

# Apply fair thresholds
y_pred_postprocessed = post_processor.apply_thresholds(
    y_prob=y_prob_regular,
    sensitive_attr=gender_test.values
)

# Compare fairness before and after post-processing
print("\nBefore post-processing:")
fairness_before = FairnessMetrics(
    y_true=y_test.values,
    y_pred=regular_model.predict(X_test_fair),
    y_prob=y_prob_regular,
    sensitive_attr=gender_test.values
)
eqo_before = fairness_before.equalized_odds()
print(f"  Max Equalized Odds Difference: {eqo_before['max_equalized_odds_difference']:.3f}")

print("\nAfter post-processing:")
fairness_after = FairnessMetrics(
    y_true=y_test.values,
    y_pred=y_pred_postprocessed,
    y_prob=y_prob_regular,
    sensitive_attr=gender_test.values
)
eqo_after = fairness_after.equalized_odds()
print(f"  Max Equalized Odds Difference: {eqo_after['max_equalized_odds_difference']:.3f}")
```

## 5. Fairness-Accuracy Trade-offs

### Pareto Frontier Analysis

```python
def fairness_accuracy_tradeoff(X_train, y_train, X_test, y_test, sensitive_test, fairness_penalties):
    """Analyze fairness-accuracy trade-offs"""
    
    results = []
    
    for penalty in fairness_penalties:
        # Train fair model with different penalty values
        fair_model = FairLogisticRegression(fairness_penalty=penalty, max_iter=300)
        
        # Convert sensitive features for training
        sensitive_train_binary = np.random.choice([0, 1], size=len(X_train))  # Simplified
        
        fair_model.fit(X_train, y_train, sensitive_train_binary)
        
        # Evaluate
        y_pred = fair_model.predict(X_test)
        y_prob = fair_model.predict_proba(X_test)[:, 1]
        
        accuracy = np.mean(y_pred == y_test)
        
        fairness_calc = FairnessMetrics(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            sensitive_attr=sensitive_test
        )
        
        sp = fairness_calc.statistical_parity()
        
        results.append({
            'penalty': penalty,
            'accuracy': accuracy,
            'fairness_violation': sp['statistical_parity_difference'],
            'disparate_impact': sp['disparate_impact_ratio']
        })
    
    return results

# Analyze trade-offs
penalties = [0, 0.5, 1.0, 2.0, 5.0, 10.0]
tradeoff_results = fairness_accuracy_tradeoff(
    X_train=X_train.values,
    y_train=y_train.values,
    X_test=X_test.values,
    y_test=y_test.values,
    sensitive_test=gender_test.values,
    fairness_penalties=penalties
)

print("\n=== FAIRNESS-ACCURACY TRADE-OFF ===")
for result in tradeoff_results:
    print(f"Penalty: {result['penalty']:>4.1f}, "
          f"Accuracy: {result['accuracy']:.3f}, "
          f"Fairness Violation: {result['fairness_violation']:.3f}")

# Visualize trade-off
plt.figure(figsize=(10, 6))
accuracies = [r['accuracy'] for r in tradeoff_results]
fairness_violations = [r['fairness_violation'] for r in tradeoff_results]

plt.subplot(1, 2, 1)
plt.plot(penalties, accuracies, 'b-o', label='Accuracy')
plt.xlabel('Fairness Penalty')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Fairness Penalty')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(fairness_violations, accuracies, 'r-o')
plt.xlabel('Fairness Violation (Statistical Parity Difference)')
plt.ylabel('Accuracy')
plt.title('Pareto Frontier: Accuracy vs Fairness')
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Summary

Fairness in machine learning requires:

### Key Concepts
1. **Multiple Definitions**: Statistical parity, equal opportunity, equalized odds
2. **Individual vs Group**: Different fairness paradigms
3. **Measurement**: Comprehensive metrics and evaluation frameworks
4. **Trade-offs**: Balancing fairness and accuracy

### Implementation Approaches
- **Pre-processing**: Data reweighting, disparate impact removal
- **In-processing**: Fairness-aware algorithms with constraints
- **Post-processing**: Threshold optimization, calibration

### Best Practices
- Define fairness criteria early in the project
- Use multiple fairness metrics for comprehensive evaluation
- Consider stakeholder perspectives and domain requirements
- Monitor fairness continuously in production systems
- Document fairness decisions and trade-offs

Fairness is context-dependent and requires careful consideration of stakeholder needs, legal requirements, and ethical implications.