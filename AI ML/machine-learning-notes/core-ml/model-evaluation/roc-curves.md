# ROC Curves

## Overview
ROC (Receiver Operating Characteristic) curves are fundamental tools for evaluating binary classification models. They visualize the trade-off between sensitivity (true positive rate) and specificity (false positive rate) across all classification thresholds, providing insights into model performance independent of class distribution.

## What are ROC Curves?
- **Definition**: Plot of True Positive Rate (TPR) vs False Positive Rate (FPR) at various threshold settings
- **Origin**: Developed during WWII for radar signal detection
- **Purpose**: Evaluate binary classifier performance across all possible thresholds
- **Key Insight**: Shows how well a model separates classes regardless of classification threshold

## Mathematical Foundation

### Core Metrics
```
True Positive Rate (TPR) = Sensitivity = Recall = TP / (TP + FN)
False Positive Rate (FPR) = 1 - Specificity = FP / (FP + TN)

Where:
- TP: True Positives
- TN: True Negatives  
- FP: False Positives
- FN: False Negatives
```

### ROC Curve Construction
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load sample dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Get prediction probabilities
y_proba = clf.predict_proba(X_test)[:, 1]  # Probability of positive class

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

print(f"AUC-ROC Score: {roc_auc:.4f}")
print(f"Number of thresholds: {len(thresholds)}")
```

### Basic ROC Curve Visualization
```python
def plot_basic_roc_curve(fpr, tpr, roc_auc, title="ROC Curve"):
    """Plot a basic ROC curve"""
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()

# Plot the ROC curve
plot_basic_roc_curve(fpr, tpr, roc_auc, "Breast Cancer Classification ROC Curve")
```

## Understanding ROC Curve Components

### Interpreting Different Points
```python
def explain_roc_curve_points(fpr, tpr, thresholds):
    """Explain key points on ROC curve"""
    
    print("Key Points on ROC Curve:")
    print("=" * 40)
    
    # Perfect classifier
    print("(0, 1) - Perfect Classifier:")
    print("  100% sensitivity, 100% specificity")
    print("  All positives correctly identified, no false positives")
    
    # Random classifier
    print("\n(0.5, 0.5) - Random Classifier:")
    print("  50% sensitivity, 50% specificity")
    print("  No better than random guessing")
    
    # Worst classifier
    print("\n(1, 0) - Worst Classifier:")
    print("  0% sensitivity, 0% specificity")
    print("  All predictions are wrong")
    
    # Conservative classifier (high threshold)
    conservative_idx = np.argmax(tpr - fpr)  # Youden's index
    print(f"\nOptimal Point (Youden's Index):")
    print(f"  FPR: {fpr[conservative_idx]:.3f}, TPR: {tpr[conservative_idx]:.3f}")
    print(f"  Threshold: {thresholds[conservative_idx]:.3f}")
    print(f"  Maximizes (Sensitivity + Specificity - 1)")
    
    # Show some actual points
    print(f"\nSample Points from ROC Curve:")
    for i in [0, len(fpr)//4, len(fpr)//2, 3*len(fpr)//4, -1]:
        if i == -1:
            i = len(fpr) - 1
        print(f"  Threshold {thresholds[i]:.3f}: FPR={fpr[i]:.3f}, TPR={tpr[i]:.3f}")

explain_roc_curve_points(fpr, tpr, thresholds)
```

### ROC Curve Anatomy Visualization
```python
def plot_roc_curve_anatomy(fpr, tpr, thresholds, roc_auc):
    """Detailed ROC curve with annotations"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Main ROC curve with annotations
    ax1.plot(fpr, tpr, color='darkorange', lw=3, label=f'Model (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    # Mark key points
    optimal_idx = np.argmax(tpr - fpr)
    ax1.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10, 
             label=f'Optimal (Youden)')
    
    # Mark perfect classifier
    ax1.plot(0, 1, 'gs', markersize=10, label='Perfect')
    
    # Add threshold annotations for key points
    for i in [0, len(thresholds)//4, len(thresholds)//2, 3*len(thresholds)//4, -1]:
        if i == -1:
            i = len(thresholds) - 1
        ax1.annotate(f'θ={thresholds[i]:.2f}', 
                    xy=(fpr[i], tpr[i]), xytext=(5, 5),
                    textcoords='offset points', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))
    
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (1 - Specificity)')
    ax1.set_ylabel('True Positive Rate (Sensitivity)')
    ax1.set_title('ROC Curve Anatomy')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Threshold vs TPR/FPR
    ax2.plot(thresholds, tpr, 'g-', label='True Positive Rate', linewidth=2)
    ax2.plot(thresholds, fpr, 'r-', label='False Positive Rate', linewidth=2)
    ax2.plot(thresholds, tpr - fpr, 'b-', label='TPR - FPR (Youden)', linewidth=2)
    ax2.axvline(x=thresholds[optimal_idx], color='k', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Classification Threshold')
    ax2.set_ylabel('Rate')
    ax2.set_title('Threshold vs Rates')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

plot_roc_curve_anatomy(fpr, tpr, thresholds, roc_auc)
```

## AUC (Area Under the Curve)

### Understanding AUC
```python
def explain_auc():
    """Explain AUC interpretation and significance"""
    
    print("AUC (Area Under the ROC Curve) Interpretation:")
    print("=" * 50)
    
    auc_ranges = {
        "1.0": "Perfect classifier - can perfectly separate classes",
        "0.9 - 1.0": "Excellent performance - very good separation",
        "0.8 - 0.9": "Good performance - reasonable separation", 
        "0.7 - 0.8": "Fair performance - some separation ability",
        "0.6 - 0.7": "Poor performance - limited separation",
        "0.5 - 0.6": "Very poor performance - barely better than random",
        "0.5": "Random classifier - no discriminative ability",
        "0.0 - 0.5": "Worse than random - predictions are inverted"
    }
    
    for range_str, interpretation in auc_ranges.items():
        print(f"AUC {range_str}: {interpretation}")
    
    print(f"\nStatistical Interpretation:")
    print(f"AUC = Probability that the model ranks a randomly chosen")
    print(f"positive instance higher than a randomly chosen negative instance")

explain_auc()
```

### AUC Calculation Methods
```python
def calculate_auc_methods(y_true, y_proba):
    """Demonstrate different AUC calculation methods"""
    
    # Method 1: Using roc_auc_score (most common)
    auc_sklearn = roc_auc_score(y_true, y_proba)
    
    # Method 2: Using roc_curve and auc
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_manual = auc(fpr, tpr)
    
    # Method 3: Trapezoidal rule implementation
    def auc_trapezoidal(fpr, tpr):
        return np.trapz(tpr, fpr)
    
    auc_trapz = auc_trapezoidal(fpr, tpr)
    
    # Method 4: Mann-Whitney U statistic approach
    def auc_mann_whitney(y_true, y_proba):
        pos_scores = y_proba[y_true == 1]
        neg_scores = y_proba[y_true == 0]
        
        total_pairs = len(pos_scores) * len(neg_scores)
        correct_pairs = 0
        tied_pairs = 0
        
        for pos_score in pos_scores:
            correct_pairs += np.sum(pos_score > neg_scores)
            tied_pairs += np.sum(pos_score == neg_scores)
        
        return (correct_pairs + 0.5 * tied_pairs) / total_pairs
    
    auc_mw = auc_mann_whitney(y_true, y_proba)
    
    print("AUC Calculation Methods:")
    print("=" * 30)
    print(f"sklearn roc_auc_score: {auc_sklearn:.6f}")
    print(f"Manual roc_curve + auc: {auc_manual:.6f}")
    print(f"Trapezoidal rule: {auc_trapz:.6f}")
    print(f"Mann-Whitney U: {auc_mw:.6f}")
    
    return auc_sklearn

# Calculate AUC using different methods
auc_result = calculate_auc_methods(y_test, y_proba)
```

## Multi-class ROC Curves

### One-vs-Rest (OvR) Approach
```python
def plot_multiclass_roc_ovr():
    """Plot ROC curves for multi-class classification using One-vs-Rest"""
    
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import label_binarize
    from sklearn.multiclass import OneVsRestClassifier
    from itertools import cycle
    
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Binarize labels for multi-class ROC
    y_bin = label_binarize(y, classes=[0, 1, 2])
    n_classes = y_bin.shape[1]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.3, random_state=42)
    
    # Train One-vs-Rest classifier
    classifier = OneVsRestClassifier(RandomForestClassifier(random_state=42))
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    
    # Compute ROC curve for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot ROC curves
    plt.figure(figsize=(12, 8))
    
    # Class-specific curves
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    class_names = iris.target_names
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    # Micro-average curve
    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4,
             label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})')
    
    # Random classifier line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curves (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return fpr, tpr, roc_auc

# Plot multi-class ROC curves
mc_fpr, mc_tpr, mc_auc = plot_multiclass_roc_ovr()
```

### Macro and Micro Averaging
```python
def explain_multiclass_averaging():
    """Explain macro and micro averaging for multi-class ROC"""
    
    print("Multi-class ROC Averaging Methods:")
    print("=" * 40)
    
    print("MICRO-AVERAGING:")
    print("  - Aggregate all individual TP, FP, TN, FN across classes")
    print("  - Calculate global FPR and TPR")
    print("  - Gives equal weight to each sample")
    print("  - Dominated by performance on majority classes")
    print("  - Single ROC curve and AUC value")
    
    print("\nMACRO-AVERAGING:")
    print("  - Calculate metrics for each class independently")
    print("  - Average the individual class metrics")
    print("  - Gives equal weight to each class")
    print("  - Better for imbalanced datasets")
    print("  - Multiple ROC curves, one per class")
    
    print("\nWhen to use each:")
    print("  - Micro: When you care about overall classification accuracy")
    print("  - Macro: When all classes are equally important")
    print("  - Micro: For imbalanced datasets where majority class dominates")
    print("  - Macro: For balanced datasets or when minority classes matter")

explain_multiclass_averaging()
```

## Model Comparison using ROC Curves

### Comparing Multiple Models
```python
def compare_models_roc():
    """Compare multiple models using ROC curves"""
    
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    
    # Models to compare
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier()
    }
    
    # Scale features for algorithms that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    plt.figure(figsize=(12, 8))
    colors = ['darkorange', 'red', 'blue', 'green', 'purple']
    
    model_aucs = {}
    
    for (name, model), color in zip(models.items(), colors):
        # Use scaled data for SVM, Logistic Regression, KNN
        if name in ['SVM', 'Logistic Regression', 'KNN']:
            model.fit(X_train_scaled, y_train)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate ROC curve
        fpr_model, tpr_model, _ = roc_curve(y_test, y_proba)
        auc_model = auc(fpr_model, tpr_model)
        model_aucs[name] = auc_model
        
        # Plot ROC curve
        plt.plot(fpr_model, tpr_model, color=color, lw=2,
                label=f'{name} (AUC = {auc_model:.3f})')
    
    # Random classifier
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison - Multiple Models')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Print ranking
    print("Model Ranking by AUC:")
    print("=" * 25)
    sorted_models = sorted(model_aucs.items(), key=lambda x: x[1], reverse=True)
    for i, (name, auc_score) in enumerate(sorted_models, 1):
        print(f"{i}. {name}: {auc_score:.4f}")
    
    return model_aucs

# Compare models
model_performances = compare_models_roc()
```

### Statistical Significance Testing
```python
def compare_roc_statistical_significance(y_true, y_proba1, y_proba2, 
                                       model1_name="Model 1", model2_name="Model 2"):
    """Compare two ROC curves for statistical significance"""
    
    from scipy import stats
    import bootstrapped.bootstrap as bs
    import bootstrapped.stats_functions as bs_stats
    
    # Calculate AUCs
    auc1 = roc_auc_score(y_true, y_proba1)
    auc2 = roc_auc_score(y_true, y_proba2)
    
    print(f"AUC Comparison:")
    print(f"{model1_name}: {auc1:.4f}")
    print(f"{model2_name}: {auc2:.4f}")
    print(f"Difference: {abs(auc1 - auc2):.4f}")
    
    # Bootstrap confidence intervals for AUCs
    def auc_score(y_true, y_proba):
        return roc_auc_score(y_true, y_proba)
    
    # Bootstrap for model 1
    try:
        # Create bootstrap samples
        n_bootstrap = 1000
        auc1_bootstrap = []
        auc2_bootstrap = []
        
        n_samples = len(y_true)
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_boot = y_true[indices]
            y_proba1_boot = y_proba1[indices]
            y_proba2_boot = y_proba2[indices]
            
            # Calculate AUCs
            auc1_bootstrap.append(roc_auc_score(y_boot, y_proba1_boot))
            auc2_bootstrap.append(roc_auc_score(y_boot, y_proba2_boot))
        
        # Calculate confidence intervals
        auc1_ci = np.percentile(auc1_bootstrap, [2.5, 97.5])
        auc2_ci = np.percentile(auc2_bootstrap, [2.5, 97.5])
        
        print(f"\n95% Confidence Intervals:")
        print(f"{model1_name}: [{auc1_ci[0]:.4f}, {auc1_ci[1]:.4f}]")
        print(f"{model2_name}: [{auc2_ci[0]:.4f}, {auc2_ci[1]:.4f}]")
        
        # Test for significant difference
        auc_diff_bootstrap = np.array(auc1_bootstrap) - np.array(auc2_bootstrap)
        p_value = 2 * min(np.mean(auc_diff_bootstrap > 0), np.mean(auc_diff_bootstrap < 0))
        
        print(f"\nStatistical Test:")
        print(f"Null hypothesis: AUCs are equal")
        print(f"P-value: {p_value:.4f}")
        print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'} (α = 0.05)")
        
    except Exception as e:
        print(f"Bootstrap analysis failed: {e}")
        print("Using simple comparison instead")
    
    # Visualize comparison
    plt.figure(figsize=(15, 5))
    
    # ROC curves
    plt.subplot(1, 3, 1)
    fpr1, tpr1, _ = roc_curve(y_true, y_proba1)
    fpr2, tpr2, _ = roc_curve(y_true, y_proba2)
    
    plt.plot(fpr1, tpr1, label=f'{model1_name} (AUC = {auc1:.3f})', lw=2)
    plt.plot(fpr2, tpr2, label=f'{model2_name} (AUC = {auc2:.3f})', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Bootstrap distributions
    if 'auc1_bootstrap' in locals():
        plt.subplot(1, 3, 2)
        plt.hist(auc1_bootstrap, alpha=0.7, label=model1_name, bins=30)
        plt.hist(auc2_bootstrap, alpha=0.7, label=model2_name, bins=30)
        plt.xlabel('AUC Score')
        plt.ylabel('Frequency')
        plt.title('Bootstrap AUC Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Difference distribution
        plt.subplot(1, 3, 3)
        plt.hist(auc_diff_bootstrap, bins=30, alpha=0.7)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('AUC Difference (Model 1 - Model 2)')
        plt.ylabel('Frequency')
        plt.title('Bootstrap Difference Distribution')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Compare two best models statistically
rf_proba = RandomForestClassifier(random_state=42).fit(X_train, y_train).predict_proba(X_test)[:, 1]
lr_proba = LogisticRegression(random_state=42, max_iter=1000).fit(
    StandardScaler().fit_transform(X_train), y_train
).predict_proba(StandardScaler().fit_transform(X_test))[:, 1]

compare_roc_statistical_significance(y_test, rf_proba, lr_proba, 
                                   "Random Forest", "Logistic Regression")
```

## ROC Curve Applications

### Threshold Selection
```python
def optimal_threshold_selection(y_true, y_proba, method='youden'):
    """Select optimal threshold using different criteria"""
    
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    
    methods = {}
    
    # Youden's Index (maximize TPR - FPR)
    youden_idx = np.argmax(tpr - fpr)
    methods['youden'] = {
        'threshold': thresholds[youden_idx],
        'fpr': fpr[youden_idx],
        'tpr': tpr[youden_idx],
        'description': 'Maximizes (Sensitivity + Specificity - 1)'
    }
    
    # Closest to top-left corner
    distances = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
    closest_idx = np.argmin(distances)
    methods['closest_topleft'] = {
        'threshold': thresholds[closest_idx],
        'fpr': fpr[closest_idx],
        'tpr': tpr[closest_idx],
        'description': 'Minimizes distance to (0,1)'
    }
    
    # Maximum F1 score
    from sklearn.metrics import f1_score
    f1_scores = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f1_scores.append(f1_score(y_true, y_pred))
    
    f1_idx = np.argmax(f1_scores)
    methods['max_f1'] = {
        'threshold': thresholds[f1_idx],
        'fpr': fpr[f1_idx],
        'tpr': tpr[f1_idx],
        'f1_score': f1_scores[f1_idx],
        'description': 'Maximizes F1 score'
    }
    
    # Cost-sensitive threshold (example: FP cost = 1, FN cost = 5)
    fp_cost = 1
    fn_cost = 5
    costs = fp_cost * fpr * np.sum(y_true == 0) + fn_cost * (1 - tpr) * np.sum(y_true == 1)
    cost_idx = np.argmin(costs)
    methods['cost_sensitive'] = {
        'threshold': thresholds[cost_idx],
        'fpr': fpr[cost_idx],
        'tpr': tpr[cost_idx],
        'cost': costs[cost_idx],
        'description': f'Minimizes cost (FP cost={fp_cost}, FN cost={fn_cost})'
    }
    
    # Visualize threshold selection
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC curve with optimal points
    axes[0, 0].plot(fpr, tpr, 'b-', lw=2, label='ROC Curve')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.7)
    
    colors = ['red', 'green', 'orange', 'purple']
    for (method_name, method_info), color in zip(methods.items(), colors):
        axes[0, 0].plot(method_info['fpr'], method_info['tpr'], 'o', 
                       color=color, markersize=10, label=method_name.replace('_', ' ').title())
    
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve with Optimal Thresholds')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Threshold vs TPR/FPR
    axes[0, 1].plot(thresholds, tpr, 'g-', label='TPR', lw=2)
    axes[0, 1].plot(thresholds, fpr, 'r-', label='FPR', lw=2)
    axes[0, 1].plot(thresholds, tpr - fpr, 'b-', label='TPR - FPR', lw=2)
    
    for method_name, method_info in methods.items():
        if method_name == 'youden':
            axes[0, 1].axvline(x=method_info['threshold'], color='red', 
                              linestyle='--', alpha=0.7, label='Youden')
    
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('Rate')
    axes[0, 1].set_title('Threshold vs Rates')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 scores vs threshold
    axes[1, 0].plot(thresholds, f1_scores, 'purple', lw=2)
    axes[1, 0].axvline(x=methods['max_f1']['threshold'], color='orange', 
                      linestyle='--', alpha=0.7, label='Max F1')
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score vs Threshold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cost vs threshold
    axes[1, 1].plot(thresholds, costs, 'brown', lw=2)
    axes[1, 1].axvline(x=methods['cost_sensitive']['threshold'], color='purple', 
                      linestyle='--', alpha=0.7, label='Min Cost')
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('Total Cost')
    axes[1, 1].set_title('Cost vs Threshold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("Optimal Threshold Selection Results:")
    print("=" * 50)
    for method_name, method_info in methods.items():
        print(f"\n{method_name.replace('_', ' ').title()}:")
        print(f"  Description: {method_info['description']}")
        print(f"  Threshold: {method_info['threshold']:.4f}")
        print(f"  FPR: {method_info['fpr']:.4f}")
        print(f"  TPR: {method_info['tpr']:.4f}")
        if 'f1_score' in method_info:
            print(f"  F1 Score: {method_info['f1_score']:.4f}")
        if 'cost' in method_info:
            print(f"  Total Cost: {method_info['cost']:.2f}")
    
    return methods

# Select optimal thresholds
optimal_thresholds = optimal_threshold_selection(y_test, y_proba)
```

### ROC Curve for Imbalanced Datasets
```python
def roc_imbalanced_analysis():
    """Analyze ROC curve behavior with imbalanced datasets"""
    
    # Create imbalanced dataset
    X_imbal, y_imbal = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=10, n_clusters_per_class=1, 
        weights=[0.95, 0.05], random_state=42
    )
    
    X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
        X_imbal, y_imbal, test_size=0.3, random_state=42
    )
    
    # Train model
    clf_imb = RandomForestClassifier(random_state=42)
    clf_imb.fit(X_train_imb, y_train_imb)
    y_proba_imb = clf_imb.predict_proba(X_test_imb)[:, 1]
    
    # Calculate metrics
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    # ROC metrics
    fpr_imb, tpr_imb, _ = roc_curve(y_test_imb, y_proba_imb)
    roc_auc_imb = auc(fpr_imb, tpr_imb)
    
    # Precision-Recall metrics
    precision, recall, _ = precision_recall_curve(y_test_imb, y_proba_imb)
    pr_auc = average_precision_score(y_test_imb, y_proba_imb)
    
    # Compare with balanced dataset
    X_bal, y_bal = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=10, n_clusters_per_class=1,
        weights=[0.5, 0.5], random_state=42
    )
    
    X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(
        X_bal, y_bal, test_size=0.3, random_state=42
    )
    
    clf_bal = RandomForestClassifier(random_state=42)
    clf_bal.fit(X_train_bal, y_train_bal)
    y_proba_bal = clf_bal.predict_proba(X_test_bal)[:, 1]
    
    fpr_bal, tpr_bal, _ = roc_curve(y_test_bal, y_proba_bal)
    roc_auc_bal = auc(fpr_bal, tpr_bal)
    
    precision_bal, recall_bal, _ = precision_recall_curve(y_test_bal, y_proba_bal)
    pr_auc_bal = average_precision_score(y_test_bal, y_proba_bal)
    
    # Visualize comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC curves comparison
    axes[0, 0].plot(fpr_imb, tpr_imb, 'r-', lw=2, 
                   label=f'Imbalanced (AUC = {roc_auc_imb:.3f})')
    axes[0, 0].plot(fpr_bal, tpr_bal, 'b-', lw=2, 
                   label=f'Balanced (AUC = {roc_auc_bal:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.7)
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curves: Balanced vs Imbalanced')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Precision-Recall curves comparison
    axes[0, 1].plot(recall, precision, 'r-', lw=2, 
                   label=f'Imbalanced (AP = {pr_auc:.3f})')
    axes[0, 1].plot(recall_bal, precision_bal, 'b-', lw=2, 
                   label=f'Balanced (AP = {pr_auc_bal:.3f})')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Class distributions
    axes[1, 0].bar(['Negative', 'Positive'], 
                  [np.sum(y_test_imb == 0), np.sum(y_test_imb == 1)],
                  color=['blue', 'red'], alpha=0.7)
    axes[1, 0].set_title('Imbalanced Dataset Distribution')
    axes[1, 0].set_ylabel('Number of Samples')
    
    axes[1, 1].bar(['Negative', 'Positive'], 
                  [np.sum(y_test_bal == 0), np.sum(y_test_bal == 1)],
                  color=['blue', 'red'], alpha=0.7)
    axes[1, 1].set_title('Balanced Dataset Distribution')
    axes[1, 1].set_ylabel('Number of Samples')
    
    plt.tight_layout()
    plt.show()
    
    print("Imbalanced Dataset Analysis:")
    print("=" * 40)
    print(f"Imbalanced dataset:")
    print(f"  Class distribution: {np.bincount(y_test_imb)}")
    print(f"  ROC AUC: {roc_auc_imb:.4f}")
    print(f"  PR AUC: {pr_auc:.4f}")
    
    print(f"\nBalanced dataset:")
    print(f"  Class distribution: {np.bincount(y_test_bal)}")
    print(f"  ROC AUC: {roc_auc_bal:.4f}")
    print(f"  PR AUC: {pr_auc_bal:.4f}")
    
    print(f"\nKey Insights:")
    print(f"  - ROC AUC is less sensitive to class imbalance")
    print(f"  - PR AUC better reflects performance on minority class")
    print(f"  - For imbalanced data, consider both ROC and PR curves")

roc_imbalanced_analysis()
```

## Advanced ROC Analysis

### Confidence Intervals for ROC Curves
```python
def roc_confidence_intervals(y_true, y_proba, confidence=0.95, n_bootstrap=1000):
    """Calculate confidence intervals for ROC curve using bootstrap"""
    
    # Original ROC curve
    fpr_orig, tpr_orig, thresholds_orig = roc_curve(y_true, y_proba)
    auc_orig = auc(fpr_orig, tpr_orig)
    
    # Bootstrap sampling
    n_samples = len(y_true)
    bootstrap_aucs = []
    bootstrap_tprs = []
    
    # Common FPR points for interpolation
    mean_fpr = np.linspace(0, 1, 100)
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_boot = y_true[indices]
        y_proba_boot = y_proba[indices]
        
        # Calculate ROC curve
        fpr_boot, tpr_boot, _ = roc_curve(y_boot, y_proba_boot)
        
        # Interpolate TPR at common FPR points
        tpr_interp = np.interp(mean_fpr, fpr_boot, tpr_boot)
        tpr_interp[0] = 0.0  # Ensure it starts at (0,0)
        bootstrap_tprs.append(tpr_interp)
        
        # Calculate AUC
        bootstrap_aucs.append(auc(fpr_boot, tpr_boot))
    
    # Calculate confidence intervals
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    # AUC confidence interval
    auc_ci = np.percentile(bootstrap_aucs, [lower_percentile, upper_percentile])
    
    # TPR confidence intervals at each FPR point
    bootstrap_tprs = np.array(bootstrap_tprs)
    tpr_lower = np.percentile(bootstrap_tprs, lower_percentile, axis=0)
    tpr_upper = np.percentile(bootstrap_tprs, upper_percentile, axis=0)
    tpr_mean = np.mean(bootstrap_tprs, axis=0)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Original ROC curve
    plt.plot(fpr_orig, tpr_orig, 'b-', lw=3, 
             label=f'Original ROC (AUC = {auc_orig:.3f})')
    
    # Mean bootstrap ROC curve
    plt.plot(mean_fpr, tpr_mean, 'r--', lw=2, 
             label=f'Bootstrap Mean (AUC = {np.mean(bootstrap_aucs):.3f})')
    
    # Confidence interval
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, alpha=0.3, 
                     label=f'{confidence*100:.0f}% Confidence Interval')
    
    # Random classifier
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve with {confidence*100:.0f}% Confidence Intervals')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Print results
    print(f"ROC Curve Confidence Analysis:")
    print(f"=" * 40)
    print(f"Original AUC: {auc_orig:.4f}")
    print(f"Bootstrap Mean AUC: {np.mean(bootstrap_aucs):.4f}")
    print(f"Bootstrap Std AUC: {np.std(bootstrap_aucs):.4f}")
    print(f"{confidence*100:.0f}% CI for AUC: [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]")
    
    return {
        'auc_original': auc_orig,
        'auc_bootstrap': bootstrap_aucs,
        'auc_ci': auc_ci,
        'fpr_points': mean_fpr,
        'tpr_mean': tpr_mean,
        'tpr_ci_lower': tpr_lower,
        'tpr_ci_upper': tpr_upper
    }

# Calculate confidence intervals
ci_results = roc_confidence_intervals(y_test, y_proba, confidence=0.95)
```

### Partial AUC Analysis
```python
def partial_auc_analysis(y_true, y_proba, fpr_range=(0.0, 0.1)):
    """Calculate partial AUC for specific FPR range"""
    
    from sklearn.metrics import roc_auc_score
    
    # Full ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    full_auc = auc(fpr, tpr)
    
    # Calculate partial AUC
    fpr_min, fpr_max = fpr_range
    
    # Find indices within the FPR range
    mask = (fpr >= fpr_min) & (fpr <= fpr_max)
    fpr_partial = fpr[mask]
    tpr_partial = tpr[mask]
    
    if len(fpr_partial) < 2:
        print("Not enough points in the specified FPR range")
        return None
    
    # Calculate partial AUC
    partial_auc = auc(fpr_partial, tpr_partial)
    
    # Normalize partial AUC to range [0, 1]
    fpr_width = fpr_max - fpr_min
    max_possible_auc = fpr_width  # Area of rectangle
    normalized_partial_auc = partial_auc / max_possible_auc
    
    # Standardized partial AUC (McClish, 1989)
    # Adjusts for the fact that partial AUC depends on the range
    min_possible_auc = 0.5 * fpr_width  # Random classifier in this range
    standardized_partial_auc = (partial_auc - min_possible_auc) / (max_possible_auc - min_possible_auc)
    
    # Visualize
    plt.figure(figsize=(12, 8))
    
    # Full ROC curve
    plt.plot(fpr, tpr, 'b-', lw=2, label=f'Full ROC (AUC = {full_auc:.3f})')
    
    # Highlight partial area
    plt.fill_between(fpr_partial, 0, tpr_partial, alpha=0.3, color='red',
                     label=f'Partial AUC = {partial_auc:.4f}')
    
    # Mark the FPR range
    plt.axvline(x=fpr_min, color='red', linestyle='--', alpha=0.7)
    plt.axvline(x=fpr_max, color='red', linestyle='--', alpha=0.7)
    
    # Random classifier
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Random Classifier')
    
    # Random classifier in partial range
    plt.plot([fpr_min, fpr_max], [fpr_min, fpr_max], 'r--', alpha=0.7,
             label=f'Random in range [{fpr_min}, {fpr_max}]')
    
    plt.xlim([0.0, max(0.2, fpr_max + 0.05)])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Partial AUC Analysis (FPR range: {fpr_min} - {fpr_max})')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Partial AUC Analysis:")
    print(f"=" * 30)
    print(f"FPR Range: [{fpr_min}, {fpr_max}]")
    print(f"Full AUC: {full_auc:.4f}")
    print(f"Partial AUC: {partial_auc:.4f}")
    print(f"Normalized Partial AUC: {normalized_partial_auc:.4f}")
    print(f"Standardized Partial AUC: {standardized_partial_auc:.4f}")
    
    print(f"\nInterpretation:")
    print(f"- Partial AUC focuses on performance at low FPR (high specificity)")
    print(f"- Useful when false positives are very costly")
    print(f"- Standardized partial AUC: {standardized_partial_auc:.4f}")
    print(f"  (0.5 = random, 1.0 = perfect in this range)")
    
    return {
        'partial_auc': partial_auc,
        'normalized_partial_auc': normalized_partial_auc,
        'standardized_partial_auc': standardized_partial_auc,
        'fpr_range': fpr_range
    }

# Analyze partial AUC for low false positive rates
partial_results = partial_auc_analysis(y_test, y_proba, fpr_range=(0.0, 0.1))
```

## Common Pitfalls and Best Practices

### ROC Curve Pitfalls
```python
def roc_pitfalls_and_solutions():
    """Common pitfalls when using ROC curves and their solutions"""
    
    print("Common ROC Curve Pitfalls and Solutions:")
    print("=" * 50)
    
    pitfalls = {
        "1. Using ROC with Highly Imbalanced Data": {
            "Problem": "ROC curves can be overly optimistic with imbalanced classes",
            "Why": "High TNR due to many true negatives inflates performance",
            "Solution": "Use Precision-Recall curves for imbalanced datasets",
            "Example": "99% negative class: ROC AUC might be 0.9, but PR AUC could be 0.3"
        },
        
        "2. Misinterpreting AUC Values": {
            "Problem": "Thinking AUC directly represents accuracy",
            "Why": "AUC is probability of ranking, not classification accuracy",
            "Solution": "Consider AUC as ranking quality, use accuracy for classification",
            "Example": "AUC=0.8 doesn't mean 80% accuracy"
        },
        
        "3. Comparing ROC Curves Visually Only": {
            "Problem": "Visual comparison can be misleading",
            "Why": "Curves can cross, partial areas matter differently",
            "Solution": "Use statistical tests, consider partial AUC",
            "Example": "Model A better at low FPR, Model B better at high FPR"
        },
        
        "4. Ignoring Class Distribution Changes": {
            "Problem": "ROC curves don't reflect deployment class distribution",
            "Why": "Training and deployment distributions may differ",
            "Solution": "Validate on deployment-representative data",
            "Example": "Model trained on 50:50 deployed on 95:5 split"
        },
        
        "5. Using Default Threshold (0.5)": {
            "Problem": "Default threshold may not be optimal",
            "Why": "Optimal threshold depends on cost/benefit trade-offs",
            "Solution": "Choose threshold based on business requirements",
            "Example": "Medical diagnosis: optimize for high sensitivity"
        },
        
        "6. Not Considering Calibration": {
            "Problem": "Probability scores may not be well-calibrated",
            "Why": "Many algorithms don't output true probabilities",
            "Solution": "Use calibration techniques (Platt scaling, isotonic)",
            "Example": "SVM outputs not interpretable as probabilities"
        }
    }
    
    for pitfall, details in pitfalls.items():
        print(f"\n{pitfall}")
        print(f"  Problem: {details['Problem']}")
        print(f"  Why: {details['Why']}")
        print(f"  Solution: {details['Solution']}")
        print(f"  Example: {details['Example']}")

roc_pitfalls_and_solutions()
```

### Best Practices Checklist
```python
def roc_best_practices():
    """Best practices for ROC curve analysis"""
    
    print("ROC Curve Best Practices Checklist:")
    print("=" * 40)
    
    practices = [
        "✓ Consider class distribution before choosing ROC vs PR curves",
        "✓ Report confidence intervals for AUC when sample size allows",
        "✓ Use cross-validation to get stable AUC estimates",
        "✓ Compare models statistically, not just visually",
        "✓ Choose thresholds based on business requirements",
        "✓ Validate on truly independent test set",
        "✓ Consider calibration for probability interpretation",
        "✓ Use partial AUC when specific FPR ranges matter",
        "✓ Report multiple metrics beyond just AUC",
        "✓ Document the evaluation protocol clearly"
    ]
    
    for practice in practices:
        print(f"  {practice}")
    
    print(f"\nReporting Guidelines:")
    print(f"  - Always report the evaluation protocol")
    print(f"  - Include confidence intervals when possible")
    print(f"  - Specify which threshold selection method was used")
    print(f"  - Mention class distribution in training/test sets")
    print(f"  - Compare with appropriate baselines")

roc_best_practices()
```

## Learning Objectives
- [ ] **Understand ROC fundamentals**: Master TPR, FPR, and threshold relationships
- [ ] **Interpret AUC values**: Understand what AUC represents and its limitations
- [ ] **Create ROC visualizations**: Plot clear, informative ROC curves with proper annotations
- [ ] **Compare models effectively**: Use ROC curves to compare multiple algorithms statistically
- [ ] **Select optimal thresholds**: Choose thresholds based on business requirements and cost considerations
- [ ] **Handle multi-class problems**: Apply ROC analysis to multi-class classification scenarios
- [ ] **Analyze imbalanced datasets**: Understand when ROC is appropriate vs alternatives like PR curves
- [ ] **Apply advanced techniques**: Use confidence intervals, partial AUC, and bootstrap analysis
- [ ] **Avoid common pitfalls**: Recognize limitations and misinterpretations of ROC curves
- [ ] **Make informed decisions**: Choose appropriate evaluation metrics based on problem context

## Practice Exercises
1. Create ROC curves for binary classification and interpret all components
2. Compare multiple models using ROC curves and statistical significance testing
3. Analyze the effect of class imbalance on ROC curve interpretation
4. Implement different threshold selection methods and compare results
5. Apply ROC analysis to multi-class problems using one-vs-rest approach
6. Calculate confidence intervals for ROC curves using bootstrap sampling
7. Perform partial AUC analysis for cost-sensitive applications
8. Create interactive ROC curve dashboards with threshold selection
9. Compare ROC and Precision-Recall curves on imbalanced datasets
10. Build automated ROC analysis pipeline for model monitoring
