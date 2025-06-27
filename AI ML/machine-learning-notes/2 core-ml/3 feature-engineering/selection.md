# Feature Selection

## Overview
Feature selection is the process of selecting a subset of relevant features from the original feature set to improve model performance, reduce overfitting, and decrease computational complexity. It's a crucial step in feature engineering that can significantly impact model accuracy and interpretability.

## Why Feature Selection Matters
- **Reduces overfitting**: Fewer features mean less chance of learning noise
- **Improves performance**: Removes irrelevant and redundant features
- **Faster training**: Fewer features mean faster computation
- **Better interpretability**: Simpler models are easier to understand
- **Reduces storage**: Less memory and storage requirements
- **Avoids curse of dimensionality**: High-dimensional data can hurt performance

## Types of Feature Selection

### 1. Filter Methods
Select features based on statistical measures, independent of the machine learning algorithm.

#### Univariate Feature Selection
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_iris, make_classification
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, f_classif, chi2, mutual_info_classif,
    VarianceThreshold, RFE, SelectFromModel
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load sample dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
feature_names = cancer.feature_names

print(f"Original dataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Variance Threshold - Remove low variance features
def variance_threshold_selection(X, threshold=0.1):
    """Remove features with low variance"""
    
    selector = VarianceThreshold(threshold=threshold)
    X_selected = selector.fit_transform(X)
    
    selected_features = np.array(feature_names)[selector.get_support()]
    
    print(f"Variance Threshold Selection (threshold={threshold}):")
    print(f"Original features: {X.shape[1]}")
    print(f"Selected features: {X_selected.shape[1]}")
    print(f"Removed features: {X.shape[1] - X_selected.shape[1]}")
    
    return X_selected, selected_features, selector

X_var_selected, var_features, var_selector = variance_threshold_selection(X_train_scaled)

# 2. Univariate Statistical Tests
def univariate_feature_selection(X, y, score_func=f_classif, k=10):
    """Select features using univariate statistical tests"""
    
    selector = SelectKBest(score_func=score_func, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get feature scores
    scores = selector.scores_
    selected_indices = selector.get_support(indices=True)
    selected_features = np.array(feature_names)[selected_indices]
    
    # Create feature ranking
    feature_scores = list(zip(feature_names, scores))
    feature_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nUnivariate Feature Selection (k={k}):")
    print(f"Score function: {score_func.__name__}")
    print(f"Selected features: {X_selected.shape[1]}")
    
    print(f"\nTop {min(10, len(feature_scores))} features by score:")
    for i, (feature, score) in enumerate(feature_scores[:10]):
        selected = "✓" if feature in selected_features else " "
        print(f"{selected} {i+1:2d}. {feature:25s}: {score:8.2f}")
    
    return X_selected, selected_features, selector, feature_scores

# Test different scoring functions
scoring_functions = [
    (f_classif, "F-statistic"),
    (mutual_info_classif, "Mutual Information")
]

for score_func, name in scoring_functions:
    print(f"\n{'='*60}")
    print(f"Using {name}")
    X_uni_selected, uni_features, uni_selector, scores = univariate_feature_selection(
        X_train_scaled, y_train, score_func=score_func, k=10
    )
```

#### Correlation-based Feature Selection
```python
def correlation_feature_selection(X, y, feature_names, threshold=0.8):
    """Remove highly correlated features"""
    
    # Create DataFrame for easier handling
    df = pd.DataFrame(X, columns=feature_names)
    
    # Calculate correlation matrix
    corr_matrix = df.corr().abs()
    
    # Find highly correlated pairs
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features to drop
    to_drop = [column for column in upper_triangle.columns 
               if any(upper_triangle[column] > threshold)]
    
    # Keep features with higher correlation to target
    target_corr = df.corrwith(pd.Series(y)).abs()
    
    refined_to_drop = []
    for feature in to_drop:
        # Find features highly correlated with this one
        highly_corr = upper_triangle[feature][upper_triangle[feature] > threshold].index
        
        if len(highly_corr) > 0:
            # Keep the one with highest target correlation
            all_features = [feature] + list(highly_corr)
            target_corrs = {f: target_corr[f] for f in all_features}
            best_feature = max(target_corrs, key=target_corrs.get)
            
            for f in all_features:
                if f != best_feature and f not in refined_to_drop:
                    refined_to_drop.append(f)
    
    # Remove duplicates
    refined_to_drop = list(set(refined_to_drop))
    
    # Apply selection
    selected_features = [f for f in feature_names if f not in refined_to_drop]
    selected_indices = [i for i, f in enumerate(feature_names) if f in selected_features]
    X_selected = X[:, selected_indices]
    
    print(f"Correlation-based Feature Selection (threshold={threshold}):")
    print(f"Original features: {len(feature_names)}")
    print(f"Highly correlated pairs found: {len(to_drop)}")
    print(f"Features removed: {len(refined_to_drop)}")
    print(f"Features selected: {len(selected_features)}")
    
    if len(refined_to_drop) > 0:
        print(f"\nRemoved features:")
        for feature in refined_to_drop:
            print(f"  - {feature}")
    
    return X_selected, selected_features, refined_to_drop, corr_matrix

X_corr_selected, corr_features, dropped_features, corr_matrix = correlation_feature_selection(
    X_train_scaled, y_train, feature_names
)

# Visualize correlation matrix
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()
```

### 2. Wrapper Methods
Use machine learning algorithms to evaluate feature subsets.

#### Recursive Feature Elimination (RFE)
```python
def recursive_feature_elimination(X, y, estimator, n_features=10, step=1):
    """Perform recursive feature elimination"""
    
    rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=step)
    X_selected = rfe.fit_transform(X, y)
    
    # Get selected features and rankings
    selected_features = np.array(feature_names)[rfe.support_]
    feature_rankings = rfe.ranking_
    
    # Create ranking summary
    feature_ranking_df = pd.DataFrame({
        'Feature': feature_names,
        'Ranking': feature_rankings,
        'Selected': rfe.support_
    }).sort_values('Ranking')
    
    print(f"Recursive Feature Elimination:")
    print(f"Estimator: {estimator.__class__.__name__}")
    print(f"Features selected: {n_features}")
    print(f"Step size: {step}")
    
    print(f"\nTop {min(15, len(feature_ranking_df))} features by ranking:")
    for _, row in feature_ranking_df.head(15).iterrows():
        status = "✓" if row['Selected'] else " "
        print(f"{status} Rank {row['Ranking']:2d}: {row['Feature']:25s}")
    
    return X_selected, selected_features, rfe, feature_ranking_df

# Test RFE with different estimators
estimators = [
    RandomForestClassifier(n_estimators=50, random_state=42),
    LogisticRegression(random_state=42, max_iter=1000)
]

for estimator in estimators:
    print(f"\n{'='*60}")
    X_rfe, rfe_features, rfe_selector, rankings = recursive_feature_elimination(
        X_train_scaled, y_train, estimator, n_features=10
    )
```

#### Forward/Backward Selection
```python
def sequential_feature_selection(X, y, direction='forward', n_features=10):
    """Perform forward or backward feature selection"""
    
    from sklearn.feature_selection import SequentialFeatureSelector
    from sklearn.ensemble import RandomForestClassifier
    
    estimator = RandomForestClassifier(n_estimators=50, random_state=42)
    
    sfs = SequentialFeatureSelector(
        estimator, 
        n_features_to_select=n_features,
        direction=direction,
        cv=3,
        scoring='accuracy'
    )
    
    X_selected = sfs.fit_transform(X, y)
    selected_features = np.array(feature_names)[sfs.get_support()]
    
    print(f"Sequential Feature Selection ({direction}):")
    print(f"Features selected: {n_features}")
    print(f"Selected features:")
    for i, feature in enumerate(selected_features):
        print(f"  {i+1:2d}. {feature}")
    
    return X_selected, selected_features, sfs

# Forward selection
print("Forward Selection:")
X_forward, forward_features, forward_selector = sequential_feature_selection(
    X_train_scaled, y_train, direction='forward', n_features=8
)

print("\nBackward Selection:")
X_backward, backward_features, backward_selector = sequential_feature_selection(
    X_train_scaled, y_train, direction='backward', n_features=8
)
```

### 3. Embedded Methods
Feature selection is built into the model training process.

#### Tree-based Feature Importance
```python
def tree_based_feature_selection(X, y, estimator, threshold='mean'):
    """Select features using tree-based importance"""
    
    # Train the estimator
    estimator.fit(X, y)
    
    # Get feature importances
    importances = estimator.feature_importances_
    
    # Create selector
    selector = SelectFromModel(estimator, threshold=threshold, prefit=True)
    X_selected = selector.transform(X)
    
    selected_features = np.array(feature_names)[selector.get_support()]
    
    # Create importance ranking
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
        'Selected': selector.get_support()
    }).sort_values('Importance', ascending=False)
    
    print(f"Tree-based Feature Selection:")
    print(f"Estimator: {estimator.__class__.__name__}")
    print(f"Threshold: {threshold}")
    print(f"Features selected: {X_selected.shape[1]}")
    
    print(f"\nTop {min(15, len(feature_importance_df))} features by importance:")
    for _, row in feature_importance_df.head(15).iterrows():
        status = "✓" if row['Selected'] else " "
        print(f"{status} {row['Feature']:25s}: {row['Importance']:.4f}")
    
    # Visualize feature importances
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_df.head(20)
    colors = ['green' if selected else 'lightgray' for selected in top_features['Selected']]
    
    plt.barh(range(len(top_features)), top_features['Importance'], color=colors)
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Feature Importances - {estimator.__class__.__name__}')
    plt.gca().invert_yaxis()
    
    # Add legend
    import matplotlib.patches as mpatches
    selected_patch = mpatches.Patch(color='green', label='Selected')
    not_selected_patch = mpatches.Patch(color='lightgray', label='Not Selected')
    plt.legend(handles=[selected_patch, not_selected_patch])
    
    plt.tight_layout()
    plt.show()
    
    return X_selected, selected_features, selector, feature_importance_df

# Random Forest feature selection
rf = RandomForestClassifier(n_estimators=100, random_state=42)
X_rf_selected, rf_features, rf_selector, rf_importance = tree_based_feature_selection(
    X_train_scaled, y_train, rf, threshold='mean'
)
```

#### L1 Regularization (LASSO)
```python
def lasso_feature_selection(X, y, alpha=0.01):
    """Select features using L1 regularization"""
    
    from sklearn.linear_model import LassoCV, Lasso
    from sklearn.preprocessing import StandardScaler
    
    # Use cross-validation to find optimal alpha
    lasso_cv = LassoCV(cv=5, random_state=42, max_iter=2000)
    lasso_cv.fit(X, y)
    
    optimal_alpha = lasso_cv.alpha_
    
    # Fit LASSO with optimal alpha
    lasso = Lasso(alpha=optimal_alpha, max_iter=2000)
    lasso.fit(X, y)
    
    # Get selected features (non-zero coefficients)
    selected_mask = lasso.coef_ != 0
    selected_features = np.array(feature_names)[selected_mask]
    X_selected = X[:, selected_mask]
    
    # Create coefficient summary
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': lasso.coef_,
        'Abs_Coefficient': np.abs(lasso.coef_),
        'Selected': selected_mask
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print(f"LASSO Feature Selection:")
    print(f"Optimal alpha: {optimal_alpha:.6f}")
    print(f"Features selected: {X_selected.shape[1]}")
    print(f"Features removed: {X.shape[1] - X_selected.shape[1]}")
    
    print(f"\nTop features by coefficient magnitude:")
    for _, row in coef_df.head(15).iterrows():
        status = "✓" if row['Selected'] else " "
        print(f"{status} {row['Feature']:25s}: {row['Coefficient']:8.4f}")
    
    # Visualize coefficients
    plt.figure(figsize=(12, 8))
    top_features = coef_df.head(20)
    colors = ['green' if selected else 'lightgray' for selected in top_features['Selected']]
    
    plt.barh(range(len(top_features)), top_features['Coefficient'], color=colors)
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('LASSO Coefficient')
    plt.title(f'LASSO Coefficients (α={optimal_alpha:.6f})')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.gca().invert_yaxis()
    
    # Add legend
    import matplotlib.patches as mpatches
    selected_patch = mpatches.Patch(color='green', label='Selected')
    not_selected_patch = mpatches.Patch(color='lightgray', label='Not Selected')
    plt.legend(handles=[selected_patch, not_selected_patch])
    
    plt.tight_layout()
    plt.show()
    
    return X_selected, selected_features, lasso, coef_df

# LASSO feature selection
X_lasso_selected, lasso_features, lasso_model, lasso_coef = lasso_feature_selection(
    X_train_scaled, y_train
)
```

## Feature Selection Evaluation

### Performance Comparison
```python
def compare_feature_selection_methods(methods_dict, X_train, X_test, y_train, y_test):
    """Compare different feature selection methods"""
    
    results = {}
    
    for method_name, (X_train_selected, X_test_selected) in methods_dict.items():
        # Train model on selected features
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_selected, y_train)
        
        # Evaluate performance
        train_score = clf.score(X_train_selected, y_train)
        test_score = clf.score(X_test_selected, y_test)
        
        # Get predictions for detailed metrics
        y_pred = clf.predict(X_test_selected)
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[method_name] = {
            'n_features': X_train_selected.shape[1],
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'overfitting': train_score - test_score
        }
    
    # Create results DataFrame
    results_df = pd.DataFrame(results).T
    
    print("Feature Selection Method Comparison:")
    print("=" * 80)
    print(results_df.round(4))
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Test accuracy vs number of features
    axes[0, 0].scatter(results_df['n_features'], results_df['test_accuracy'])
    for method, row in results_df.iterrows():
        axes[0, 0].annotate(method, (row['n_features'], row['test_accuracy']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[0, 0].set_xlabel('Number of Features')
    axes[0, 0].set_ylabel('Test Accuracy')
    axes[0, 0].set_title('Accuracy vs Number of Features')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Overfitting analysis
    axes[0, 1].bar(results_df.index, results_df['overfitting'])
    axes[0, 1].set_ylabel('Overfitting (Train - Test Accuracy)')
    axes[0, 1].set_title('Overfitting by Method')
    plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Performance metrics comparison
    metrics = ['precision', 'recall', 'f1_score']
    x = np.arange(len(results_df))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        axes[1, 0].bar(x + i*width, results_df[metric], width, 
                      label=metric.replace('_', ' ').title())
    
    axes[1, 0].set_xlabel('Method')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Performance Metrics Comparison')
    axes[1, 0].set_xticks(x + width)
    axes[1, 0].set_xticklabels(results_df.index, rotation=45, ha='right')
    axes[1, 0].legend()
    
    # Feature count comparison
    axes[1, 1].bar(results_df.index, results_df['n_features'])
    axes[1, 1].set_ylabel('Number of Features')
    axes[1, 1].set_title('Feature Count by Method')
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    return results_df

# Prepare data for comparison
# Apply selectors to test data
methods_comparison = {
    'Original': (X_train_scaled, X_test_scaled),
    'Variance Threshold': (
        var_selector.transform(X_train_scaled), 
        var_selector.transform(X_test_scaled)
    ),
    'Univariate (k=10)': (
        uni_selector.transform(X_train_scaled), 
        uni_selector.transform(X_test_scaled)
    ),
    'RFE (RF)': (
        rfe_selector.transform(X_train_scaled),
        rfe_selector.transform(X_test_scaled)
    ),
    'Random Forest': (
        rf_selector.transform(X_train_scaled),
        rf_selector.transform(X_test_scaled)
    ),
    'LASSO': (
        X_lasso_selected,
        X_test_scaled[:, lasso_model.coef_ != 0]
    )
}

# Compare methods
comparison_results = compare_feature_selection_methods(
    methods_comparison, X_train_scaled, X_test_scaled, y_train, y_test
)
```

### Learning Curves with Feature Selection
```python
def plot_learning_curves_feature_selection(X, y, feature_selectors, cv=5):
    """Plot learning curves for different feature selection methods"""
    
    from sklearn.model_selection import learning_curve
    from sklearn.pipeline import Pipeline
    
    plt.figure(figsize=(15, 10))
    
    # Define base estimator
    estimator = RandomForestClassifier(n_estimators=50, random_state=42)
    
    for i, (name, selector) in enumerate(feature_selectors.items()):
        plt.subplot(2, 3, i+1)
        
        if selector is None:
            # Original features
            train_sizes, train_scores, val_scores = learning_curve(
                estimator, X, y, cv=cv, train_sizes=np.linspace(0.1, 1.0, 10)
            )
        else:
            # With feature selection
            pipeline = Pipeline([
                ('selector', selector),
                ('classifier', estimator)
            ])
            train_sizes, train_scores, val_scores = learning_curve(
                pipeline, X, y, cv=cv, train_sizes=np.linspace(0.1, 1.0, 10)
            )
        
        # Calculate means and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot learning curves
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.title(name)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Prepare feature selectors for learning curves
selectors_for_curves = {
    'Original Features': None,
    'Variance Threshold': VarianceThreshold(threshold=0.1),
    'Univariate k=10': SelectKBest(f_classif, k=10),
    'RFE k=10': RFE(RandomForestClassifier(n_estimators=20, random_state=42), n_features_to_select=10),
    'Random Forest': SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=42)),
    'LASSO': SelectFromModel(LassoCV(cv=3, random_state=42, max_iter=1000))
}

plot_learning_curves_feature_selection(X_train_scaled, y_train, selectors_for_curves)
```

## Advanced Feature Selection Techniques

### Genetic Algorithm Feature Selection
```python
def genetic_algorithm_feature_selection(X, y, population_size=50, generations=20, 
                                      mutation_rate=0.1, crossover_rate=0.7):
    """Feature selection using genetic algorithm"""
    
    from sklearn.model_selection import cross_val_score
    
    n_features = X.shape[1]
    
    # Initialize population (random binary arrays)
    population = np.random.randint(0, 2, (population_size, n_features))
    
    def fitness_function(individual):
        """Evaluate fitness of an individual (feature subset)"""
        if np.sum(individual) == 0:  # No features selected
            return 0
        
        X_subset = X[:, individual.astype(bool)]
        clf = RandomForestClassifier(n_estimators=20, random_state=42)
        
        # Use cross-validation for fitness
        scores = cross_val_score(clf, X_subset, y, cv=3, scoring='accuracy')
        
        # Fitness = accuracy - penalty for too many features
        penalty = 0.001 * np.sum(individual) / n_features
        return np.mean(scores) - penalty
    
    # Evolution
    best_fitness_history = []
    
    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = np.array([fitness_function(ind) for ind in population])
        
        # Track best fitness
        best_fitness = np.max(fitness_scores)
        best_fitness_history.append(best_fitness)
        
        if generation % 5 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness:.4f}")
        
        # Selection (tournament selection)
        new_population = []
        for _ in range(population_size):
            # Tournament selection
            tournament_indices = np.random.choice(population_size, size=3, replace=False)
            tournament_fitness = fitness_scores[tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            new_population.append(population[winner_idx].copy())
        
        population = np.array(new_population)
        
        # Crossover
        for i in range(0, population_size-1, 2):
            if np.random.random() < crossover_rate:
                # Single-point crossover
                crossover_point = np.random.randint(1, n_features)
                temp = population[i, crossover_point:].copy()
                population[i, crossover_point:] = population[i+1, crossover_point:]
                population[i+1, crossover_point:] = temp
        
        # Mutation
        for i in range(population_size):
            for j in range(n_features):
                if np.random.random() < mutation_rate:
                    population[i, j] = 1 - population[i, j]
    
    # Get best individual
    final_fitness = np.array([fitness_function(ind) for ind in population])
    best_individual = population[np.argmax(final_fitness)]
    selected_features = np.array(feature_names)[best_individual.astype(bool)]
    
    print(f"\nGenetic Algorithm Feature Selection Results:")
    print(f"Best fitness: {np.max(final_fitness):.4f}")
    print(f"Features selected: {np.sum(best_individual)}")
    print(f"Selected features: {list(selected_features)}")
    
    # Plot fitness evolution
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness_history)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Genetic Algorithm Evolution')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return best_individual, selected_features, best_fitness_history

# Run genetic algorithm (uncomment to run - takes time)
# best_genes, ga_features, fitness_history = genetic_algorithm_feature_selection(
#     X_train_scaled, y_train, population_size=30, generations=15
# )
```

### Mutual Information Feature Selection
```python
def mutual_information_analysis(X, y, feature_names):
    """Analyze features using mutual information"""
    
    from sklearn.feature_selection import mutual_info_classif
    
    # Calculate mutual information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    # Create MI DataFrame
    mi_df = pd.DataFrame({
        'Feature': feature_names,
        'MI_Score': mi_scores
    }).sort_values('MI_Score', ascending=False)
    
    print("Mutual Information Analysis:")
    print("=" * 40)
    print(f"{'Rank':<4} {'Feature':<25} {'MI Score':<10}")
    print("-" * 40)
    
    for i, row in mi_df.head(15).iterrows():
        print(f"{mi_df.index.get_loc(i)+1:<4} {row['Feature']:<25} {row['MI_Score']:<10.4f}")
    
    # Visualize MI scores
    plt.figure(figsize=(12, 8))
    top_features = mi_df.head(20)
    
    plt.barh(range(len(top_features)), top_features['MI_Score'])
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Mutual Information Score')
    plt.title('Feature Mutual Information with Target')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return mi_df

mi_analysis = mutual_information_analysis(X_train_scaled, y_train, feature_names)
```

### Stability Selection
```python
def stability_selection(X, y, feature_names, n_bootstrap=100, threshold=0.6):
    """Perform stability selection using bootstrap sampling"""
    
    from sklearn.utils import resample
    
    n_features = X.shape[1]
    selection_counts = np.zeros(n_features)
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        X_boot, y_boot = resample(X, y, random_state=i)
        
        # Feature selection on bootstrap sample
        selector = SelectKBest(f_classif, k=min(10, n_features//2))
        selector.fit(X_boot, y_boot)
        
        # Count selections
        selection_counts[selector.get_support()] += 1
    
    # Calculate selection probabilities
    selection_probs = selection_counts / n_bootstrap
    
    # Select stable features
    stable_features_mask = selection_probs >= threshold
    stable_features = np.array(feature_names)[stable_features_mask]
    
    # Create stability DataFrame
    stability_df = pd.DataFrame({
        'Feature': feature_names,
        'Selection_Probability': selection_probs,
        'Stable': stable_features_mask
    }).sort_values('Selection_Probability', ascending=False)
    
    print(f"Stability Selection (threshold={threshold}):")
    print(f"Bootstrap samples: {n_bootstrap}")
    print(f"Stable features: {np.sum(stable_features_mask)}")
    
    print(f"\nTop features by stability:")
    for _, row in stability_df.head(15).iterrows():
        status = "✓" if row['Stable'] else " "
        print(f"{status} {row['Feature']:<25}: {row['Selection_Probability']:.3f}")
    
    # Visualize stability
    plt.figure(figsize=(12, 8))
    top_features = stability_df.head(20)
    colors = ['green' if stable else 'lightgray' for stable in top_features['Stable']]
    
    plt.barh(range(len(top_features)), top_features['Selection_Probability'], color=colors)
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Selection Probability')
    plt.title(f'Feature Stability Selection (threshold={threshold})')
    plt.axvline(x=threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold={threshold}')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return stability_df, stable_features

stability_df, stable_features = stability_selection(X_train_scaled, y_train, feature_names)
```

## Feature Selection for Different Data Types

### Text Data Feature Selection
```python
def text_feature_selection_example():
    """Feature selection for text data"""
    
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import chi2
    
    # Load sample text data
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_text = vectorizer.fit_transform(newsgroups_train.data)
    y_text = newsgroups_train.target
    
    feature_names_text = vectorizer.get_feature_names_out()
    
    print("Text Feature Selection Example:")
    print(f"Original features: {X_text.shape[1]}")
    print(f"Samples: {X_text.shape[0]}")
    
    # Chi-square test for text features
    chi2_scores, p_values = chi2(X_text, y_text)
    
    # Select top features
    k_best = 100
    selector = SelectKBest(chi2, k=k_best)
    X_text_selected = selector.fit_transform(X_text, y_text)
    
    selected_features_text = feature_names_text[selector.get_support()]
    
    print(f"Features after chi-square selection: {X_text_selected.shape[1]}")
    
    # Show top features
    feature_scores_text = list(zip(feature_names_text, chi2_scores))
    feature_scores_text.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 20 features by chi-square score:")
    for i, (feature, score) in enumerate(feature_scores_text[:20]):
        print(f"{i+1:2d}. {feature:<15}: {score:8.2f}")
    
    return X_text_selected, selected_features_text

# Uncomment to run text example
# X_text_sel, text_features = text_feature_selection_example()
```

### Time Series Feature Selection
```python
def time_series_feature_selection():
    """Feature selection considerations for time series data"""
    
    print("Time Series Feature Selection Considerations:")
    print("=" * 50)
    
    considerations = {
        "Temporal Dependencies": [
            "Features may have temporal correlations",
            "Standard cross-validation may not be appropriate",
            "Use time series cross-validation",
            "Consider lag features and moving averages"
        ],
        
        "Stationarity": [
            "Check for stationarity in features",
            "Apply differencing if needed",
            "Consider seasonal decomposition",
            "Remove trend components if necessary"
        ],
        
        "Lag Selection": [
            "Select optimal lag features",
            "Use autocorrelation function (ACF)",
            "Consider partial autocorrelation (PACF)",
            "Avoid using future information"
        ],
        
        "Multicollinearity": [
            "Lag features often highly correlated",
            "Use regularization methods",
            "Consider principal component analysis",
            "Apply variance inflation factor (VIF)"
        ]
    }
    
    for category, points in considerations.items():
        print(f"\n{category}:")
        for point in points:
            print(f"  • {point}")

time_series_feature_selection()
```

## Best Practices and Guidelines

### Feature Selection Workflow
```python
def feature_selection_workflow_guide():
    """Complete workflow for feature selection"""
    
    workflow = {
        "1. Data Understanding": [
            "Analyze feature distributions",
            "Check for missing values",
            "Identify feature types (numerical, categorical)",
            "Understand domain knowledge"
        ],
        
        "2. Initial Cleaning": [
            "Remove features with too many missing values",
            "Handle duplicate features",
            "Remove constant or quasi-constant features",
            "Apply variance threshold"
        ],
        
        "3. Correlation Analysis": [
            "Calculate correlation matrix",
            "Identify highly correlated features",
            "Remove redundant features",
            "Consider feature interactions"
        ],
        
        "4. Statistical Testing": [
            "Apply univariate statistical tests",
            "Use appropriate test for data type",
            "Consider multiple testing correction",
            "Select top k features"
        ],
        
        "5. Model-based Selection": [
            "Use embedded methods (LASSO, tree importance)",
            "Apply wrapper methods (RFE)",
            "Compare different algorithms",
            "Consider computational cost"
        ],
        
        "6. Stability Assessment": [
            "Use bootstrap sampling",
            "Check feature selection stability",
            "Validate on multiple datasets",
            "Consider ensemble methods"
        ],
        
        "7. Final Evaluation": [
            "Compare performance metrics",
            "Assess overfitting reduction",
            "Evaluate interpretability",
            "Document selected features"
        ]
    }
    
    print("Feature Selection Workflow:")
    print("=" * 40)
    
    for step, actions in workflow.items():
        print(f"\n{step}")
        for action in actions:
            print(f"  ✓ {action}")

feature_selection_workflow_guide()
```

### Common Pitfalls and Solutions
```python
def feature_selection_pitfalls():
    """Common pitfalls in feature selection and how to avoid them"""
    
    pitfalls = {
        "Data Leakage": {
            "Problem": "Using future information or target-derived features",
            "Solution": "Ensure temporal order, avoid target encoding in selection",
            "Example": "Using statistics computed on entire dataset including test set"
        },
        
        "Selection Bias": {
            "Problem": "Selecting features on entire dataset before splitting",
            "Solution": "Perform feature selection within cross-validation",
            "Example": "High performance due to overfitting to entire dataset"
        },
        
        "Ignoring Domain Knowledge": {
            "Problem": "Purely algorithmic selection ignoring expert knowledge",
            "Solution": "Combine algorithmic methods with domain expertise",
            "Example": "Removing medically important features with low statistical significance"
        },
        
        "Multiple Testing": {
            "Problem": "Not correcting for multiple hypothesis testing",
            "Solution": "Apply Bonferroni or FDR correction",
            "Example": "False discovery due to testing many features simultaneously"
        },
        
        "Stability Issues": {
            "Problem": "Selected features change dramatically with small data changes",
            "Solution": "Use stability selection and ensemble methods",
            "Example": "Different feature sets selected on similar datasets"
        },
        
        "Evaluation Methodology": {
            "Problem": "Inappropriate evaluation of feature selection performance",
            "Solution": "Use nested cross-validation for unbiased estimates",
            "Example": "Overestimating performance due to selection bias"
        }
    }
    
    print("Feature Selection Pitfalls and Solutions:")
    print("=" * 50)
    
    for pitfall, details in pitfalls.items():
        print(f"\n{pitfall}:")
        print(f"  Problem: {details['Problem']}")
        print(f"  Solution: {details['Solution']}")
        print(f"  Example: {details['Example']}")

feature_selection_pitfalls()
```

### Performance vs Complexity Trade-offs
```python
def analyze_performance_complexity_tradeoff():
    """Analyze trade-offs between performance and model complexity"""
    
    # Simulate performance vs number of features
    n_features_range = range(1, 31)
    
    # Simulated performance curves
    performance_scores = []
    complexity_scores = []
    
    for n_feat in n_features_range:
        # Train model with top n features
        selector = SelectKBest(f_classif, k=n_feat)
        X_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_selected, y_train)
        
        # Performance
        test_score = clf.score(X_test_selected, y_test)
        performance_scores.append(test_score)
        
        # Complexity (simplified as number of features)
        complexity_scores.append(n_feat)
    
    # Find optimal point (best performance with fewest features)
    efficiency_scores = np.array(performance_scores) / np.array(complexity_scores)
    optimal_idx = np.argmax(efficiency_scores)
    optimal_features = n_features_range[optimal_idx]
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Performance vs number of features
    axes[0].plot(n_features_range, performance_scores, 'b-o')
    axes[0].axvline(x=optimal_features, color='red', linestyle='--', 
                   label=f'Optimal: {optimal_features} features')
    axes[0].set_xlabel('Number of Features')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Performance vs Number of Features')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Efficiency score
    axes[1].plot(n_features_range, efficiency_scores, 'g-o')
    axes[1].axvline(x=optimal_features, color='red', linestyle='--', 
                   label=f'Maximum efficiency')
    axes[1].set_xlabel('Number of Features')
    axes[1].set_ylabel('Efficiency (Performance/Complexity)')
    axes[1].set_title('Feature Selection Efficiency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Performance improvement
    baseline_performance = performance_scores[0]  # 1 feature
    improvement = [(score - baseline_performance) for score in performance_scores]
    
    axes[2].plot(n_features_range, improvement, 'purple', marker='o')
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[2].set_xlabel('Number of Features')
    axes[2].set_ylabel('Performance Improvement')
    axes[2].set_title('Marginal Performance Gain')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Optimal number of features: {optimal_features}")
    print(f"Performance at optimal point: {performance_scores[optimal_idx]:.4f}")
    print(f"Efficiency score: {efficiency_scores[optimal_idx]:.6f}")
    
    return optimal_features, performance_scores, efficiency_scores

optimal_n_features, perf_scores, eff_scores = analyze_performance_complexity_tradeoff()
```

## Learning Objectives
- [ ] **Understand feature selection importance**: Recognize when and why to apply feature selection
- [ ] **Master filter methods**: Apply statistical tests and correlation analysis for feature selection
- [ ] **Implement wrapper methods**: Use RFE and sequential selection techniques effectively
- [ ] **Apply embedded methods**: Leverage model-based feature importance and regularization
- [ ] **Compare selection methods**: Evaluate different approaches and choose appropriate techniques
- [ ] **Handle different data types**: Apply appropriate selection methods for text, time series, and other data
- [ ] **Avoid common pitfalls**: Prevent data leakage and selection bias in feature selection
- [ ] **Evaluate selection stability**: Assess robustness of selected feature sets
- [ ] **Optimize performance-complexity trade-offs**: Balance model performance with interpretability
- [ ] **Create selection pipelines**: Build automated feature selection workflows

## Practice Exercises
1. Apply different univariate selection methods and compare results
2. Implement RFE with different base estimators and analyze feature rankings
3. Use LASSO regularization for feature selection and interpret coefficients
4. Compare filter, wrapper, and embedded methods on the same dataset
5. Implement stability selection and analyze feature selection robustness
6. Create a feature selection pipeline with proper cross-validation
7. Apply feature selection to text data using chi-square tests
8. Analyze the bias-variance trade-off in feature selection
9. Implement genetic algorithm for feature selection optimization
10. Build an automated feature selection system with multiple methods and evaluation metrics
