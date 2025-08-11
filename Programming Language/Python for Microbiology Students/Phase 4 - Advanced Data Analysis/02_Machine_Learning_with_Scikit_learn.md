# Machine Learning with Scikit-learn

Machine learning applications in microbiology include bacterial identification, antimicrobial resistance prediction, and microbiome analysis.

## Introduction to Machine Learning in Microbiology

### Setup and Data Preparation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
```

### Generate Sample Microbiology Data

```python
def generate_bacterial_classification_data(n_samples=500):
    """Generate synthetic bacterial classification dataset"""
    
    # Features that might distinguish bacterial species
    np.random.seed(42)
    
    # Generate features for different bacterial species
    species = []
    features = []
    
    # E. coli characteristics
    for _ in range(150):
        gram_stain = 0  # Gram-negative
        catalase = 1
        oxidase = 0
        motility = 1
        glucose_fermentation = 1
        lactose_fermentation = np.random.choice([0, 1], p=[0.2, 0.8])  # Usually positive
        
        # Biochemical test results with some noise
        indole = np.random.choice([0, 1], p=[0.1, 0.9])  # Usually positive
        voges_proskauer = np.random.choice([0, 1], p=[0.9, 0.1])  # Usually negative
        citrate = np.random.choice([0, 1], p=[0.8, 0.2])  # Usually negative
        
        features.append([gram_stain, catalase, oxidase, motility, glucose_fermentation, 
                        lactose_fermentation, indole, voges_proskauer, citrate])
        species.append('E_coli')
    
    # S. aureus characteristics
    for _ in range(150):
        gram_stain = 1  # Gram-positive
        catalase = 1
        oxidase = 0
        motility = 0
        glucose_fermentation = 1
        lactose_fermentation = 0
        
        # Biochemical tests
        coagulase = np.random.choice([0, 1], p=[0.1, 0.9])  # Usually positive
        dnase = np.random.choice([0, 1], p=[0.2, 0.8])  # Usually positive
        mannitol = np.random.choice([0, 1], p=[0.2, 0.8])  # Usually positive
        
        features.append([gram_stain, catalase, oxidase, motility, glucose_fermentation, 
                        lactose_fermentation, coagulase, dnase, mannitol])
        species.append('S_aureus')
    
    # P. aeruginosa characteristics
    for _ in range(100):
        gram_stain = 0  # Gram-negative
        catalase = 1
        oxidase = 1  # Key differentiator
        motility = 1
        glucose_fermentation = 0  # Non-fermenter
        lactose_fermentation = 0
        
        # Additional tests
        pyocyanin = np.random.choice([0, 1], p=[0.3, 0.7])  # Usually positive
        fluorescein = np.random.choice([0, 1], p=[0.2, 0.8])  # Usually positive
        growth_42c = np.random.choice([0, 1], p=[0.1, 0.9])  # Usually positive
        
        features.append([gram_stain, catalase, oxidase, motility, glucose_fermentation, 
                        lactose_fermentation, pyocyanin, fluorescein, growth_42c])
        species.append('P_aeruginosa')
    
    # B. subtilis characteristics
    for _ in range(100):
        gram_stain = 1  # Gram-positive
        catalase = 1
        oxidase = 0
        motility = 1
        glucose_fermentation = 1
        lactose_fermentation = 0
        
        # Spore formation tests
        spore_formation = np.random.choice([0, 1], p=[0.1, 0.9])  # Usually positive
        starch_hydrolysis = np.random.choice([0, 1], p=[0.2, 0.8])  # Usually positive
        lecithinase = np.random.choice([0, 1], p=[0.3, 0.7])  # Variable
        
        features.append([gram_stain, catalase, oxidase, motility, glucose_fermentation, 
                        lactose_fermentation, spore_formation, starch_hydrolysis, lecithinase])
        species.append('B_subtilis')
    
    # Create DataFrame
    feature_names = ['Gram_Stain', 'Catalase', 'Oxidase', 'Motility', 'Glucose_Fermentation',
                     'Lactose_Fermentation', 'Test_7', 'Test_8', 'Test_9']
    
    df = pd.DataFrame(features, columns=feature_names)
    df['Species'] = species
    
    return df

# Generate dataset
bacterial_data = generate_bacterial_classification_data()
print("Bacterial classification dataset created:")
print(f"Shape: {bacterial_data.shape}")
print(f"Species distribution:\n{bacterial_data['Species'].value_counts()}")
print(f"\nFirst few rows:")
print(bacterial_data.head())
```

## Classification Algorithms

### Data Preprocessing

```python
def preprocess_data(df, target_column='Species'):
    """Preprocess data for machine learning"""
    
    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Encode categorical target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    
    # Scale features (important for some algorithms)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, label_encoder, scaler

# Preprocess the data
X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, label_encoder, scaler = preprocess_data(bacterial_data)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"Classes: {label_encoder.classes_}")
```

### Random Forest Classifier

```python
def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate Random Forest classifier"""
    
    # Initialize and train model
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5
    )
    
    rf_classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_classifier.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_classifier.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Random Forest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nTop 5 Most Important Features:")
    print(feature_importance.head())
    
    return rf_classifier, y_pred, feature_importance

# Train Random Forest
rf_model, rf_predictions, rf_importance = train_random_forest(X_train, y_train, X_test, y_test)
```

### Support Vector Machine

```python
def train_svm(X_train_scaled, y_train, X_test_scaled, y_test):
    """Train and evaluate SVM classifier"""
    
    # Initialize and train SVM
    svm_classifier = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=42
    )
    
    svm_classifier.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = svm_classifier.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print("SVM Results:")
    print(f"Accuracy: {accuracy:.4f}")
    
    return svm_classifier, y_pred

# Train SVM (using scaled data)
svm_model, svm_predictions = train_svm(X_train_scaled, y_train, X_test_scaled, y_test)
```

### K-Nearest Neighbors

```python
def train_knn(X_train_scaled, y_train, X_test_scaled, y_test, k=5):
    """Train and evaluate KNN classifier"""
    
    # Initialize and train KNN
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = knn_classifier.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"KNN (k={k}) Results:")
    print(f"Accuracy: {accuracy:.4f}")
    
    return knn_classifier, y_pred

# Train KNN
knn_model, knn_predictions = train_knn(X_train_scaled, y_train, X_test_scaled, y_test)
```

### Model Comparison and Evaluation

```python
def comprehensive_evaluation(y_test, predictions_dict, label_encoder):
    """Comprehensive model evaluation and comparison"""
    
    results = {}
    
    for model_name, y_pred in predictions_dict.items():
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(
            y_test, y_pred, 
            target_names=label_encoder.classes_,
            output_dict=True
        )
        
        results[model_name] = {
            'accuracy': accuracy,
            'classification_report': class_report
        }
        
        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"{'='*50}")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    return results

# Compare all models
predictions_dict = {
    'Random Forest': rf_predictions,
    'SVM': svm_predictions,
    'KNN': knn_predictions
}

evaluation_results = comprehensive_evaluation(y_test, predictions_dict, label_encoder)
```

### Confusion Matrix Visualization

```python
def plot_confusion_matrices(y_test, predictions_dict, label_encoder):
    """Plot confusion matrices for all models"""
    
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(15, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            ax=axes[i]
        )
        
        axes[i].set_title(f'{model_name}\nConfusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()

# Plot confusion matrices
plot_confusion_matrices(y_test, predictions_dict, label_encoder)
```

## Clustering for Community Analysis

### K-Means Clustering

```python
def perform_kmeans_clustering(data, n_clusters=4):
    """Perform K-means clustering on bacterial data"""
    
    # Prepare data (exclude species column)
    features = data.drop('Species', axis=1)
    
    # Scale the data
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    # Add cluster labels to original data
    clustered_data = data.copy()
    clustered_data['Cluster'] = cluster_labels
    
    # Analyze clusters
    print(f"K-Means Clustering Results (k={n_clusters}):")
    print("\nCluster composition by species:")
    cluster_composition = pd.crosstab(clustered_data['Species'], clustered_data['Cluster'])
    print(cluster_composition)
    
    # Calculate silhouette score
    from sklearn.metrics import silhouette_score
    silhouette_avg = silhouette_score(features_scaled, cluster_labels)
    print(f"\nAverage silhouette score: {silhouette_avg:.4f}")
    
    return kmeans, cluster_labels, features_scaled, scaler

# Perform clustering
kmeans_model, clusters, scaled_features, feature_scaler = perform_kmeans_clustering(bacterial_data)
```

### Finding Optimal Number of Clusters

```python
def find_optimal_clusters(features_scaled, max_k=10):
    """Find optimal number of clusters using elbow method and silhouette analysis"""
    
    from sklearn.metrics import silhouette_score
    
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(features_scaled, cluster_labels))
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Elbow plot
    ax1.plot(k_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True)
    
    # Silhouette plot
    ax2.plot(k_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Find optimal k
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    print(f"Optimal k based on silhouette score: {optimal_k_silhouette}")
    
    return k_range, inertias, silhouette_scores

# Find optimal number of clusters
k_values, inertia_values, silhouette_values = find_optimal_clusters(scaled_features)
```

## Principal Component Analysis (PCA)

### Dimensionality Reduction

```python
def perform_pca_analysis(features_scaled, variance_threshold=0.95):
    """Perform PCA for dimensionality reduction and visualization"""
    
    # Perform PCA
    pca = PCA()
    pca_features = pca.fit_transform(features_scaled)
    
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Find number of components for desired variance
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    print(f"PCA Analysis Results:")
    print(f"Number of components for {variance_threshold*100}% variance: {n_components}")
    print(f"Explained variance ratio by component:")
    for i, var_ratio in enumerate(pca.explained_variance_ratio_[:5]):
        print(f"  PC{i+1}: {var_ratio:.4f} ({var_ratio*100:.2f}%)")
    
    # Plot explained variance
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Component')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
    plt.axhline(y=variance_threshold, color='r', linestyle='--', 
                label=f'{variance_threshold*100}% threshold')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return pca, pca_features

# Perform PCA
pca_model, pca_transformed = perform_pca_analysis(scaled_features)
```

### PCA Visualization

```python
def visualize_pca_results(pca_features, original_data, pca_model):
    """Visualize PCA results with species labels"""
    
    # Create DataFrame with PCA results
    pca_df = pd.DataFrame({
        'PC1': pca_features[:, 0],
        'PC2': pca_features[:, 1],
        'Species': original_data['Species']
    })
    
    # Plot PCA results
    plt.figure(figsize=(12, 8))
    
    # Scatter plot colored by species
    for species in pca_df['Species'].unique():
        species_data = pca_df[pca_df['Species'] == species]
        plt.scatter(species_data['PC1'], species_data['PC2'], 
                   label=species, alpha=0.7, s=50)
    
    plt.xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA: Bacterial Species Separation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return pca_df

# Visualize PCA results
pca_visualization = visualize_pca_results(pca_transformed, bacterial_data, pca_model)
```

## Cross-Validation and Model Selection

### Grid Search for Hyperparameter Tuning

```python
def optimize_random_forest(X_train, y_train):
    """Optimize Random Forest hyperparameters using Grid Search"""
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Initialize Random Forest
    rf = RandomForestClassifier(random_state=42)
    
    # Perform Grid Search with cross-validation
    grid_search = GridSearchCV(
        rf, 
        param_grid, 
        cv=5, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    print("Performing Grid Search for Random Forest optimization...")
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

# Optimize Random Forest
optimized_rf = optimize_random_forest(X_train, y_train)
```

### Cross-Validation Scores

```python
def evaluate_with_cross_validation(models_dict, X_train, y_train, cv=5):
    """Evaluate models using cross-validation"""
    
    results = {}
    
    for model_name, model in models_dict.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        
        results[model_name] = {
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std(),
            'cv_scores': cv_scores
        }
        
        print(f"{model_name}:")
        print(f"  CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  Individual scores: {cv_scores}")
        print()
    
    return results

# Prepare models for cross-validation
models_for_cv = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

# Evaluate with cross-validation
cv_results = evaluate_with_cross_validation(models_for_cv, X_train_scaled, y_train)
```

## Best Practices for Machine Learning in Microbiology

1. **Use appropriate evaluation metrics** for your specific problem
2. **Always validate on independent test sets**
3. **Consider biological relevance** when interpreting results
4. **Handle imbalanced datasets** appropriately
5. **Use cross-validation** for robust model evaluation
6. **Feature selection** can improve model performance
7. **Document preprocessing steps** for reproducibility
8. **Consider ensemble methods** for improved accuracy
