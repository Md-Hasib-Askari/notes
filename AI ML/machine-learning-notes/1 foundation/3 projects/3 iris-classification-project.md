# Iris Classification Project

## Project Overview
Complete end-to-end classification project using the famous Iris dataset. This project demonstrates the full machine learning pipeline from data exploration to model evaluation.

## Dataset Description
The Iris dataset is perfect for beginners:
- **150 samples** (50 per class)
- **4 features**: sepal length, sepal width, petal length, petal width
- **3 classes**: Setosa, Versicolor, Virginica
- **No missing values**
- **Well-balanced classes**

## Step-by-Step Implementation

### 1. Environment Setup
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
```

### 2. Data Loading and Initial Exploration
```python
# Load the dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Basic information
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nClass Distribution:")
print(df['species'].value_counts())
```

### 3. Exploratory Data Analysis

#### Basic Statistics
```python
# Check for missing values
print("Missing Values:")
print(df.isnull().sum())

# Feature correlations
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()
```

#### Data Visualization
```python
# Pairplot to see relationships
plt.figure(figsize=(12, 8))
sns.pairplot(df, hue='species', markers=['o', 's', 'D'])
plt.suptitle('Iris Dataset Pairplot', y=1.02)
plt.show()

# Box plots for each feature
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
features = iris.feature_names

for i, feature in enumerate(features):
    row = i // 2
    col = i % 2
    sns.boxplot(data=df, x='species', y=feature, ax=axes[row, col])
    axes[row, col].set_title(f'{feature} by Species')

plt.tight_layout()
plt.show()

# Distribution plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, feature in enumerate(features):
    row = i // 2
    col = i % 2
    for species in df['species'].unique():
        subset = df[df['species'] == species]
        axes[row, col].hist(subset[feature], alpha=0.7, label=species, bins=15)
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Frequency')
    axes[row, col].set_title(f'{feature} Distribution')
    axes[row, col].legend()

plt.tight_layout()
plt.show()
```

### 4. Data Preprocessing
```python
# Prepare features and target
X = df[iris.feature_names]
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Feature scaling (optional for some algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 5. Model Training and Comparison
```python
# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'SVM': SVC(random_state=42)
}

# Train and evaluate models
results = {}

for name, model in models.items():
    # Use scaled data for SVM and Logistic Regression
    if name in ['SVM', 'Logistic Regression']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

### 6. Model Evaluation and Visualization
```python
# Compare model performances
plt.figure(figsize=(10, 6))
models_names = list(results.keys())
accuracies = list(results.values())

bars = plt.bar(models_names, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
plt.title('Model Comparison - Accuracy Scores')
plt.ylabel('Accuracy')
plt.ylim(0.8, 1.0)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{acc:.3f}', ha='center', va='bottom')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Confusion matrix for best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

if best_model_name in ['SVM', 'Logistic Regression']:
    y_pred_best = best_model.predict(X_test_scaled)
else:
    y_pred_best = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
```

### 7. Feature Importance (for tree-based models)
```python
# Feature importance for Random Forest
rf_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance - Random Forest')
plt.xlabel('Importance')
plt.show()

print("Feature Importance Rankings:")
print(feature_importance)
```

### 8. Making Predictions on New Data
```python
# Example: Predict on new sample
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Setosa-like features

# Use the best model
if best_model_name in ['SVM', 'Logistic Regression']:
    new_sample_scaled = scaler.transform(new_sample)
    prediction = best_model.predict(new_sample_scaled)
    probability = best_model.predict_proba(new_sample_scaled)
else:
    prediction = best_model.predict(new_sample)
    probability = best_model.predict_proba(new_sample)

predicted_species = iris.target_names[prediction[0]]
print(f"\nPrediction for new sample {new_sample[0]}:")
print(f"Predicted species: {predicted_species}")
print(f"Prediction probabilities:")
for i, prob in enumerate(probability[0]):
    print(f"  {iris.target_names[i]}: {prob:.3f}")
```

## Key Insights and Takeaways

### Dataset Characteristics
- **Setosa** is easily separable from other species
- **Petal length and width** are most discriminative features
- **High correlation** between petal features and sepal features
- **No missing values** - clean dataset perfect for beginners

### Model Performance
- Most algorithms perform excellently (>95% accuracy)
- **Random Forest** and **SVM** typically achieve highest accuracy
- **Decision Tree** may overfit but provides interpretability
- **Logistic Regression** offers good baseline performance

### Best Practices Demonstrated
1. **Systematic EDA** before modeling
2. **Multiple algorithm comparison**
3. **Proper train/test splitting**
4. **Feature scaling** when needed
5. **Comprehensive evaluation** with multiple metrics

## Extensions and Next Steps

### Beginner Extensions
- [ ] Try different train/test split ratios
- [ ] Experiment with cross-validation
- [ ] Add more visualization techniques
- [ ] Try different random seeds

### Intermediate Extensions
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Implement k-fold cross-validation
- [ ] Add feature selection techniques
- [ ] Try ensemble methods

### Advanced Extensions
- [ ] Implement custom evaluation metrics
- [ ] Add dimensionality reduction (PCA)
- [ ] Try neural networks
- [ ] Create interactive visualizations

## Learning Objectives
- [x] Load and explore a real dataset
- [x] Perform comprehensive EDA
- [x] Compare multiple ML algorithms
- [x] Evaluate model performance properly
- [x] Visualize results effectively
- [x] Make predictions on new data
- [x] Understand feature importance
- [x] Follow ML best practices

## Common Pitfalls to Avoid
1. **Skipping EDA** - Always explore data first
2. **Data leakage** - Don't use test data for preprocessing decisions
3. **Overfitting** - Use cross-validation for better estimates
4. **Ignoring class balance** - Check target distribution
5. **Wrong scaling** - Scale features for distance-based algorithms

This project provides a solid foundation for understanding the complete machine learning workflow!