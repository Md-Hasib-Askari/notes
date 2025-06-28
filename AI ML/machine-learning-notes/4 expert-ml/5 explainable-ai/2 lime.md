# LIME (Local Interpretable Model-Agnostic Explanations)

## Learning Objectives
- Understand the fundamentals of LIME and local interpretability
- Implement LIME for various data types (tabular, text, images)
- Apply LIME to real-world model interpretation tasks
- Compare LIME with other explainability techniques
- Integrate LIME into ML pipelines for production use

## 1. Introduction to LIME

### What is LIME?
LIME is a technique that explains individual predictions of any classifier in an interpretable and faithful manner by learning an interpretable model locally around the prediction.

### Key Concepts
- **Local Surrogate Models**: Simple models that approximate complex models locally
- **Model-Agnostic**: Works with any machine learning model
- **Perturbation-Based**: Creates explanations by perturbing input features
- **Faithful**: Explanations should accurately reflect the model's local behavior

### Core Algorithm
1. Perturb the instance around its neighborhood
2. Get predictions for perturbed samples
3. Weight samples by proximity to original instance
4. Train interpretable model on weighted dataset
5. Extract feature importance from surrogate model

## 2. LIME Implementation

### Installation and Setup

```python
# Install LIME
!pip install lime

# Essential imports
import lime
import lime.lime_tabular
import lime.lime_text
import lime.lime_image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
```

### Tabular Data Example

```python
# Load and prepare data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Create LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    mode='classification'
)

# Generate explanation for a single instance
instance_idx = 0
instance = X_test[instance_idx]

explanation = explainer.explain_instance(
    instance,
    rf.predict_proba,
    num_features=4
)

# Visualize explanation
explanation.show_in_notebook(show_table=True)

# Get explanation as list
print("Feature contributions:")
for feature, importance in explanation.as_list():
    print(f"{feature}: {importance:.4f}")
```

### Advanced Tabular Configuration

```python
# Custom explainer with advanced options
explainer_advanced = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    mode='classification',
    discretize_continuous=True,
    kernel_width=3,
    verbose=False,
    feature_selection='auto'
)

# Explain multiple instances
explanations = []
for i in range(5):
    exp = explainer_advanced.explain_instance(
        X_test[i],
        rf.predict_proba,
        num_features=4,
        num_samples=1000
    )
    explanations.append(exp)
    
# Analyze feature importance across instances
feature_importance_matrix = []
for exp in explanations:
    importance_dict = dict(exp.as_list())
    feature_importance_matrix.append([
        importance_dict.get(feature, 0) for feature in iris.feature_names
    ])

importance_df = pd.DataFrame(
    feature_importance_matrix,
    columns=iris.feature_names
)
print("Feature importance across instances:")
print(importance_df)
```

## 3. Text Data with LIME

### Text Classification Example

```python
# Prepare text data
categories = ['alt.atheism', 'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(
    subset='train',
    categories=categories,
    remove=('headers', 'footers', 'quotes')
)
newsgroups_test = fetch_20newsgroups(
    subset='test',
    categories=categories,
    remove=('headers', 'footers', 'quotes')
)

# Create text classification pipeline
text_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

text_pipeline.fit(newsgroups_train.data, newsgroups_train.target)

# Create text explainer
text_explainer = lime.lime_text.LimeTextExplainer(
    class_names=newsgroups_train.target_names
)

# Explain a text instance
idx = 0
text_instance = newsgroups_test.data[idx]

text_explanation = text_explainer.explain_instance(
    text_instance,
    text_pipeline.predict_proba,
    num_features=10
)

print("Text explanation:")
print(text_explanation.as_list())

# Visualize in notebook
text_explanation.show_in_notebook(text=True)
```

### Custom Text Preprocessing

```python
def custom_text_pipeline_func(texts):
    """Custom function for text pipeline prediction"""
    return text_pipeline.predict_proba(texts)

# Advanced text explanation
text_explanation_advanced = text_explainer.explain_instance(
    text_instance,
    custom_text_pipeline_func,
    num_features=15,
    num_samples=1000,
    labels=[1]  # Explain only positive class
)

# Extract important words
important_words = text_explanation_advanced.as_list()
positive_words = [word for word, score in important_words if score > 0]
negative_words = [word for word, score in important_words if score < 0]

print(f"Positive words: {positive_words}")
print(f"Negative words: {negative_words}")
```

## 4. Image Data with LIME

### Image Classification Example

```python
from skimage.segmentation import mark_boundaries
from PIL import Image
import requests
from io import BytesIO

# Load pre-trained model (example with a simple model)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits

# Use digits dataset for demonstration
digits = load_digits()
X_digits, y_digits = digits.data, digits.target
X_train_dig, X_test_dig, y_train_dig, y_test_dig = train_test_split(
    X_digits, y_digits, test_size=0.2, random_state=42
)

# Train simple model on flattened images
rf_image = RandomForestClassifier(n_estimators=100, random_state=42)
rf_image.fit(X_train_dig, y_train_dig)

# Reshape for image format
image_instance = X_test_dig[0].reshape(8, 8)

# Create image explainer
image_explainer = lime.lime_image.LimeImageExplainer()

# Explain image prediction
def predict_fn(images):
    """Prediction function for image explainer"""
    flat_images = images.reshape(images.shape[0], -1)
    return rf_image.predict_proba(flat_images)

image_explanation = image_explainer.explain_instance(
    image_instance,
    predict_fn,
    top_labels=3,
    hide_color=0,
    num_samples=1000
)

# Visualize explanation
temp, mask = image_explanation.get_image_and_mask(
    image_explanation.top_labels[0],
    positive_only=True,
    num_features=5,
    hide_rest=False
)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image_instance, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(mark_boundaries(temp, mask))
plt.title('Explanation')

plt.subplot(1, 3, 3)
plt.imshow(mask, cmap='gray')
plt.title('Mask')
plt.show()
```

## 5. Production Integration

### LIME Service Class

```python
class LIMEExplainerService:
    """Production-ready LIME explainer service"""
    
    def __init__(self, model, training_data, feature_names=None, mode='classification'):
        self.model = model
        self.training_data = training_data
        self.feature_names = feature_names
        self.mode = mode
        self._setup_explainers()
    
    def _setup_explainers(self):
        """Initialize LIME explainers"""
        if hasattr(self.training_data, 'shape') and len(self.training_data.shape) == 2:
            # Tabular data
            self.tabular_explainer = lime.lime_tabular.LimeTabularExplainer(
                self.training_data,
                feature_names=self.feature_names,
                mode=self.mode,
                discretize_continuous=True
            )
        
        # Text explainer
        self.text_explainer = lime.lime_text.LimeTextExplainer(
            mode=self.mode
        )
        
        # Image explainer
        self.image_explainer = lime.lime_image.LimeImageExplainer()
    
    def explain_tabular(self, instance, num_features=10, num_samples=1000):
        """Explain tabular instance"""
        try:
            explanation = self.tabular_explainer.explain_instance(
                instance,
                self.model.predict_proba,
                num_features=num_features,
                num_samples=num_samples
            )
            return {
                'success': True,
                'explanation': explanation.as_list(),
                'prediction_proba': self.model.predict_proba([instance])[0].tolist()
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def explain_text(self, text, num_features=10):
        """Explain text instance"""
        try:
            explanation = self.text_explainer.explain_instance(
                text,
                self.model.predict_proba,
                num_features=num_features
            )
            return {
                'success': True,
                'explanation': explanation.as_list(),
                'prediction_proba': self.model.predict_proba([text])[0].tolist()
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def batch_explain(self, instances, data_type='tabular', **kwargs):
        """Explain multiple instances"""
        explanations = []
        for instance in instances:
            if data_type == 'tabular':
                exp = self.explain_tabular(instance, **kwargs)
            elif data_type == 'text':
                exp = self.explain_text(instance, **kwargs)
            explanations.append(exp)
        return explanations

# Usage example
service = LIMEExplainerService(
    model=rf,
    training_data=X_train,
    feature_names=iris.feature_names,
    mode='classification'
)

# Explain single instance
result = service.explain_tabular(X_test[0])
print("Explanation result:", result)
```

### REST API Integration

```python
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model and setup explainer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('training_data.pkl', 'rb') as f:
    training_data = pickle.load(f)

explainer_service = LIMEExplainerService(model, training_data)

@app.route('/explain', methods=['POST'])
def explain_prediction():
    """API endpoint for LIME explanations"""
    try:
        data = request.json
        instance = data['instance']
        data_type = data.get('data_type', 'tabular')
        num_features = data.get('num_features', 10)
        
        if data_type == 'tabular':
            result = explainer_service.explain_tabular(
                instance, num_features=num_features
            )
        elif data_type == 'text':
            result = explainer_service.explain_text(
                instance, num_features=num_features
            )
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/batch_explain', methods=['POST'])
def batch_explain():
    """API endpoint for batch explanations"""
    try:
        data = request.json
        instances = data['instances']
        data_type = data.get('data_type', 'tabular')
        
        results = explainer_service.batch_explain(instances, data_type)
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
```

## 6. Best Practices and Considerations

### Performance Optimization

```python
# Efficient explanation generation
def optimized_explain(explainer, instance, model, num_samples=500):
    """Optimized explanation with caching"""
    explanation = explainer.explain_instance(
        instance,
        model.predict_proba,
        num_samples=num_samples,  # Reduced for speed
        num_features=10
    )
    return explanation

# Parallel explanation generation
from multiprocessing import Pool
import functools

def parallel_explain_batch(instances, explainer, model, n_jobs=4):
    """Generate explanations in parallel"""
    explain_func = functools.partial(
        optimized_explain, explainer, model=model
    )
    
    with Pool(n_jobs) as pool:
        explanations = pool.map(explain_func, instances)
    
    return explanations
```

### Explanation Validation

```python
def validate_explanation_fidelity(explainer, instance, model, num_trials=10):
    """Validate explanation consistency"""
    explanations = []
    for _ in range(num_trials):
        exp = explainer.explain_instance(
            instance,
            model.predict_proba,
            num_samples=1000
        )
        explanations.append(dict(exp.as_list()))
    
    # Calculate feature importance stability
    all_features = set()
    for exp in explanations:
        all_features.update(exp.keys())
    
    stability_scores = {}
    for feature in all_features:
        importances = [exp.get(feature, 0) for exp in explanations]
        stability_scores[feature] = np.std(importances)
    
    return stability_scores

# Usage
stability = validate_explanation_fidelity(explainer, X_test[0], rf)
print("Feature stability scores:", stability)
```

## 7. Comparison with Other Methods

### LIME vs SHAP Comparison

```python
import shap

# LIME explanation
lime_exp = explainer.explain_instance(
    X_test[0],
    rf.predict_proba,
    num_features=4
)

# SHAP explanation
shap_explainer = shap.TreeExplainer(rf)
shap_values = shap_explainer.shap_values(X_test[0:1])

# Compare feature importances
lime_importances = dict(lime_exp.as_list())
shap_importances = dict(zip(iris.feature_names, shap_values[1][0]))

comparison_df = pd.DataFrame({
    'LIME': [lime_importances.get(f, 0) for f in iris.feature_names],
    'SHAP': [shap_importances.get(f, 0) for f in iris.feature_names]
}, index=iris.feature_names)

print("LIME vs SHAP comparison:")
print(comparison_df)

# Correlation between methods
correlation = comparison_df.corr().iloc[0, 1]
print(f"Correlation between LIME and SHAP: {correlation:.4f}")
```

## 8. Advanced Use Cases

### Custom Distance Functions

```python
from lime.lime_tabular import LimeTabularExplainer

class CustomLimeExplainer(LimeTabularExplainer):
    """LIME explainer with custom distance function"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def kernel(self, d, kernel_width):
        """Custom kernel function"""
        return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

# Usage with custom kernel
custom_explainer = CustomLimeExplainer(
    X_train,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    mode='classification'
)
```

### Explanation Aggregation

```python
def aggregate_explanations(explanations, method='mean'):
    """Aggregate multiple explanations"""
    all_features = set()
    for exp in explanations:
        for feature, _ in exp.as_list():
            all_features.add(feature)
    
    aggregated = {}
    for feature in all_features:
        importances = []
        for exp in explanations:
            exp_dict = dict(exp.as_list())
            importances.append(exp_dict.get(feature, 0))
        
        if method == 'mean':
            aggregated[feature] = np.mean(importances)
        elif method == 'median':
            aggregated[feature] = np.median(importances)
        elif method == 'std':
            aggregated[feature] = np.std(importances)
    
    return aggregated

# Generate multiple explanations for the same instance
multiple_explanations = []
for _ in range(10):
    exp = explainer.explain_instance(
        X_test[0],
        rf.predict_proba,
        num_samples=500
    )
    multiple_explanations.append(exp)

# Aggregate explanations
aggregated = aggregate_explanations(multiple_explanations, method='mean')
print("Aggregated feature importances:", aggregated)
```

## Summary

LIME provides local, interpretable explanations for any machine learning model through:

1. **Local Surrogate Models**: Simple models that approximate complex models locally
2. **Model Agnostic**: Works with any ML algorithm
3. **Multiple Data Types**: Supports tabular, text, and image data
4. **Perturbation-Based**: Creates explanations by analyzing feature perturbations
5. **Production Ready**: Can be integrated into ML pipelines and APIs

### Key Advantages
- Easy to understand and implement
- Works with any model type
- Provides local explanations for individual predictions
- Supports multiple data modalities

### Limitations
- Explanations can be unstable across runs
- Local explanations may not represent global model behavior
- Computational overhead for real-time applications
- Choice of perturbation strategy affects explanation quality

LIME is an essential tool for understanding individual predictions and building trust in machine learning models.