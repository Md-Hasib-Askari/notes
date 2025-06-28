# Bias Detection in Machine Learning

## Learning Objectives
- Understand different types of bias in machine learning systems
- Learn to detect and measure bias in datasets and models
- Implement bias detection tools and metrics
- Apply bias mitigation strategies during different phases of ML development
- Design bias monitoring systems for production environments

## 1. Introduction to AI Bias

### What is AI Bias?
AI bias refers to systematic errors or prejudices in machine learning models that result in unfair treatment of certain individuals or groups. These biases can lead to discriminatory outcomes and perpetuate or amplify existing societal inequalities.

### Types of Bias

#### Data-Related Bias
- **Historical Bias**: Past discrimination reflected in training data
- **Representation Bias**: Underrepresentation of certain groups
- **Measurement Bias**: Systematic errors in data collection
- **Evaluation Bias**: Inappropriate benchmarks or metrics

#### Algorithmic Bias
- **Aggregation Bias**: Inappropriate pooling of data across groups
- **Confirmation Bias**: Seeking information that confirms existing beliefs
- **Automation Bias**: Over-reliance on automated systems

#### Deployment Bias
- **Population Shift**: Model applied to different population than training
- **Temporal Shift**: Model performance degrading over time
- **Usage Bias**: Misuse of model outside intended scope

## 2. Bias Detection Framework

### Comprehensive Bias Assessment

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class BiasDetectionFramework:
    """Comprehensive framework for detecting bias in ML systems"""
    
    def __init__(self, data, target_col, sensitive_attributes):
        self.data = data
        self.target_col = target_col
        self.sensitive_attributes = sensitive_attributes
        self.bias_metrics = {}
        
    def analyze_data_bias(self):
        """Analyze bias in the dataset"""
        bias_analysis = {}
        
        # Representation analysis
        bias_analysis['representation'] = self._analyze_representation()
        
        # Correlation analysis
        bias_analysis['correlation'] = self._analyze_correlations()
        
        # Statistical parity
        bias_analysis['statistical_parity'] = self._analyze_statistical_parity()
        
        # Missing data patterns
        bias_analysis['missing_data'] = self._analyze_missing_data()
        
        return bias_analysis
    
    def _analyze_representation(self):
        """Analyze representation of different groups"""
        representation = {}
        
        for attr in self.sensitive_attributes:
            if attr in self.data.columns:
                value_counts = self.data[attr].value_counts()
                proportions = self.data[attr].value_counts(normalize=True)
                
                representation[attr] = {
                    'counts': value_counts.to_dict(),
                    'proportions': proportions.to_dict(),
                    'balance_ratio': proportions.min() / proportions.max()
                }
        
        return representation
    
    def _analyze_correlations(self):
        """Analyze correlations between sensitive attributes and target"""
        correlations = {}
        
        for attr in self.sensitive_attributes:
            if attr in self.data.columns:
                # For categorical variables, use Cramér's V
                if self.data[attr].dtype == 'object' or self.data[attr].nunique() < 10:
                    cramers_v = self._cramers_v(self.data[attr], self.data[self.target_col])
                    correlations[attr] = {
                        'type': 'cramers_v',
                        'value': cramers_v
                    }
                else:
                    # For numerical variables, use correlation coefficient
                    corr = self.data[attr].corr(self.data[self.target_col])
                    correlations[attr] = {
                        'type': 'pearson_correlation',
                        'value': corr
                    }
        
        return correlations
    
    def _cramers_v(self, x, y):
        """Calculate Cramér's V for categorical variables"""
        confusion_matrix = pd.crosstab(x, y)
        chi2 = stats.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
    
    def _analyze_statistical_parity(self):
        """Analyze statistical parity across groups"""
        parity_analysis = {}
        
        for attr in self.sensitive_attributes:
            if attr in self.data.columns:
                groups = self.data.groupby(attr)[self.target_col].agg(['mean', 'count'])
                
                # Calculate parity metrics
                max_rate = groups['mean'].max()
                min_rate = groups['mean'].min()
                
                parity_analysis[attr] = {
                    'group_rates': groups['mean'].to_dict(),
                    'group_counts': groups['count'].to_dict(),
                    'statistical_parity_difference': max_rate - min_rate,
                    'disparate_impact_ratio': min_rate / max_rate if max_rate > 0 else 0
                }
        
        return parity_analysis
    
    def _analyze_missing_data(self):
        """Analyze patterns in missing data"""
        missing_analysis = {}
        
        for attr in self.sensitive_attributes:
            if attr in self.data.columns:
                missing_by_target = self.data.groupby(self.target_col)[attr].apply(
                    lambda x: x.isnull().sum() / len(x)
                )
                
                missing_analysis[attr] = {
                    'overall_missing_rate': self.data[attr].isnull().mean(),
                    'missing_by_target': missing_by_target.to_dict()
                }
        
        return missing_analysis

# Create synthetic biased dataset for demonstration
def create_biased_dataset(n_samples=10000):
    """Create a synthetic dataset with bias"""
    
    # Generate base features
    X, y = make_classification(
        n_samples=n_samples, n_features=10, n_informative=5,
        n_redundant=2, n_clusters_per_class=1, random_state=42
    )
    
    # Create sensitive attributes with bias
    np.random.seed(42)
    
    # Gender (binary)
    gender = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.6, 0.4])
    
    # Age groups
    age = np.random.normal(40, 15, n_samples)
    age_group = pd.cut(age, bins=[0, 30, 50, 100], labels=['Young', 'Middle', 'Senior'])
    
    # Race/Ethnicity (simplified)
    race = np.random.choice(
        ['White', 'Black', 'Hispanic', 'Asian', 'Other'], 
        size=n_samples, 
        p=[0.6, 0.15, 0.15, 0.08, 0.02]
    )
    
    # Introduce bias: Make outcomes correlated with sensitive attributes
    bias_factor = np.zeros(n_samples)
    bias_factor[gender == 'Female'] += 0.3  # Gender bias
    bias_factor[age_group == 'Senior'] += 0.2  # Age bias
    bias_factor[race == 'Black'] += 0.4  # Racial bias
    bias_factor[race == 'Hispanic'] += 0.2
    
    # Modify target based on bias
    biased_prob = 1 / (1 + np.exp(-(X[:, 0] + bias_factor)))
    y_biased = (np.random.random(n_samples) < biased_prob).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['gender'] = gender
    df['age_group'] = age_group
    df['race'] = race
    df['target'] = y_biased
    
    return df

# Demonstrate bias detection
data = create_biased_dataset()
print("Dataset shape:", data.shape)
print("\nDataset head:")
print(data.head())

# Initialize bias detection framework
bias_detector = BiasDetectionFramework(
    data=data,
    target_col='target',
    sensitive_attributes=['gender', 'age_group', 'race']
)

# Analyze data bias
bias_analysis = bias_detector.analyze_data_bias()

print("\n=== BIAS ANALYSIS RESULTS ===")

# Representation analysis
print("\n1. REPRESENTATION ANALYSIS:")
for attr, analysis in bias_analysis['representation'].items():
    print(f"\n{attr.upper()}:")
    print(f"  Proportions: {analysis['proportions']}")
    print(f"  Balance ratio: {analysis['balance_ratio']:.3f}")

# Statistical parity analysis
print("\n2. STATISTICAL PARITY ANALYSIS:")
for attr, analysis in bias_analysis['statistical_parity'].items():
    print(f"\n{attr.upper()}:")
    print(f"  Group rates: {analysis['group_rates']}")
    print(f"  Statistical parity difference: {analysis['statistical_parity_difference']:.3f}")
    print(f"  Disparate impact ratio: {analysis['disparate_impact_ratio']:.3f}")
```

### Visualization of Bias Patterns

```python
def visualize_bias_patterns(data, sensitive_attributes, target_col):
    """Create comprehensive bias visualization"""
    
    n_attrs = len(sensitive_attributes)
    fig, axes = plt.subplots(2, n_attrs, figsize=(5*n_attrs, 10))
    if n_attrs == 1:
        axes = axes.reshape(2, 1)
    
    for i, attr in enumerate(sensitive_attributes):
        # Distribution by sensitive attribute
        data.groupby(attr)[target_col].mean().plot(kind='bar', ax=axes[0, i])
        axes[0, i].set_title(f'Average Target Rate by {attr}')
        axes[0, i].set_ylabel('Positive Rate')
        axes[0, i].tick_params(axis='x', rotation=45)
        
        # Count by sensitive attribute
        data[attr].value_counts().plot(kind='bar', ax=axes[1, i])
        axes[1, i].set_title(f'Count Distribution by {attr}')
        axes[1, i].set_ylabel('Count')
        axes[1, i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# Visualize bias patterns
visualize_bias_patterns(data, ['gender', 'age_group', 'race'], 'target')

# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_data = data.copy()

# Encode categorical variables for correlation
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in ['gender', 'age_group', 'race']:
    correlation_data[col + '_encoded'] = le.fit_transform(correlation_data[col])

# Select numerical columns for correlation
numerical_cols = [col for col in correlation_data.columns 
                 if correlation_data[col].dtype in ['int64', 'float64']]
correlation_matrix = correlation_data[numerical_cols].corr()

# Plot heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix: Features vs Target')
plt.tight_layout()
plt.show()
```

## 3. Model Bias Detection

### Fairness Metrics Implementation

```python
class ModelBiasDetector:
    """Detect bias in trained models"""
    
    def __init__(self, model, X_test, y_test, sensitive_features):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.sensitive_features = sensitive_features
        self.predictions = model.predict(X_test)
        self.pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    def calculate_fairness_metrics(self):
        """Calculate comprehensive fairness metrics"""
        metrics = {}
        
        for feature in self.sensitive_features:
            feature_metrics = {}
            
            # Get unique groups
            groups = np.unique(self.X_test[feature])
            
            for group in groups:
                mask = self.X_test[feature] == group
                y_true_group = self.y_test[mask]
                y_pred_group = self.predictions[mask]
                
                # Basic metrics
                feature_metrics[group] = {
                    'count': len(y_true_group),
                    'positive_rate': np.mean(y_pred_group),
                    'true_positive_rate': np.mean(y_pred_group[y_true_group == 1]) if np.any(y_true_group == 1) else 0,
                    'false_positive_rate': np.mean(y_pred_group[y_true_group == 0]) if np.any(y_true_group == 0) else 0,
                    'accuracy': np.mean(y_true_group == y_pred_group)
                }
            
            # Calculate parity metrics
            positive_rates = [feature_metrics[group]['positive_rate'] for group in groups]
            tpr_rates = [feature_metrics[group]['true_positive_rate'] for group in groups]
            fpr_rates = [feature_metrics[group]['false_positive_rate'] for group in groups]
            
            feature_metrics['fairness_metrics'] = {
                'statistical_parity_difference': max(positive_rates) - min(positive_rates),
                'equal_opportunity_difference': max(tpr_rates) - min(tpr_rates),
                'equalized_odds_difference': max(abs(np.array(tpr_rates) - np.array(fpr_rates))),
                'disparate_impact_ratio': min(positive_rates) / max(positive_rates) if max(positive_rates) > 0 else 0
            }
            
            metrics[feature] = feature_metrics
        
        return metrics
    
    def calculate_individual_fairness_metrics(self):
        """Calculate individual fairness metrics"""
        if self.pred_proba is None:
            return "Model does not support probability predictions"
        
        individual_metrics = {}
        
        for feature in self.sensitive_features:
            # Calculate consistency across similar individuals
            feature_values = self.X_test[feature].values
            similarity_scores = []
            
            for i in range(min(100, len(self.X_test))):  # Sample for efficiency
                # Find similar individuals (same sensitive attribute value)
                same_group_mask = feature_values == feature_values[i]
                same_group_indices = np.where(same_group_mask)[0]
                
                if len(same_group_indices) > 1:
                    # Calculate prediction variance within group
                    group_predictions = self.pred_proba[same_group_indices]
                    prediction_variance = np.var(group_predictions)
                    similarity_scores.append(prediction_variance)
            
            individual_metrics[feature] = {
                'avg_prediction_variance': np.mean(similarity_scores) if similarity_scores else 0,
                'consistency_score': 1 - np.mean(similarity_scores) if similarity_scores else 1
            }
        
        return individual_metrics

# Train models and detect bias
X = data.drop(['target', 'gender', 'age_group', 'race'], axis=1)
y = data['target']
sensitive_df = data[['gender', 'age_group', 'race']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sensitive_train, sensitive_test = train_test_split(sensitive_df, test_size=0.2, random_state=42)

# Combine test features with sensitive attributes
X_test_with_sensitive = pd.concat([X_test.reset_index(drop=True), 
                                  sensitive_test.reset_index(drop=True)], axis=1)

# Train multiple models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

bias_results = {}

for model_name, model in models.items():
    print(f"\n=== {model_name.upper()} BIAS ANALYSIS ===")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Detect bias
    bias_detector = ModelBiasDetector(
        model=model,
        X_test=X_test_with_sensitive,
        y_test=y_test.values,
        sensitive_features=['gender', 'age_group', 'race']
    )
    
    # Calculate fairness metrics
    fairness_metrics = bias_detector.calculate_fairness_metrics()
    individual_metrics = bias_detector.calculate_individual_fairness_metrics()
    
    bias_results[model_name] = {
        'fairness_metrics': fairness_metrics,
        'individual_metrics': individual_metrics
    }
    
    # Print results
    for feature, metrics in fairness_metrics.items():
        print(f"\n{feature.upper()} Fairness:")
        fairness = metrics['fairness_metrics']
        print(f"  Statistical Parity Difference: {fairness['statistical_parity_difference']:.3f}")
        print(f"  Equal Opportunity Difference: {fairness['equal_opportunity_difference']:.3f}")
        print(f"  Disparate Impact Ratio: {fairness['disparate_impact_ratio']:.3f}")
        
        # Flag potential bias
        if fairness['statistical_parity_difference'] > 0.1:
            print(f"  ⚠️  WARNING: High statistical parity difference!")
        if fairness['disparate_impact_ratio'] < 0.8:
            print(f"  ⚠️  WARNING: Low disparate impact ratio!")
```

### Advanced Bias Detection Techniques

```python
class AdvancedBiasDetector:
    """Advanced bias detection methods"""
    
    def __init__(self, model, X, y, sensitive_features):
        self.model = model
        self.X = X
        self.y = y
        self.sensitive_features = sensitive_features
    
    def intersectional_bias_analysis(self):
        """Analyze bias across intersections of sensitive attributes"""
        intersectional_results = {}
        
        # Create intersectional groups
        X_with_sensitive = self.X.copy()
        
        # Create intersection columns
        for i in range(len(self.sensitive_features)):
            for j in range(i+1, len(self.sensitive_features)):
                attr1, attr2 = self.sensitive_features[i], self.sensitive_features[j]
                intersection_name = f"{attr1}_{attr2}"
                X_with_sensitive[intersection_name] = (
                    X_with_sensitive[attr1].astype(str) + "_" + 
                    X_with_sensitive[attr2].astype(str)
                )
                
                # Analyze bias for intersection
                predictions = self.model.predict(X_with_sensitive.drop(self.sensitive_features + [intersection_name], axis=1))
                
                group_results = {}
                for group in X_with_sensitive[intersection_name].unique():
                    mask = X_with_sensitive[intersection_name] == group
                    group_predictions = predictions[mask]
                    group_actuals = self.y[mask]
                    
                    if len(group_predictions) > 0:
                        group_results[group] = {
                            'count': len(group_predictions),
                            'positive_rate': np.mean(group_predictions),
                            'accuracy': np.mean(group_predictions == group_actuals)
                        }
                
                intersectional_results[intersection_name] = group_results
        
        return intersectional_results
    
    def temporal_bias_analysis(self, time_column):
        """Analyze how bias changes over time"""
        if time_column not in self.X.columns:
            return "Time column not found in data"
        
        temporal_results = {}
        
        # Group by time periods (e.g., quarters, years)
        time_periods = pd.to_datetime(self.X[time_column]).dt.to_period('Q')
        
        for period in time_periods.unique():
            period_mask = time_periods == period
            period_X = self.X[period_mask]
            period_y = self.y[period_mask]
            
            if len(period_X) > 0:
                predictions = self.model.predict(period_X.drop(self.sensitive_features + [time_column], axis=1))
                
                period_bias = {}
                for feature in self.sensitive_features:
                    feature_bias = {}
                    for group in period_X[feature].unique():
                        group_mask = period_X[feature] == group
                        group_preds = predictions[group_mask]
                        
                        if len(group_preds) > 0:
                            feature_bias[group] = np.mean(group_preds)
                    
                    period_bias[feature] = feature_bias
                
                temporal_results[str(period)] = period_bias
        
        return temporal_results
    
    def counterfactual_bias_detection(self):
        """Detect bias using counterfactual analysis"""
        counterfactual_results = {}
        
        for feature in self.sensitive_features:
            feature_results = {}
            unique_values = self.X[feature].unique()
            
            if len(unique_values) == 2:  # Binary feature
                # For each instance, flip the sensitive attribute and check prediction change
                prediction_changes = []
                
                for idx in range(min(1000, len(self.X))):  # Sample for efficiency
                    original_instance = self.X.iloc[idx:idx+1].copy()
                    original_pred = self.model.predict_proba(
                        original_instance.drop(self.sensitive_features, axis=1)
                    )[0, 1] if hasattr(self.model, 'predict_proba') else self.model.predict(
                        original_instance.drop(self.sensitive_features, axis=1)
                    )[0]
                    
                    # Create counterfactual
                    counterfactual_instance = original_instance.copy()
                    current_value = original_instance[feature].iloc[0]
                    other_value = [v for v in unique_values if v != current_value][0]
                    counterfactual_instance[feature] = other_value
                    
                    counterfactual_pred = self.model.predict_proba(
                        counterfactual_instance.drop(self.sensitive_features, axis=1)
                    )[0, 1] if hasattr(self.model, 'predict_proba') else self.model.predict(
                        counterfactual_instance.drop(self.sensitive_features, axis=1)
                    )[0]
                    
                    prediction_change = abs(original_pred - counterfactual_pred)
                    prediction_changes.append(prediction_change)
                
                feature_results = {
                    'avg_prediction_change': np.mean(prediction_changes),
                    'max_prediction_change': np.max(prediction_changes),
                    'instances_affected': np.sum(np.array(prediction_changes) > 0.1)
                }
            
            counterfactual_results[feature] = feature_results
        
        return counterfactual_results

# Example usage of advanced bias detection
# Note: For demonstration, we'll simulate some of these analyses

print("\n=== ADVANCED BIAS DETECTION ===")

# Select the Random Forest model for advanced analysis
rf_model = models['Random Forest']

advanced_detector = AdvancedBiasDetector(
    model=rf_model,
    X=X_test_with_sensitive,
    y=y_test.values,
    sensitive_features=['gender', 'age_group', 'race']
)

# Intersectional bias analysis
intersectional_bias = advanced_detector.intersectional_bias_analysis()
print("\nIntersectional Bias Analysis:")
for intersection, groups in intersectional_bias.items():
    print(f"\n{intersection}:")
    for group, metrics in groups.items():
        print(f"  {group}: Positive Rate = {metrics['positive_rate']:.3f}, Count = {metrics['count']}")

# Counterfactual bias detection
counterfactual_bias = advanced_detector.counterfactual_bias_detection()
print("\nCounterfactual Bias Analysis:")
for feature, results in counterfactual_bias.items():
    if isinstance(results, dict):
        print(f"\n{feature}:")
        print(f"  Average prediction change: {results['avg_prediction_change']:.3f}")
        print(f"  Max prediction change: {results['max_prediction_change']:.3f}")
        print(f"  Instances significantly affected: {results['instances_affected']}")
```

## 4. Automated Bias Detection Pipeline

### Production Bias Monitoring

```python
class BiasMonitoringSystem:
    """Production system for continuous bias monitoring"""
    
    def __init__(self, model, reference_data, sensitive_features, thresholds=None):
        self.model = model
        self.reference_data = reference_data
        self.sensitive_features = sensitive_features
        self.thresholds = thresholds or {
            'statistical_parity_difference': 0.1,
            'disparate_impact_ratio': 0.8,
            'equal_opportunity_difference': 0.1
        }
        self.monitoring_history = []
    
    def monitor_batch(self, new_data, timestamp=None):
        """Monitor bias in a new batch of data"""
        if timestamp is None:
            timestamp = pd.Timestamp.now()
        
        # Calculate current bias metrics
        current_metrics = self._calculate_bias_metrics(new_data)
        
        # Compare with reference
        bias_alerts = self._check_bias_thresholds(current_metrics)
        
        # Store monitoring results
        monitoring_result = {
            'timestamp': timestamp,
            'metrics': current_metrics,
            'alerts': bias_alerts,
            'data_size': len(new_data)
        }
        
        self.monitoring_history.append(monitoring_result)
        
        return monitoring_result
    
    def _calculate_bias_metrics(self, data):
        """Calculate bias metrics for given data"""
        metrics = {}
        
        # Get predictions
        feature_cols = [col for col in data.columns if col not in self.sensitive_features]
        predictions = self.model.predict(data[feature_cols])
        
        for feature in self.sensitive_features:
            if feature in data.columns:
                feature_metrics = {}
                groups = data[feature].unique()
                
                group_positive_rates = []
                for group in groups:
                    mask = data[feature] == group
                    group_preds = predictions[mask]
                    positive_rate = np.mean(group_preds) if len(group_preds) > 0 else 0
                    feature_metrics[group] = {
                        'positive_rate': positive_rate,
                        'count': len(group_preds)
                    }
                    group_positive_rates.append(positive_rate)
                
                # Calculate fairness metrics
                if len(group_positive_rates) > 1:
                    feature_metrics['fairness'] = {
                        'statistical_parity_difference': max(group_positive_rates) - min(group_positive_rates),
                        'disparate_impact_ratio': min(group_positive_rates) / max(group_positive_rates) if max(group_positive_rates) > 0 else 0
                    }
                
                metrics[feature] = feature_metrics
        
        return metrics
    
    def _check_bias_thresholds(self, metrics):
        """Check if bias metrics exceed thresholds"""
        alerts = []
        
        for feature, feature_metrics in metrics.items():
            if 'fairness' in feature_metrics:
                fairness = feature_metrics['fairness']
                
                # Check statistical parity
                if fairness['statistical_parity_difference'] > self.thresholds['statistical_parity_difference']:
                    alerts.append({
                        'type': 'statistical_parity_violation',
                        'feature': feature,
                        'value': fairness['statistical_parity_difference'],
                        'threshold': self.thresholds['statistical_parity_difference']
                    })
                
                # Check disparate impact
                if fairness['disparate_impact_ratio'] < self.thresholds['disparate_impact_ratio']:
                    alerts.append({
                        'type': 'disparate_impact_violation',
                        'feature': feature,
                        'value': fairness['disparate_impact_ratio'],
                        'threshold': self.thresholds['disparate_impact_ratio']
                    })
        
        return alerts
    
    def generate_bias_report(self, time_range=None):
        """Generate comprehensive bias monitoring report"""
        if not self.monitoring_history:
            return "No monitoring data available"
        
        # Filter by time range if specified
        history = self.monitoring_history
        if time_range:
            start_time, end_time = time_range
            history = [h for h in history if start_time <= h['timestamp'] <= end_time]
        
        report = {
            'summary': {
                'total_batches_monitored': len(history),
                'total_alerts': sum(len(h['alerts']) for h in history),
                'time_range': (history[0]['timestamp'], history[-1]['timestamp']) if history else None
            },
            'trend_analysis': self._analyze_bias_trends(history),
            'alert_summary': self._summarize_alerts(history)
        }
        
        return report
    
    def _analyze_bias_trends(self, history):
        """Analyze trends in bias metrics over time"""
        trends = {}
        
        for feature in self.sensitive_features:
            feature_trends = []
            
            for record in history:
                if feature in record['metrics'] and 'fairness' in record['metrics'][feature]:
                    fairness = record['metrics'][feature]['fairness']
                    feature_trends.append({
                        'timestamp': record['timestamp'],
                        'statistical_parity_difference': fairness['statistical_parity_difference'],
                        'disparate_impact_ratio': fairness['disparate_impact_ratio']
                    })
            
            if feature_trends:
                # Calculate trend direction
                spd_values = [t['statistical_parity_difference'] for t in feature_trends]
                dir_values = [t['disparate_impact_ratio'] for t in feature_trends]
                
                trends[feature] = {
                    'statistical_parity_trend': 'increasing' if len(spd_values) > 1 and spd_values[-1] > spd_values[0] else 'stable',
                    'disparate_impact_trend': 'decreasing' if len(dir_values) > 1 and dir_values[-1] < dir_values[0] else 'stable',
                    'avg_statistical_parity': np.mean(spd_values),
                    'avg_disparate_impact': np.mean(dir_values)
                }
        
        return trends
    
    def _summarize_alerts(self, history):
        """Summarize alerts by type and feature"""
        alert_summary = {}
        
        for record in history:
            for alert in record['alerts']:
                alert_type = alert['type']
                feature = alert['feature']
                
                key = f"{feature}_{alert_type}"
                if key not in alert_summary:
                    alert_summary[key] = {
                        'count': 0,
                        'feature': feature,
                        'type': alert_type,
                        'max_value': 0,
                        'avg_value': 0,
                        'values': []
                    }
                
                alert_summary[key]['count'] += 1
                alert_summary[key]['values'].append(alert['value'])
                alert_summary[key]['max_value'] = max(alert_summary[key]['max_value'], alert['value'])
        
        # Calculate averages
        for key in alert_summary:
            values = alert_summary[key]['values']
            alert_summary[key]['avg_value'] = np.mean(values) if values else 0
            del alert_summary[key]['values']  # Remove raw values for cleaner output
        
        return alert_summary

# Demonstrate bias monitoring system
print("\n=== BIAS MONITORING SYSTEM ===")

# Initialize monitoring system
bias_monitor = BiasMonitoringSystem(
    model=rf_model,
    reference_data=X_test_with_sensitive,
    sensitive_features=['gender', 'age_group', 'race']
)

# Simulate monitoring over time with different bias levels
for i in range(5):
    # Create slightly biased batches
    batch_data = data.sample(1000, replace=True).copy()
    
    # Simulate drift - gradually increase bias
    if i > 2:
        # Increase bias for certain groups
        mask = batch_data['gender'] == 'Female'
        batch_data.loc[mask, 'target'] = np.random.choice([0, 1], size=mask.sum(), p=[0.8, 0.2])
    
    # Monitor batch
    result = bias_monitor.monitor_batch(
        batch_data,
        timestamp=pd.Timestamp.now() + pd.Timedelta(days=i)
    )
    
    print(f"\nBatch {i+1} Monitoring Results:")
    print(f"  Alerts: {len(result['alerts'])}")
    for alert in result['alerts']:
        print(f"    {alert['type']} for {alert['feature']}: {alert['value']:.3f} (threshold: {alert['threshold']:.3f})")

# Generate comprehensive report
bias_report = bias_monitor.generate_bias_report()
print(f"\n=== BIAS MONITORING REPORT ===")
print(f"Total batches monitored: {bias_report['summary']['total_batches_monitored']}")
print(f"Total alerts generated: {bias_report['summary']['total_alerts']}")

print(f"\nTrend Analysis:")
for feature, trends in bias_report['trend_analysis'].items():
    print(f"  {feature}:")
    print(f"    Average Statistical Parity Difference: {trends['avg_statistical_parity']:.3f}")
    print(f"    Average Disparate Impact Ratio: {trends['avg_disparate_impact']:.3f}")
```

## 5. Integration with ML Pipeline

### Automated Bias Testing

```python
class BiasTestSuite:
    """Automated bias testing for ML pipelines"""
    
    def __init__(self, test_config):
        self.test_config = test_config
        self.test_results = {}
    
    def run_all_tests(self, model, data, sensitive_features):
        """Run comprehensive bias test suite"""
        
        tests = [
            self._test_statistical_parity,
            self._test_equal_opportunity,
            self._test_demographic_parity,
            self._test_individual_fairness,
            self._test_counterfactual_fairness
        ]
        
        for test in tests:
            test_name = test.__name__.replace('_test_', '')
            try:
                result = test(model, data, sensitive_features)
                self.test_results[test_name] = {
                    'status': 'PASS' if result['passed'] else 'FAIL',
                    'score': result['score'],
                    'details': result['details'],
                    'recommendations': result.get('recommendations', [])
                }
            except Exception as e:
                self.test_results[test_name] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'recommendations': ['Fix test implementation error']
                }
        
        return self.test_results
    
    def _test_statistical_parity(self, model, data, sensitive_features):
        """Test statistical parity across groups"""
        threshold = self.test_config.get('statistical_parity_threshold', 0.1)
        
        max_violation = 0
        violations = []
        
        for feature in sensitive_features:
            if feature in data.columns:
                groups = data[feature].unique()
                positive_rates = []
                
                for group in groups:
                    mask = data[feature] == group
                    group_data = data[mask]
                    feature_cols = [col for col in data.columns if col not in sensitive_features]
                    predictions = model.predict(group_data[feature_cols])
                    positive_rate = np.mean(predictions)
                    positive_rates.append(positive_rate)
                
                if len(positive_rates) > 1:
                    parity_diff = max(positive_rates) - min(positive_rates)
                    max_violation = max(max_violation, parity_diff)
                    
                    if parity_diff > threshold:
                        violations.append({
                            'feature': feature,
                            'violation': parity_diff,
                            'groups': dict(zip(groups, positive_rates))
                        })
        
        passed = max_violation <= threshold
        score = max(0, 1 - (max_violation / threshold)) if threshold > 0 else 1
        
        recommendations = []
        if not passed:
            recommendations.extend([
                'Consider bias mitigation techniques during preprocessing',
                'Apply fairness constraints during model training',
                'Use post-processing calibration methods'
            ])
        
        return {
            'passed': passed,
            'score': score,
            'details': {
                'max_violation': max_violation,
                'threshold': threshold,
                'violations': violations
            },
            'recommendations': recommendations
        }
    
    def _test_equal_opportunity(self, model, data, sensitive_features):
        """Test equal opportunity (equal TPR across groups)"""
        threshold = self.test_config.get('equal_opportunity_threshold', 0.1)
        
        if 'target' not in data.columns:
            return {
                'passed': False,
                'score': 0,
                'details': {'error': 'Target column required for equal opportunity test'}
            }
        
        max_violation = 0
        violations = []
        
        for feature in sensitive_features:
            if feature in data.columns:
                groups = data[feature].unique()
                tpr_rates = []
                
                for group in groups:
                    mask = data[feature] == group
                    group_data = data[mask]
                    
                    # Only consider positive cases
                    positive_cases = group_data[group_data['target'] == 1]
                    if len(positive_cases) > 0:
                        feature_cols = [col for col in data.columns if col not in sensitive_features + ['target']]
                        predictions = model.predict(positive_cases[feature_cols])
                        tpr = np.mean(predictions)
                        tpr_rates.append(tpr)
                
                if len(tpr_rates) > 1:
                    tpr_diff = max(tpr_rates) - min(tpr_rates)
                    max_violation = max(max_violation, tpr_diff)
                    
                    if tpr_diff > threshold:
                        violations.append({
                            'feature': feature,
                            'violation': tpr_diff,
                            'tpr_rates': tpr_rates
                        })
        
        passed = max_violation <= threshold
        score = max(0, 1 - (max_violation / threshold)) if threshold > 0 else 1
        
        return {
            'passed': passed,
            'score': score,
            'details': {
                'max_violation': max_violation,
                'threshold': threshold,
                'violations': violations
            }
        }
    
    def _test_demographic_parity(self, model, data, sensitive_features):
        """Test demographic parity"""
        # Similar to statistical parity but with different interpretation
        return self._test_statistical_parity(model, data, sensitive_features)
    
    def _test_individual_fairness(self, model, data, sensitive_features):
        """Test individual fairness (similar individuals get similar predictions)"""
        sample_size = min(100, len(data))
        consistency_scores = []
        
        for feature in sensitive_features:
            if feature in data.columns:
                feature_consistency = []
                
                # Sample individuals
                sampled_data = data.sample(sample_size)
                
                for _, individual in sampled_data.iterrows():
                    # Find similar individuals (same sensitive attribute)
                    similar_mask = data[feature] == individual[feature]
                    similar_individuals = data[similar_mask]
                    
                    if len(similar_individuals) > 1:
                        feature_cols = [col for col in data.columns if col not in sensitive_features]
                        
                        if hasattr(model, 'predict_proba'):
                            predictions = model.predict_proba(similar_individuals[feature_cols])[:, 1]
                        else:
                            predictions = model.predict(similar_individuals[feature_cols])
                        
                        # Calculate prediction variance (lower is better)
                        prediction_variance = np.var(predictions)
                        consistency = 1 / (1 + prediction_variance)  # Convert to consistency score
                        feature_consistency.append(consistency)
                
                if feature_consistency:
                    consistency_scores.append(np.mean(feature_consistency))
        
        overall_consistency = np.mean(consistency_scores) if consistency_scores else 1
        threshold = self.test_config.get('individual_fairness_threshold', 0.8)
        
        passed = overall_consistency >= threshold
        
        return {
            'passed': passed,
            'score': overall_consistency,
            'details': {
                'consistency_score': overall_consistency,
                'threshold': threshold,
                'feature_scores': dict(zip(sensitive_features, consistency_scores))
            }
        }
    
    def _test_counterfactual_fairness(self, model, data, sensitive_features):
        """Test counterfactual fairness"""
        max_impact = 0
        impacts = []
        
        sample_size = min(50, len(data))  # Reduced for efficiency
        sampled_data = data.sample(sample_size)
        
        for feature in sensitive_features:
            if feature in data.columns:
                unique_values = data[feature].unique()
                
                if len(unique_values) == 2:  # Only test binary features
                    feature_impacts = []
                    
                    for _, individual in sampled_data.iterrows():
                        original_value = individual[feature]
                        other_value = [v for v in unique_values if v != original_value][0]
                        
                        # Original prediction
                        feature_cols = [col for col in data.columns if col not in sensitive_features]
                        original_individual = individual[feature_cols].values.reshape(1, -1)
                        
                        if hasattr(model, 'predict_proba'):
                            original_pred = model.predict_proba(original_individual)[0, 1]
                        else:
                            original_pred = model.predict(original_individual)[0]
                        
                        # Counterfactual prediction (flip sensitive attribute)
                        counterfactual_individual = individual.copy()
                        counterfactual_individual[feature] = other_value
                        counterfactual_data = counterfactual_individual[feature_cols].values.reshape(1, -1)
                        
                        if hasattr(model, 'predict_proba'):
                            counterfactual_pred = model.predict_proba(counterfactual_data)[0, 1]
                        else:
                            counterfactual_pred = model.predict(counterfactual_data)[0]
                        
                        impact = abs(original_pred - counterfactual_pred)
                        feature_impacts.append(impact)
                    
                    if feature_impacts:
                        avg_impact = np.mean(feature_impacts)
                        max_impact = max(max_impact, avg_impact)
                        impacts.append({
                            'feature': feature,
                            'avg_impact': avg_impact,
                            'max_impact': np.max(feature_impacts)
                        })
        
        threshold = self.test_config.get('counterfactual_fairness_threshold', 0.1)
        passed = max_impact <= threshold
        score = max(0, 1 - (max_impact / threshold)) if threshold > 0 else 1
        
        return {
            'passed': passed,
            'score': score,
            'details': {
                'max_impact': max_impact,
                'threshold': threshold,
                'feature_impacts': impacts
            }
        }
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        if not self.test_results:
            return "No test results available"
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'PASS')
        failed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'FAIL')
        error_tests = sum(1 for result in self.test_results.values() if result['status'] == 'ERROR')
        
        average_score = np.mean([
            result['score'] for result in self.test_results.values() 
            if 'score' in result and isinstance(result['score'], (int, float))
        ])
        
        report = f"""
=== BIAS TEST SUITE REPORT ===

SUMMARY:
  Total Tests: {total_tests}
  Passed: {passed_tests}
  Failed: {failed_tests}
  Errors: {error_tests}
  Average Score: {average_score:.3f}

DETAILED RESULTS:
"""
        
        for test_name, result in self.test_results.items():
            report += f"\n{test_name.upper()}:"
            report += f"\n  Status: {result['status']}"
            
            if 'score' in result:
                report += f"\n  Score: {result['score']:.3f}"
            
            if 'error' in result:
                report += f"\n  Error: {result['error']}"
            
            if 'recommendations' in result and result['recommendations']:
                report += f"\n  Recommendations:"
                for rec in result['recommendations']:
                    report += f"\n    - {rec}"
            
            report += "\n"
        
        return report

# Example usage of bias test suite
print("\n=== AUTOMATED BIAS TESTING ===")

# Configure test thresholds
test_config = {
    'statistical_parity_threshold': 0.1,
    'equal_opportunity_threshold': 0.1,
    'individual_fairness_threshold': 0.8,
    'counterfactual_fairness_threshold': 0.1
}

# Initialize test suite
bias_tests = BiasTestSuite(test_config)

# Prepare test data (include target for equal opportunity test)
test_data = data.sample(1000).copy()

# Run all bias tests
test_results = bias_tests.run_all_tests(
    model=rf_model,
    data=test_data,
    sensitive_features=['gender', 'age_group', 'race']
)

# Generate and print report
test_report = bias_tests.generate_test_report()
print(test_report)
```

## Summary

Bias detection in machine learning is critical for building fair and trustworthy AI systems. Key components include:

### Detection Methods
1. **Data Bias Analysis**: Representation, correlation, and statistical parity analysis
2. **Model Bias Metrics**: Fairness metrics like statistical parity, equal opportunity
3. **Advanced Techniques**: Intersectional, temporal, and counterfactual analysis
4. **Production Monitoring**: Continuous bias tracking and alerting

### Implementation Framework
- **Comprehensive Detection**: Multiple bias types and metrics
- **Automated Testing**: Integration with ML pipelines
- **Monitoring Systems**: Real-time bias tracking
- **Reporting**: Clear documentation and recommendations

### Best Practices
- Start bias detection early in the ML development process
- Use multiple detection methods for comprehensive coverage
- Implement continuous monitoring in production
- Combine automated detection with human review
- Document findings and mitigation strategies

Bias detection is an ongoing process that requires vigilance throughout the ML lifecycle to ensure fair and equitable AI systems.