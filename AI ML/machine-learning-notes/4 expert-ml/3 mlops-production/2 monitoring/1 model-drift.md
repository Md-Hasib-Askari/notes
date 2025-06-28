# Model Drift Monitoring

## Overview
Model drift refers to the degradation of a machine learning model's performance over time due to changes in data patterns, feature distributions, or relationships between features and targets. Monitoring and detecting drift is crucial for maintaining model performance in production.

## Types of Drift

### Data Drift (Covariate Shift)
- **Definition**: Changes in input feature distributions
- **Characteristics**: P(X) changes, but P(Y|X) remains stable
- **Example**: Image classification model seeing different lighting conditions
- **Impact**: Can affect model performance if model is sensitive to input changes

### Concept Drift
- **Definition**: Changes in the relationship between features and target
- **Characteristics**: P(Y|X) changes, P(X) may or may not change
- **Example**: Customer behavior changes affecting churn prediction
- **Impact**: Directly affects model accuracy and predictions

### Label Drift (Prior Probability Shift)
- **Definition**: Changes in target variable distribution
- **Characteristics**: P(Y) changes
- **Example**: Seasonal changes in product sales categories
- **Impact**: Can affect model calibration and decision thresholds

### Prediction Drift
- **Definition**: Changes in model prediction distributions
- **Characteristics**: Changes in P(ŷ) over time
- **Example**: Model starts predicting different class proportions
- **Impact**: May indicate underlying data or concept drift

## Drift Detection Methods

### Statistical Tests

#### Kolmogorov-Smirnov Test
```python
import numpy as np
from scipy import stats

def ks_drift_test(reference_data, current_data, alpha=0.05):
    """
    Perform KS test for drift detection
    """
    statistic, p_value = stats.ks_2samp(reference_data, current_data)
    
    drift_detected = p_value < alpha
    
    return {
        'drift_detected': drift_detected,
        'p_value': p_value,
        'ks_statistic': statistic,
        'threshold': alpha
    }

# Example usage
reference = np.random.normal(0, 1, 1000)
current = np.random.normal(0.5, 1, 1000)  # Shifted distribution

result = ks_drift_test(reference, current)
print(f"Drift detected: {result['drift_detected']}")
print(f"P-value: {result['p_value']:.4f}")
```

#### Chi-Square Test (Categorical Features)
```python
from scipy.stats import chi2_contingency

def chi2_drift_test(reference_data, current_data, alpha=0.05):
    """
    Chi-square test for categorical drift detection
    """
    # Create contingency table
    ref_counts = np.bincount(reference_data)
    cur_counts = np.bincount(current_data)
    
    # Ensure same length
    max_len = max(len(ref_counts), len(cur_counts))
    ref_counts = np.pad(ref_counts, (0, max_len - len(ref_counts)))
    cur_counts = np.pad(cur_counts, (0, max_len - len(cur_counts)))
    
    contingency_table = np.array([ref_counts, cur_counts])
    
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    drift_detected = p_value < alpha
    
    return {
        'drift_detected': drift_detected,
        'p_value': p_value,
        'chi2_statistic': chi2,
        'threshold': alpha
    }
```

#### Population Stability Index (PSI)
```python
import numpy as np

def calculate_psi(reference, current, bins=10):
    """
    Calculate Population Stability Index
    """
    def get_psi_buckets(data, bins):
        """Create PSI buckets"""
        if isinstance(bins, int):
            # Equal-width binning
            bins = np.linspace(data.min(), data.max(), bins + 1)
        
        bucket_counts = np.histogram(data, bins=bins)[0]
        bucket_props = bucket_counts / len(data)
        
        return bucket_props
    
    # Get bucket proportions
    ref_props = get_psi_buckets(reference, bins)
    cur_props = get_psi_buckets(current, bins)
    
    # Avoid division by zero
    ref_props = np.where(ref_props == 0, 0.0001, ref_props)
    cur_props = np.where(cur_props == 0, 0.0001, cur_props)
    
    # Calculate PSI
    psi_values = (cur_props - ref_props) * np.log(cur_props / ref_props)
    psi = np.sum(psi_values)
    
    # PSI interpretation
    if psi < 0.1:
        stability = "Stable"
    elif psi < 0.2:
        stability = "Moderate drift"
    else:
        stability = "Significant drift"
    
    return {
        'psi': psi,
        'stability': stability,
        'bucket_psi': psi_values
    }

# Example usage
reference = np.random.normal(0, 1, 1000)
current = np.random.normal(0.3, 1.2, 1000)

psi_result = calculate_psi(reference, current)
print(f"PSI: {psi_result['psi']:.4f}")
print(f"Stability: {psi_result['stability']}")
```

### Distance-Based Methods

#### Wasserstein Distance
```python
from scipy.stats import wasserstein_distance

def wasserstein_drift_test(reference_data, current_data, threshold=0.1):
    """
    Wasserstein distance for drift detection
    """
    distance = wasserstein_distance(reference_data, current_data)
    
    drift_detected = distance > threshold
    
    return {
        'drift_detected': drift_detected,
        'wasserstein_distance': distance,
        'threshold': threshold
    }
```

#### Jensen-Shannon Divergence
```python
import numpy as np
from scipy.spatial.distance import jensenshannon

def js_divergence_test(reference_data, current_data, bins=50, threshold=0.1):
    """
    Jensen-Shannon divergence for drift detection
    """
    # Create histograms
    hist_range = (min(reference_data.min(), current_data.min()),
                  max(reference_data.max(), current_data.max()))
    
    ref_hist, _ = np.histogram(reference_data, bins=bins, range=hist_range, density=True)
    cur_hist, _ = np.histogram(current_data, bins=bins, range=hist_range, density=True)
    
    # Normalize to probabilities
    ref_hist = ref_hist / ref_hist.sum()
    cur_hist = cur_hist / cur_hist.sum()
    
    # Calculate JS divergence
    js_div = jensenshannon(ref_hist, cur_hist)
    
    drift_detected = js_div > threshold
    
    return {
        'drift_detected': drift_detected,
        'js_divergence': js_div,
        'threshold': threshold
    }
```

## Advanced Drift Detection

### Multivariate Drift Detection
```python
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

class MultivariateDriftDetector:
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.detector = IsolationForest(contamination=contamination, random_state=42)
        self.is_fitted = False
    
    def fit(self, reference_data):
        """Fit on reference data"""
        self.detector.fit(reference_data)
        self.reference_scores = self.detector.decision_function(reference_data)
        self.threshold = np.percentile(self.reference_scores, self.contamination * 100)
        self.is_fitted = True
    
    def detect_drift(self, current_data):
        """Detect drift in current data"""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first")
        
        current_scores = self.detector.decision_function(current_data)
        outlier_ratio = np.mean(current_scores < self.threshold)
        
        # Drift detected if outlier ratio significantly higher than expected
        drift_detected = outlier_ratio > (self.contamination * 2)
        
        return {
            'drift_detected': drift_detected,
            'outlier_ratio': outlier_ratio,
            'expected_ratio': self.contamination,
            'current_scores': current_scores
        }
```

### Domain Classifier Approach
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def domain_classifier_drift_test(reference_data, current_data, threshold=0.75):
    """
    Use a classifier to distinguish between reference and current data
    """
    # Combine data and create labels
    X = np.vstack([reference_data, current_data])
    y = np.hstack([np.zeros(len(reference_data)), np.ones(len(current_data))])
    
    # Split for validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # AUC close to 0.5 means no drift, close to 1.0 means significant drift
    drift_detected = auc_score > threshold
    
    return {
        'drift_detected': drift_detected,
        'auc_score': auc_score,
        'threshold': threshold,
        'feature_importance': clf.feature_importances_
    }
```

## Real-time Drift Monitoring

### Sliding Window Approach
```python
from collections import deque
import pandas as pd

class SlidingWindowDriftMonitor:
    def __init__(self, reference_data, window_size=1000, drift_threshold=0.1):
        self.reference_data = reference_data
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.current_window = deque(maxlen=window_size)
        self.drift_history = []
    
    def add_sample(self, sample):
        """Add new sample to current window"""
        self.current_window.append(sample)
        
        if len(self.current_window) >= self.window_size:
            drift_result = self.check_drift()
            self.drift_history.append({
                'timestamp': pd.Timestamp.now(),
                'drift_detected': drift_result['drift_detected'],
                'drift_score': drift_result.get('drift_score', 0)
            })
            
            return drift_result
        
        return None
    
    def check_drift(self):
        """Check for drift in current window"""
        current_data = np.array(list(self.current_window))
        
        # Use PSI for drift detection
        psi_result = calculate_psi(self.reference_data, current_data)
        
        return {
            'drift_detected': psi_result['psi'] > self.drift_threshold,
            'drift_score': psi_result['psi'],
            'method': 'PSI'
        }
    
    def get_drift_summary(self):
        """Get summary of drift history"""
        if not self.drift_history:
            return {"message": "No drift checks performed yet"}
        
        df = pd.DataFrame(self.drift_history)
        
        return {
            'total_checks': len(df),
            'drift_count': df['drift_detected'].sum(),
            'drift_rate': df['drift_detected'].mean(),
            'latest_check': df.iloc[-1].to_dict(),
            'average_drift_score': df['drift_score'].mean()
        }
```

### Batch Monitoring Pipeline
```python
import logging
from datetime import datetime, timedelta

class DriftMonitoringPipeline:
    def __init__(self, reference_data, monitoring_config):
        self.reference_data = reference_data
        self.config = monitoring_config
        self.detectors = self._initialize_detectors()
        self.alerts = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _initialize_detectors(self):
        """Initialize different drift detectors"""
        return {
            'psi': lambda ref, cur: calculate_psi(ref, cur),
            'ks': lambda ref, cur: ks_drift_test(ref, cur),
            'wasserstein': lambda ref, cur: wasserstein_drift_test(ref, cur)
        }
    
    def monitor_batch(self, batch_data, batch_id=None):
        """Monitor a batch of data for drift"""
        results = {}
        
        for detector_name, detector_func in self.detectors.items():
            try:
                result = detector_func(self.reference_data, batch_data)
                results[detector_name] = result
                
                # Log results
                self.logger.info(f"Batch {batch_id} - {detector_name}: {result}")
                
                # Check for alerts
                if result.get('drift_detected', False):
                    self._trigger_alert(detector_name, result, batch_id)
                    
            except Exception as e:
                self.logger.error(f"Error in {detector_name} detector: {e}")
                results[detector_name] = {'error': str(e)}
        
        return {
            'batch_id': batch_id,
            'timestamp': datetime.now(),
            'results': results,
            'overall_drift': any(r.get('drift_detected', False) for r in results.values())
        }
    
    def _trigger_alert(self, detector_name, result, batch_id):
        """Trigger drift alert"""
        alert = {
            'timestamp': datetime.now(),
            'detector': detector_name,
            'batch_id': batch_id,
            'result': result,
            'severity': self._calculate_severity(result)
        }
        
        self.alerts.append(alert)
        self.logger.warning(f"DRIFT ALERT: {alert}")
        
        # Here you could send notifications (email, Slack, etc.)
        self._send_notification(alert)
    
    def _calculate_severity(self, result):
        """Calculate alert severity based on drift magnitude"""
        if 'psi' in result:
            psi = result['psi']
            if psi > 0.25:
                return 'HIGH'
            elif psi > 0.1:
                return 'MEDIUM'
            else:
                return 'LOW'
        
        return 'MEDIUM'  # Default severity
    
    def _send_notification(self, alert):
        """Send notification (implement based on your needs)"""
        # Example: Send to monitoring system, email, Slack, etc.
        pass
```

## Feature-Level Drift Monitoring

### Individual Feature Monitoring
```python
class FeatureDriftMonitor:
    def __init__(self, reference_data, feature_names):
        self.reference_data = reference_data
        self.feature_names = feature_names
        self.feature_stats = self._calculate_reference_stats()
    
    def _calculate_reference_stats(self):
        """Calculate reference statistics for each feature"""
        stats = {}
        
        for i, feature_name in enumerate(self.feature_names):
            feature_data = self.reference_data[:, i]
            
            if self._is_categorical(feature_data):
                # Categorical feature
                unique_values, counts = np.unique(feature_data, return_counts=True)
                stats[feature_name] = {
                    'type': 'categorical',
                    'distribution': dict(zip(unique_values, counts / len(feature_data)))
                }
            else:
                # Numerical feature
                stats[feature_name] = {
                    'type': 'numerical',
                    'mean': np.mean(feature_data),
                    'std': np.std(feature_data),
                    'min': np.min(feature_data),
                    'max': np.max(feature_data),
                    'percentiles': np.percentile(feature_data, [25, 50, 75])
                }
        
        return stats
    
    def _is_categorical(self, data):
        """Simple heuristic to determine if feature is categorical"""
        return len(np.unique(data)) / len(data) < 0.05
    
    def monitor_features(self, current_data):
        """Monitor drift for each feature"""
        feature_results = {}
        
        for i, feature_name in enumerate(self.feature_names):
            ref_feature = self.reference_data[:, i]
            cur_feature = current_data[:, i]
            
            feature_stats = self.feature_stats[feature_name]
            
            if feature_stats['type'] == 'categorical':
                result = chi2_drift_test(ref_feature.astype(int), cur_feature.astype(int))
            else:
                result = ks_drift_test(ref_feature, cur_feature)
            
            feature_results[feature_name] = result
        
        return feature_results
```

## Model Performance Monitoring

### Accuracy Degradation Detection
```python
class ModelPerformanceMonitor:
    def __init__(self, reference_metrics, threshold=0.05):
        self.reference_metrics = reference_metrics
        self.threshold = threshold
        self.performance_history = []
    
    def monitor_performance(self, y_true, y_pred, timestamp=None):
        """Monitor model performance metrics"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate current metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        current_metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Check for degradation
        degradation_detected = {}
        for metric, value in current_metrics.items():
            if metric in self.reference_metrics:
                degradation = self.reference_metrics[metric] - value
                degradation_detected[metric] = degradation > self.threshold
        
        performance_record = {
            'timestamp': timestamp,
            'metrics': current_metrics,
            'degradation_detected': degradation_detected,
            'overall_degradation': any(degradation_detected.values())
        }
        
        self.performance_history.append(performance_record)
        
        return performance_record
    
    def get_performance_trend(self, metric='accuracy', window=10):
        """Get performance trend for a specific metric"""
        if len(self.performance_history) < window:
            return None
        
        recent_values = [record['metrics'][metric] 
                        for record in self.performance_history[-window:]]
        
        # Simple trend calculation
        trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
        
        return {
            'metric': metric,
            'trend_slope': trend,
            'trend_direction': 'increasing' if trend > 0 else 'decreasing',
            'recent_values': recent_values
        }
```

## Automated Retraining Triggers

### Drift-Based Retraining
```python
class RetrainingTrigger:
    def __init__(self, drift_threshold=0.2, performance_threshold=0.05, 
                 consecutive_alerts=3, time_threshold_days=30):
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        self.consecutive_alerts = consecutive_alerts
        self.time_threshold = timedelta(days=time_threshold_days)
        
        self.drift_alerts = []
        self.performance_alerts = []
        self.last_retrain = datetime.now()
    
    def should_retrain(self, drift_result=None, performance_result=None):
        """Determine if model should be retrained"""
        current_time = datetime.now()
        retrain_reasons = []
        
        # Check drift-based triggers
        if drift_result and drift_result.get('drift_detected', False):
            self.drift_alerts.append(current_time)
            
            # Remove old alerts
            self.drift_alerts = [alert for alert in self.drift_alerts 
                               if current_time - alert < self.time_threshold]
            
            if len(self.drift_alerts) >= self.consecutive_alerts:
                retrain_reasons.append("Consecutive drift alerts")
        
        # Check performance-based triggers
        if performance_result and performance_result.get('overall_degradation', False):
            self.performance_alerts.append(current_time)
            
            # Remove old alerts
            self.performance_alerts = [alert for alert in self.performance_alerts 
                                     if current_time - alert < self.time_threshold]
            
            if len(self.performance_alerts) >= self.consecutive_alerts:
                retrain_reasons.append("Consecutive performance degradation")
        
        # Check time-based trigger
        if current_time - self.last_retrain > self.time_threshold:
            retrain_reasons.append("Time threshold exceeded")
        
        should_retrain = len(retrain_reasons) > 0
        
        if should_retrain:
            self.last_retrain = current_time
            self.drift_alerts = []
            self.performance_alerts = []
        
        return {
            'should_retrain': should_retrain,
            'reasons': retrain_reasons,
            'drift_alert_count': len(self.drift_alerts),
            'performance_alert_count': len(self.performance_alerts)
        }
```

## Production Implementation

### Complete Monitoring System
```python
import json
from pathlib import Path

class ProductionDriftMonitor:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.reference_data = self._load_reference_data()
        self.drift_detector = self._initialize_drift_detector()
        self.performance_monitor = self._initialize_performance_monitor()
        self.retrain_trigger = self._initialize_retrain_trigger()
        
        # Setup storage
        self.results_storage = self.config.get('results_storage', './monitoring_results')
        Path(self.results_storage).mkdir(exist_ok=True)
    
    def _load_config(self, config_path):
        """Load monitoring configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _load_reference_data(self):
        """Load reference data for drift detection"""
        # Implement based on your data storage
        pass
    
    def _initialize_drift_detector(self):
        """Initialize drift detection pipeline"""
        return DriftMonitoringPipeline(
            self.reference_data, 
            self.config['drift_detection']
        )
    
    def _initialize_performance_monitor(self):
        """Initialize performance monitoring"""
        return ModelPerformanceMonitor(
            self.config['reference_metrics'],
            self.config.get('performance_threshold', 0.05)
        )
    
    def _initialize_retrain_trigger(self):
        """Initialize retraining trigger"""
        return RetrainingTrigger(**self.config['retraining'])
    
    def monitor_batch(self, batch_data, batch_labels=None, batch_predictions=None):
        """Monitor a batch of data"""
        timestamp = datetime.now()
        
        # Drift monitoring
        drift_result = self.drift_detector.monitor_batch(batch_data, timestamp)
        
        # Performance monitoring (if labels available)
        performance_result = None
        if batch_labels is not None and batch_predictions is not None:
            performance_result = self.performance_monitor.monitor_performance(
                batch_labels, batch_predictions, timestamp
            )
        
        # Retraining decision
        retrain_decision = self.retrain_trigger.should_retrain(
            drift_result, performance_result
        )
        
        # Combine results
        monitoring_result = {
            'timestamp': timestamp,
            'drift_result': drift_result,
            'performance_result': performance_result,
            'retrain_decision': retrain_decision
        }
        
        # Save results
        self._save_results(monitoring_result)
        
        return monitoring_result
    
    def _save_results(self, result):
        """Save monitoring results"""
        filename = f"{result['timestamp'].strftime('%Y%m%d_%H%M%S')}_monitoring.json"
        filepath = Path(self.results_storage) / filename
        
        # Convert datetime objects to strings for JSON serialization
        serializable_result = json.loads(json.dumps(result, default=str))
        
        with open(filepath, 'w') as f:
            json.dump(serializable_result, f, indent=2)
```

## Best Practices

1. **Choose Appropriate Methods**: Select drift detection methods based on data types and use case
2. **Set Realistic Thresholds**: Tune thresholds to balance false positives and missed drift
3. **Monitor Multiple Dimensions**: Track data drift, concept drift, and performance degradation
4. **Implement Gradual Response**: Don't retrain immediately on first drift signal
5. **Keep Historical Data**: Maintain reference datasets and monitoring history
6. **Automate Alerting**: Set up notifications for drift detection
7. **Regular Review**: Periodically review and update monitoring strategies
8. **Feature-Level Monitoring**: Monitor individual features to identify drift sources

## Tools and Frameworks

- **Evidently AI**: Open-source ML monitoring
- **Alibi Detect**: Drift detection library
- **Great Expectations**: Data validation and monitoring
- **Weights & Biases**: Experiment tracking with drift monitoring
- **MLflow**: Model monitoring and lifecycle management
- **WhyLabs**: Data and ML monitoring platform
- **Arize AI**: ML observability platform

## Resources

- **Research Papers**: Latest drift detection algorithms
- **Open Source Libraries**: Evidently, Alibi Detect, River
- **Cloud Services**: AWS SageMaker Model Monitor, Azure ML monitoring
- **Best Practices**: MLOps drift monitoring guidelines
- **Case Studies**: Real-world drift detection implementations
