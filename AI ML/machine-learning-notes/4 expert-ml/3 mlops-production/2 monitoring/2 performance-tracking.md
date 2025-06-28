# Performance Tracking in MLOps

## Overview
Performance tracking in MLOps involves continuous monitoring of model behavior, system metrics, and business outcomes to ensure deployed models maintain their effectiveness over time. It encompasses both technical performance (latency, throughput) and model performance (accuracy, precision, recall).

## Key Performance Metrics

### Model Performance Metrics
#### Classification Metrics
- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: Detailed breakdown of prediction results

#### Regression Metrics
- **MAE (Mean Absolute Error)**: Average absolute difference
- **MSE (Mean Squared Error)**: Average squared difference
- **RMSE (Root Mean Squared Error)**: Square root of MSE
- **R²**: Coefficient of determination
- **MAPE (Mean Absolute Percentage Error)**: Percentage-based error

### System Performance Metrics
- **Latency**: Response time for predictions
- **Throughput**: Requests processed per second
- **CPU/Memory Usage**: Resource utilization
- **Error Rate**: Percentage of failed requests
- **Availability/Uptime**: System availability percentage

## Implementation Strategies

### Real-time Performance Monitoring
```python
import time
import logging
from functools import wraps
from typing import Dict, Any
import psutil
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge

# Prometheus metrics
PREDICTION_COUNTER = Counter('ml_predictions_total', 'Total predictions', ['model_version', 'status'])
PREDICTION_LATENCY = Histogram('ml_prediction_duration_seconds', 'Prediction latency')
MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Current model accuracy', ['model_version'])
SYSTEM_MEMORY = Gauge('ml_system_memory_usage_bytes', 'Memory usage')
SYSTEM_CPU = Gauge('ml_system_cpu_usage_percent', 'CPU usage')

class PerformanceTracker:
    def __init__(self):
        self.predictions = []
        self.ground_truth = []
        self.latencies = []
        self.logger = logging.getLogger(__name__)
    
    def track_prediction(self, prediction_time: float, prediction: Any, 
                        ground_truth: Any = None, model_version: str = "v1"):
        """Track individual prediction performance"""
        
        # Record latency
        PREDICTION_LATENCY.observe(prediction_time)
        self.latencies.append(prediction_time)
        
        # Record prediction
        PREDICTION_COUNTER.labels(model_version=model_version, status='success').inc()
        
        if ground_truth is not None:
            self.predictions.append(prediction)
            self.ground_truth.append(ground_truth)
            
            # Update accuracy if we have enough samples
            if len(self.predictions) >= 100:
                accuracy = self.calculate_accuracy()
                MODEL_ACCURACY.labels(model_version=model_version).set(accuracy)
        
        # Log performance
        self.logger.info(f"Prediction tracked: latency={prediction_time:.3f}s, "
                        f"model_version={model_version}")
    
    def calculate_accuracy(self):
        """Calculate current accuracy"""
        if not self.predictions or not self.ground_truth:
            return 0.0
        
        correct = sum(1 for p, t in zip(self.predictions, self.ground_truth) if p == t)
        return correct / len(self.predictions)
    
    def track_system_metrics(self):
        """Track system resource usage"""
        memory_usage = psutil.virtual_memory().used
        cpu_usage = psutil.cpu_percent()
        
        SYSTEM_MEMORY.set(memory_usage)
        SYSTEM_CPU.set(cpu_usage)
        
        return {
            'memory_usage_bytes': memory_usage,
            'cpu_usage_percent': cpu_usage
        }

# Decorator for automatic performance tracking
def track_performance(tracker: PerformanceTracker, model_version: str = "v1"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                
                # Track successful prediction
                tracker.track_prediction(
                    prediction_time=end_time - start_time,
                    prediction=result,
                    model_version=model_version
                )
                
                return result
            
            except Exception as e:
                # Track failed prediction
                PREDICTION_COUNTER.labels(model_version=model_version, status='error').inc()
                tracker.logger.error(f"Prediction failed: {str(e)}")
                raise
        
        return wrapper
    return decorator

# Usage example
tracker = PerformanceTracker()

@track_performance(tracker, model_version="v2.1")
def predict(features):
    # Your model prediction logic here
    prediction = model.predict(features)
    return prediction
```

### Batch Performance Evaluation
```python
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from datetime import datetime, timedelta

class BatchPerformanceEvaluator:
    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
        self.performance_history = []
    
    def evaluate_batch(self, predictions: list, ground_truth: list, 
                      timestamp: datetime = None) -> Dict[str, float]:
        """Evaluate performance on a batch of predictions"""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate metrics
        accuracy = accuracy_score(ground_truth, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            ground_truth, predictions, average='weighted'
        )
        
        # Calculate confusion matrix
        cm = confusion_matrix(ground_truth, predictions)
        
        metrics = {
            'timestamp': timestamp,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'sample_count': len(predictions),
            'confusion_matrix': cm.tolist()
        }
        
        self.performance_history.append(metrics)
        
        return metrics
    
    def get_performance_trend(self, days: int = 7) -> pd.DataFrame:
        """Get performance trend over specified days"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_metrics = [
            m for m in self.performance_history 
            if m['timestamp'] >= cutoff_date
        ]
        
        if not recent_metrics:
            return pd.DataFrame()
        
        df = pd.DataFrame(recent_metrics)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def detect_performance_degradation(self, threshold: float = 0.05) -> bool:
        """Detect if performance has degraded significantly"""
        
        if len(self.performance_history) < 2:
            return False
        
        recent_accuracy = self.performance_history[-1]['accuracy']
        baseline_accuracy = np.mean([
            m['accuracy'] for m in self.performance_history[-10:-1]
        ])
        
        degradation = baseline_accuracy - recent_accuracy
        
        return degradation > threshold
```

### Real-time Dashboard Integration
```python
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

class PerformanceDashboard:
    def __init__(self, tracker: PerformanceTracker):
        self.tracker = tracker
    
    def create_dashboard(self):
        """Create Streamlit dashboard for performance monitoring"""
        
        st.title("ML Model Performance Dashboard")
        
        # Real-time metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_accuracy = self.tracker.calculate_accuracy()
            st.metric("Current Accuracy", f"{current_accuracy:.3f}")
        
        with col2:
            avg_latency = np.mean(self.tracker.latencies[-100:]) if self.tracker.latencies else 0
            st.metric("Avg Latency", f"{avg_latency:.3f}s")
        
        with col3:
            system_metrics = self.tracker.track_system_metrics()
            st.metric("CPU Usage", f"{system_metrics['cpu_usage_percent']:.1f}%")
        
        with col4:
            memory_gb = system_metrics['memory_usage_bytes'] / (1024**3)
            st.metric("Memory Usage", f"{memory_gb:.2f}GB")
        
        # Performance over time
        self.plot_performance_trend()
        
        # Latency distribution
        self.plot_latency_distribution()
        
        # System resource usage
        self.plot_system_metrics()
    
    def plot_performance_trend(self):
        """Plot accuracy trend over time"""
        st.subheader("Accuracy Trend")
        
        if len(self.tracker.predictions) < 10:
            st.warning("Not enough data for trend analysis")
            return
        
        # Calculate rolling accuracy
        window_size = 50
        rolling_accuracy = []
        timestamps = []
        
        for i in range(window_size, len(self.tracker.predictions)):
            window_preds = self.tracker.predictions[i-window_size:i]
            window_truth = self.tracker.ground_truth[i-window_size:i]
            
            accuracy = sum(1 for p, t in zip(window_preds, window_truth) if p == t) / window_size
            rolling_accuracy.append(accuracy)
            timestamps.append(datetime.now() - timedelta(minutes=len(self.tracker.predictions)-i))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=rolling_accuracy,
            mode='lines',
            name='Rolling Accuracy',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title="Model Accuracy Over Time",
            xaxis_title="Time",
            yaxis_title="Accuracy",
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_latency_distribution(self):
        """Plot latency distribution"""
        st.subheader("Prediction Latency Distribution")
        
        if not self.tracker.latencies:
            st.warning("No latency data available")
            return
        
        fig = px.histogram(
            x=self.tracker.latencies[-1000:],  # Last 1000 predictions
            nbins=50,
            title="Latency Distribution",
            labels={'x': 'Latency (seconds)', 'y': 'Count'}
        )
        
        fig.add_vline(
            x=np.mean(self.tracker.latencies[-1000:]),
            line_dash="dash",
            line_color="red",
            annotation_text="Mean"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_system_metrics(self):
        """Plot system resource usage"""
        st.subheader("System Resource Usage")
        
        # This would typically pull from a time series database
        # For demo, showing current values
        system_metrics = self.tracker.track_system_metrics()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_cpu = go.Figure(go.Indicator(
                mode="gauge+number",
                value=system_metrics['cpu_usage_percent'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "CPU Usage (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "red"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}
            ))
            st.plotly_chart(fig_cpu, use_container_width=True)
        
        with col2:
            memory_gb = system_metrics['memory_usage_bytes'] / (1024**3)
            fig_memory = go.Figure(go.Indicator(
                mode="gauge+number",
                value=memory_gb,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Memory Usage (GB)"},
                gauge={'axis': {'range': [None, 16]},  # Assuming 16GB max
                       'bar': {'color': "darkgreen"}}
            ))
            st.plotly_chart(fig_memory, use_container_width=True)
```

## Alerting and Notification Systems

### Performance Alert System
```python
import smtplib
from email.mime.text import MIMEText
from dataclasses import dataclass
from typing import List, Callable
import logging

@dataclass
class AlertRule:
    name: str
    metric: str
    threshold: float
    condition: str  # 'greater_than', 'less_than', 'equals'
    severity: str  # 'critical', 'warning', 'info'
    
@dataclass
class Alert:
    rule_name: str
    metric_value: float
    threshold: float
    severity: str
    timestamp: datetime
    message: str

class AlertManager:
    def __init__(self):
        self.rules: List[AlertRule] = []
        self.alert_handlers: List[Callable] = []
        self.logger = logging.getLogger(__name__)
    
    def add_rule(self, rule: AlertRule):
        """Add an alerting rule"""
        self.rules.append(rule)
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def add_handler(self, handler: Callable):
        """Add an alert handler (email, Slack, etc.)"""
        self.alert_handlers.append(handler)
    
    def check_metrics(self, metrics: Dict[str, float]):
        """Check metrics against all rules and trigger alerts"""
        
        for rule in self.rules:
            if rule.metric not in metrics:
                continue
            
            metric_value = metrics[rule.metric]
            triggered = False
            
            if rule.condition == 'greater_than' and metric_value > rule.threshold:
                triggered = True
            elif rule.condition == 'less_than' and metric_value < rule.threshold:
                triggered = True
            elif rule.condition == 'equals' and metric_value == rule.threshold:
                triggered = True
            
            if triggered:
                alert = Alert(
                    rule_name=rule.name,
                    metric_value=metric_value,
                    threshold=rule.threshold,
                    severity=rule.severity,
                    timestamp=datetime.now(),
                    message=f"{rule.metric} is {metric_value}, threshold: {rule.threshold}"
                )
                
                self.trigger_alert(alert)
    
    def trigger_alert(self, alert: Alert):
        """Trigger an alert using all registered handlers"""
        
        self.logger.warning(f"ALERT: {alert.rule_name} - {alert.message}")
        
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {e}")

def email_alert_handler(alert: Alert, smtp_config: Dict[str, str]):
    """Send email alert"""
    
    subject = f"[{alert.severity.upper()}] ML Model Alert: {alert.rule_name}"
    body = f"""
    Alert: {alert.rule_name}
    Severity: {alert.severity}
    Time: {alert.timestamp}
    
    {alert.message}
    
    Current Value: {alert.metric_value}
    Threshold: {alert.threshold}
    """
    
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = smtp_config['from']
    msg['To'] = smtp_config['to']
    
    with smtplib.SMTP(smtp_config['host'], smtp_config['port']) as server:
        server.starttls()
        server.login(smtp_config['username'], smtp_config['password'])
        server.send_message(msg)

def slack_alert_handler(alert: Alert, webhook_url: str):
    """Send Slack alert"""
    import requests
    
    color = {
        'critical': '#FF0000',
        'warning': '#FFA500',
        'info': '#0000FF'
    }.get(alert.severity, '#808080')
    
    payload = {
        "attachments": [
            {
                "color": color,
                "title": f"ML Model Alert: {alert.rule_name}",
                "text": alert.message,
                "fields": [
                    {"title": "Severity", "value": alert.severity, "short": True},
                    {"title": "Current Value", "value": str(alert.metric_value), "short": True},
                    {"title": "Threshold", "value": str(alert.threshold), "short": True},
                    {"title": "Time", "value": alert.timestamp.isoformat(), "short": True}
                ]
            }
        ]
    }
    
    requests.post(webhook_url, json=payload)
```

### Usage Example
```python
# Setup performance tracking with alerts
tracker = PerformanceTracker()
alert_manager = AlertManager()

# Define alert rules
alert_manager.add_rule(AlertRule(
    name="Low Accuracy",
    metric="accuracy",
    threshold=0.85,
    condition="less_than",
    severity="warning"
))

alert_manager.add_rule(AlertRule(
    name="High Latency",
    metric="avg_latency",
    threshold=1.0,
    condition="greater_than",
    severity="critical"
))

alert_manager.add_rule(AlertRule(
    name="High CPU Usage",
    metric="cpu_usage_percent",
    threshold=80.0,
    condition="greater_than",
    severity="warning"
))

# Add alert handlers
smtp_config = {
    'host': 'smtp.gmail.com',
    'port': 587,
    'username': 'your-email@gmail.com',
    'password': 'your-password',
    'from': 'alerts@yourcompany.com',
    'to': 'team@yourcompany.com'
}

alert_manager.add_handler(lambda alert: email_alert_handler(alert, smtp_config))
alert_manager.add_handler(lambda alert: slack_alert_handler(alert, "YOUR_SLACK_WEBHOOK_URL"))

# Periodic metric checking
def check_performance():
    current_accuracy = tracker.calculate_accuracy()
    avg_latency = np.mean(tracker.latencies[-100:]) if tracker.latencies else 0
    system_metrics = tracker.track_system_metrics()
    
    metrics = {
        'accuracy': current_accuracy,
        'avg_latency': avg_latency,
        'cpu_usage_percent': system_metrics['cpu_usage_percent']
    }
    
    alert_manager.check_metrics(metrics)

# Run check every 5 minutes
import schedule
schedule.every(5).minutes.do(check_performance)
```

## Integration with MLOps Platforms

### MLflow Integration
```python
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

class MLflowPerformanceTracker:
    def __init__(self, experiment_name: str):
        self.client = MlflowClient()
        self.experiment_name = experiment_name
        
        # Create experiment if it doesn't exist
        try:
            self.experiment_id = self.client.create_experiment(experiment_name)
        except:
            self.experiment_id = self.client.get_experiment_by_name(experiment_name).experiment_id
    
    def log_performance_metrics(self, metrics: Dict[str, float], run_name: str = None):
        """Log performance metrics to MLflow"""
        
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name):
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
            
            # Log timestamp
            mlflow.log_param("timestamp", datetime.now().isoformat())
    
    def compare_model_versions(self, model_name: str) -> pd.DataFrame:
        """Compare performance across model versions"""
        
        models = self.client.search_model_versions(f"name='{model_name}'")
        
        results = []
        for model in models:
            run_id = model.run_id
            run = self.client.get_run(run_id)
            
            metrics = run.data.metrics
            version = model.version
            
            result = {'version': version, **metrics}
            results.append(result)
        
        return pd.DataFrame(results)
```

### Weights & Biases Integration
```python
import wandb

class WandBPerformanceTracker:
    def __init__(self, project_name: str, entity: str = None):
        self.project_name = project_name
        self.entity = entity
        self.run = None
    
    def start_tracking(self, run_name: str = None, config: Dict = None):
        """Start a new W&B run"""
        self.run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=run_name,
            config=config
        )
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to W&B"""
        if self.run:
            wandb.log(metrics, step=step)
    
    def log_confusion_matrix(self, y_true, y_pred, class_names: List[str]):
        """Log confusion matrix visualization"""
        if self.run:
            wandb.log({
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true,
                    preds=y_pred,
                    class_names=class_names
                )
            })
    
    def finish_tracking(self):
        """Finish the current run"""
        if self.run:
            wandb.finish()
```

## Best Practices

### 1. Establish Baseline Metrics
```python
class BaselineEstablisher:
    def __init__(self, validation_data):
        self.validation_data = validation_data
        self.baseline_metrics = {}
    
    def establish_baseline(self, model):
        """Establish baseline performance metrics"""
        predictions = model.predict(self.validation_data['X'])
        
        self.baseline_metrics = {
            'accuracy': accuracy_score(self.validation_data['y'], predictions),
            'precision': precision_score(self.validation_data['y'], predictions, average='weighted'),
            'recall': recall_score(self.validation_data['y'], predictions, average='weighted'),
            'f1': f1_score(self.validation_data['y'], predictions, average='weighted')
        }
        
        return self.baseline_metrics
```

### 2. Automated Performance Reports
```python
class PerformanceReporter:
    def __init__(self, tracker: PerformanceTracker):
        self.tracker = tracker
    
    def generate_daily_report(self) -> str:
        """Generate daily performance report"""
        
        current_accuracy = self.tracker.calculate_accuracy()
        avg_latency = np.mean(self.tracker.latencies[-1000:]) if self.tracker.latencies else 0
        prediction_count = len(self.tracker.predictions)
        
        report = f"""
        Daily ML Model Performance Report
        ================================
        Date: {datetime.now().strftime('%Y-%m-%d')}
        
        Model Performance:
        - Current Accuracy: {current_accuracy:.3f}
        - Predictions Made: {prediction_count}
        
        System Performance:
        - Average Latency: {avg_latency:.3f}s
        - Total Requests: {prediction_count}
        
        Recommendations:
        {self._generate_recommendations(current_accuracy, avg_latency)}
        """
        
        return report
    
    def _generate_recommendations(self, accuracy: float, latency: float) -> str:
        """Generate performance recommendations"""
        recommendations = []
        
        if accuracy < 0.85:
            recommendations.append("- Consider retraining the model due to low accuracy")
        
        if latency > 0.5:
            recommendations.append("- Optimize model inference for better latency")
        
        if not recommendations:
            recommendations.append("- Performance is within acceptable thresholds")
        
        return "\n".join(recommendations)
```

## Common Pitfalls and Solutions

### 1. Metric Selection
**Pitfall**: Tracking too many or irrelevant metrics
**Solution**: Focus on business-critical metrics and establish clear thresholds

### 2. Alert Fatigue
**Pitfall**: Too many false positive alerts
**Solution**: Implement alert suppression and intelligent thresholds

### 3. Performance Degradation Detection
**Pitfall**: Missing gradual performance degradation
**Solution**: Use statistical process control and trend analysis

### 4. Data Quality Issues
**Pitfall**: Not tracking data quality metrics
**Solution**: Include data validation and quality metrics in tracking

## Tools and Platforms

- **Prometheus + Grafana**: Open-source monitoring stack
- **DataDog**: Commercial monitoring platform
- **New Relic**: Application performance monitoring
- **MLflow**: Open-source MLOps platform
- **Weights & Biases**: Experiment tracking and monitoring
- **TensorBoard**: TensorFlow's visualization toolkit
- **Evidently AI**: ML model monitoring
- **Arize AI**: ML observability platform

## Resources

- **MLOps Best Practices**: Industry guidelines for model monitoring
- **Prometheus Documentation**: Metrics collection and alerting
- **Grafana Tutorials**: Dashboard creation and visualization
- **Statistical Process Control**: Methods for detecting performance changes
- **A/B Testing**: Comparing model performance
- **Model Validation**: Ensuring model reliability in production
