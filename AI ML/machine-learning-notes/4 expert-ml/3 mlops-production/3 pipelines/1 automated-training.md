# Automated Training Pipelines

## Learning Objectives
- Understand automated training pipeline concepts and architecture
- Implement CI/CD for machine learning workflows
- Build scalable training pipelines with orchestration tools
- Integrate data validation, model training, and deployment
- Implement automated retraining and model versioning
- Monitor and manage pipeline execution and failures

## Introduction

Automated training pipelines enable continuous integration and deployment (CI/CD) for machine learning models, ensuring reproducible, scalable, and reliable model development and deployment processes.

### Key Benefits
- **Reproducibility**: Consistent training environments and processes
- **Scalability**: Handle large datasets and complex workflows
- **Efficiency**: Reduce manual intervention and errors
- **Monitoring**: Track pipeline execution and model performance
- **Versioning**: Manage model and data versions systematically

## Core Pipeline Components

### 1. Pipeline Architecture
```python
import os
import yaml
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Pipeline configuration
@dataclass
class PipelineConfig:
    """Configuration for training pipeline"""
    project_name: str
    model_type: str
    data_source: str
    target_column: str
    test_size: float
    random_state: int
    model_params: Dict[str, Any]
    validation_threshold: float
    artifact_path: str
    
    def save(self, path: str):
        """Save configuration to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

# Pipeline step base class
class PipelineStep:
    """Base class for pipeline steps"""
    
    def __init__(self, name: str, config: PipelineConfig):
        self.name = name
        self.config = config
        self.logger = self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for pipeline step"""
        logger = logging.getLogger(f"{self.config.project_name}.{self.name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def execute(self, input_data: Any = None) -> Any:
        """Execute pipeline step"""
        raise NotImplementedError("Subclasses must implement execute method")
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data"""
        return True
    
    def validate_output(self, output_data: Any) -> bool:
        """Validate output data"""
        return True

# Example configuration
def create_sample_config():
    """Create sample pipeline configuration"""
    config = PipelineConfig(
        project_name="ml_pipeline_demo",
        model_type="random_forest",
        data_source="data/dataset.csv",
        target_column="target",
        test_size=0.2,
        random_state=42,
        model_params={
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5
        },
        validation_threshold=0.8,
        artifact_path="artifacts/"
    )
    
    return config
```

### 2. Data Pipeline Steps
```python
class DataIngestionStep(PipelineStep):
    """Data ingestion pipeline step"""
    
    def execute(self, input_data: Any = None) -> pd.DataFrame:
        """Load and validate data"""
        self.logger.info(f"Loading data from {self.config.data_source}")
        
        try:
            # Load data
            if self.config.data_source.endswith('.csv'):
                data = pd.read_csv(self.config.data_source)
            elif self.config.data_source.endswith('.parquet'):
                data = pd.read_parquet(self.config.data_source)
            else:
                raise ValueError(f"Unsupported data format: {self.config.data_source}")
            
            # Basic validation
            if not self.validate_input(data):
                raise ValueError("Data validation failed")
            
            self.logger.info(f"Successfully loaded {len(data)} rows")
            return data
            
        except Exception as e:
            self.logger.error(f"Data ingestion failed: {str(e)}")
            raise
    
    def validate_input(self, data: pd.DataFrame) -> bool:
        """Validate loaded data"""
        checks = [
            len(data) > 0,  # Non-empty dataset
            self.config.target_column in data.columns,  # Target column exists
            data[self.config.target_column].notna().sum() > 0  # Target has non-null values
        ]
        
        if not all(checks):
            self.logger.error("Data validation failed")
            return False
        
        self.logger.info("Data validation passed")
        return True

class DataValidationStep(PipelineStep):
    """Data validation pipeline step"""
    
    def execute(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean data"""
        self.logger.info("Starting data validation")
        
        # Validation checks
        validation_results = self._run_validation_checks(input_data)
        
        if not validation_results['passed']:
            raise ValueError(f"Data validation failed: {validation_results['errors']}")
        
        # Data cleaning
        cleaned_data = self._clean_data(input_data)
        
        self.logger.info("Data validation completed successfully")
        return cleaned_data
    
    def _run_validation_checks(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive data validation checks"""
        results = {'passed': True, 'errors': [], 'warnings': []}
        
        # Check for missing values
        missing_percent = data.isnull().sum() / len(data) * 100
        high_missing = missing_percent[missing_percent > 50]
        
        if len(high_missing) > 0:
            results['warnings'].append(f"Columns with >50% missing: {list(high_missing.index)}")
        
        # Check target distribution
        target_dist = data[self.config.target_column].value_counts()
        min_class_size = target_dist.min()
        
        if min_class_size < 10:
            results['errors'].append(f"Insufficient samples in some classes: {target_dist.to_dict()}")
            results['passed'] = False
        
        # Check for duplicates
        duplicate_count = data.duplicated().sum()
        if duplicate_count > len(data) * 0.1:
            results['warnings'].append(f"High duplicate rate: {duplicate_count} duplicates")
        
        # Schema validation
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            results['errors'].append("No numeric features found")
            results['passed'] = False
        
        return results
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data"""
        cleaned_data = data.copy()
        
        # Remove duplicates
        cleaned_data = cleaned_data.drop_duplicates()
        
        # Handle missing values
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        categorical_cols = cleaned_data.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_cols:
            if col != self.config.target_column:
                cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].median())
        
        # Fill categorical missing values with mode
        for col in categorical_cols:
            if col != self.config.target_column:
                mode_value = cleaned_data[col].mode()
                if len(mode_value) > 0:
                    cleaned_data[col] = cleaned_data[col].fillna(mode_value[0])
        
        return cleaned_data

class FeatureEngineeringStep(PipelineStep):
    """Feature engineering pipeline step"""
    
    def execute(self, input_data: pd.DataFrame) -> Dict[str, Any]:
        """Engineer features and prepare data for training"""
        self.logger.info("Starting feature engineering")
        
        # Separate features and target
        X = input_data.drop(columns=[self.config.target_column])
        y = input_data[self.config.target_column]
        
        # Feature engineering
        X_engineered = self._engineer_features(X)
        
        # Encode categorical variables
        X_encoded = self._encode_categorical_features(X_engineered)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y if len(y.unique()) > 1 else None
        )
        
        self.logger.info(f"Feature engineering completed. Features: {X_encoded.shape[1]}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': list(X_encoded.columns)
        }
    
    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create new features"""
        X_engineered = X.copy()
        
        # Generate polynomial features for numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Square features
            X_engineered[f'{col}_squared'] = X[col] ** 2
            
            # Log features (for positive values)
            if (X[col] > 0).all():
                X_engineered[f'{col}_log'] = np.log(X[col])
        
        # Interaction features
        if len(numeric_cols) >= 2:
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    X_engineered[f'{col1}_{col2}_interaction'] = X[col1] * X[col2]
        
        return X_engineered
    
    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        
        X_encoded = X.copy()
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            # Use one-hot encoding for low cardinality, label encoding for high cardinality
            unique_values = X[col].nunique()
            
            if unique_values <= 10:
                # One-hot encoding
                dummies = pd.get_dummies(X[col], prefix=col)
                X_encoded = pd.concat([X_encoded.drop(columns=[col]), dummies], axis=1)
            else:
                # Label encoding
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col].astype(str))
        
        return X_encoded
```

### 3. Model Training Steps
```python
class ModelTrainingStep(PipelineStep):
    """Model training pipeline step"""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train machine learning model"""
        self.logger.info("Starting model training")
        
        X_train = input_data['X_train']
        y_train = input_data['y_train']
        
        # Initialize model
        model = self._create_model()
        
        # Train model
        model.fit(X_train, y_train)
        
        # Generate training metadata
        metadata = {
            'training_timestamp': datetime.now().isoformat(),
            'training_samples': len(X_train),
            'feature_count': X_train.shape[1],
            'feature_names': input_data['feature_names'],
            'model_params': self.config.model_params
        }
        
        self.logger.info("Model training completed")
        
        return {
            'model': model,
            'metadata': metadata,
            'training_data': input_data
        }
    
    def _create_model(self):
        """Create model instance based on configuration"""
        if self.config.model_type == 'random_forest':
            return RandomForestClassifier(**self.config.model_params)
        elif self.config.model_type == 'svm':
            from sklearn.svm import SVC
            return SVC(**self.config.model_params)
        elif self.config.model_type == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**self.config.model_params)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")

class ModelEvaluationStep(PipelineStep):
    """Model evaluation pipeline step"""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate trained model"""
        self.logger.info("Starting model evaluation")
        
        model = input_data['model']
        training_data = input_data['training_data']
        
        X_test = training_data['X_test']
        y_test = training_data['y_test']
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Validation check
        if not self._validate_model_performance(metrics):
            raise ValueError(f"Model performance below threshold: {metrics}")
        
        self.logger.info(f"Model evaluation completed. Accuracy: {metrics['accuracy']:.4f}")
        
        return {
            'model': model,
            'metadata': input_data['metadata'],
            'metrics': metrics,
            'predictions': {
                'y_true': y_test.tolist(),
                'y_pred': y_pred.tolist(),
                'y_pred_proba': y_pred_proba.tolist() if y_pred_proba is not None else None
            }
        }
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba=None) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Add AUC for binary classification
        if len(np.unique(y_true)) == 2 and y_pred_proba is not None:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def _validate_model_performance(self, metrics: Dict[str, float]) -> bool:
        """Validate that model meets performance threshold"""
        accuracy = metrics.get('accuracy', 0)
        return accuracy >= self.config.validation_threshold

class ModelPersistenceStep(PipelineStep):
    """Model persistence pipeline step"""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save model and artifacts"""
        self.logger.info("Starting model persistence")
        
        model = input_data['model']
        metadata = input_data['metadata']
        metrics = input_data['metrics']
        
        # Create artifact directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifact_dir = Path(self.config.artifact_path) / f"model_{timestamp}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = artifact_dir / "model.joblib"
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata_path = artifact_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save metrics
        metrics_path = artifact_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save predictions
        if 'predictions' in input_data:
            predictions_path = artifact_dir / "predictions.json"
            with open(predictions_path, 'w') as f:
                json.dump(input_data['predictions'], f, indent=2)
        
        # Create model registry entry
        registry_entry = {
            'model_id': f"{self.config.project_name}_{timestamp}",
            'model_path': str(model_path),
            'metadata_path': str(metadata_path),
            'metrics': metrics,
            'timestamp': timestamp,
            'status': 'trained'
        }
        
        self.logger.info(f"Model saved to {artifact_dir}")
        
        return {
            'model_id': registry_entry['model_id'],
            'artifact_dir': str(artifact_dir),
            'registry_entry': registry_entry
        }
```

### 4. Pipeline Orchestration
```python
class TrainingPipeline:
    """Main training pipeline orchestrator"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.steps = self._initialize_steps()
        self.execution_log = []
    
    def _setup_logging(self):
        """Setup pipeline logging"""
        logger = logging.getLogger(f"{self.config.project_name}.pipeline")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler
            log_dir = Path(self.config.artifact_path) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _initialize_steps(self) -> List[PipelineStep]:
        """Initialize pipeline steps"""
        return [
            DataIngestionStep("data_ingestion", self.config),
            DataValidationStep("data_validation", self.config),
            FeatureEngineeringStep("feature_engineering", self.config),
            ModelTrainingStep("model_training", self.config),
            ModelEvaluationStep("model_evaluation", self.config),
            ModelPersistenceStep("model_persistence", self.config)
        ]
    
    def run(self) -> Dict[str, Any]:
        """Execute the complete training pipeline"""
        self.logger.info("Starting training pipeline")
        pipeline_start_time = datetime.now()
        
        try:
            # Execute pipeline steps
            current_data = None
            
            for step in self.steps:
                step_start_time = datetime.now()
                self.logger.info(f"Executing step: {step.name}")
                
                try:
                    current_data = step.execute(current_data)
                    step_duration = (datetime.now() - step_start_time).total_seconds()
                    
                    self.execution_log.append({
                        'step': step.name,
                        'status': 'success',
                        'duration': step_duration,
                        'timestamp': step_start_time.isoformat()
                    })
                    
                    self.logger.info(f"Step {step.name} completed in {step_duration:.2f}s")
                    
                except Exception as e:
                    step_duration = (datetime.now() - step_start_time).total_seconds()
                    error_msg = f"Step {step.name} failed: {str(e)}"
                    
                    self.execution_log.append({
                        'step': step.name,
                        'status': 'failed',
                        'error': str(e),
                        'duration': step_duration,
                        'timestamp': step_start_time.isoformat()
                    })
                    
                    self.logger.error(error_msg)
                    raise
            
            pipeline_duration = (datetime.now() - pipeline_start_time).total_seconds()
            
            # Pipeline summary
            pipeline_result = {
                'status': 'success',
                'duration': pipeline_duration,
                'execution_log': self.execution_log,
                'result': current_data
            }
            
            self.logger.info(f"Pipeline completed successfully in {pipeline_duration:.2f}s")
            return pipeline_result
            
        except Exception as e:
            pipeline_duration = (datetime.now() - pipeline_start_time).total_seconds()
            
            pipeline_result = {
                'status': 'failed',
                'error': str(e),
                'duration': pipeline_duration,
                'execution_log': self.execution_log
            }
            
            self.logger.error(f"Pipeline failed after {pipeline_duration:.2f}s: {str(e)}")
            return pipeline_result
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get pipeline execution summary"""
        if not self.execution_log:
            return {'message': 'No execution data available'}
        
        total_duration = sum(step['duration'] for step in self.execution_log)
        successful_steps = [step for step in self.execution_log if step['status'] == 'success']
        failed_steps = [step for step in self.execution_log if step['status'] == 'failed']
        
        return {
            'total_steps': len(self.execution_log),
            'successful_steps': len(successful_steps),
            'failed_steps': len(failed_steps),
            'total_duration': total_duration,
            'step_details': self.execution_log
        }

# Pipeline execution example
def run_training_pipeline_example():
    """Example of running a complete training pipeline"""
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(1000, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Create sample dataset
    sample_data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    sample_data['target'] = y
    
    # Save sample data
    os.makedirs('data', exist_ok=True)
    sample_data.to_csv('data/dataset.csv', index=False)
    
    # Create configuration
    config = create_sample_config()
    
    # Create and run pipeline
    pipeline = TrainingPipeline(config)
    result = pipeline.run()
    
    # Print results
    print("Pipeline Execution Summary:")
    print(json.dumps(pipeline.get_execution_summary(), indent=2))
    
    if result['status'] == 'success':
        print(f"\nModel ID: {result['result']['model_id']}")
        print(f"Artifacts saved to: {result['result']['artifact_dir']}")
    
    return result, pipeline

# Run example
# result, pipeline = run_training_pipeline_example()
```

## CI/CD Integration

### 1. GitHub Actions Workflow
```yaml
# .github/workflows/ml-training.yml
name: ML Training Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    # Run daily at 2 AM UTC
    - cron: '0 2 * * *'

jobs:
  training:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run data validation
      run: |
        python scripts/validate_data.py
    
    - name: Run training pipeline
      run: |
        python scripts/train_model.py
      env:
        MODEL_CONFIG: ${{ secrets.MODEL_CONFIG }}
        DATA_SOURCE: ${{ secrets.DATA_SOURCE }}
    
    - name: Run model tests
      run: |
        python -m pytest tests/test_model.py -v
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: model-artifacts
        path: artifacts/
    
    - name: Deploy model
      if: github.ref == 'refs/heads/main'
      run: |
        python scripts/deploy_model.py
      env:
        DEPLOYMENT_KEY: ${{ secrets.DEPLOYMENT_KEY }}
```

### 2. Docker Integration
```python
# Dockerfile for training pipeline
dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p artifacts/logs data

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run training pipeline
CMD ["python", "scripts/train_model.py"]
"""

# Docker Compose for multi-service setup
docker_compose_content = """
version: '3.8'

services:
  training:
    build: .
    volumes:
      - ./data:/app/data
      - ./artifacts:/app/artifacts
      - ./config:/app/config
    environment:
      - CONFIG_PATH=/app/config/pipeline_config.yaml
    depends_on:
      - mlflow
      - postgres
  
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    environment:
      - BACKEND_STORE_URI=postgresql://mlflow:mlflow@postgres/mlflow
      - ARTIFACT_ROOT=/mlflow/artifacts
    volumes:
      - ./mlflow:/mlflow/artifacts
    depends_on:
      - postgres
    command: >
      mlflow server
      --backend-store-uri postgresql://mlflow:mlflow@postgres/mlflow
      --default-artifact-root /mlflow/artifacts
      --host 0.0.0.0
      --port 5000
  
  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=mlflow
      - POSTGRES_USER=mlflow
      - POSTGRES_PASSWORD=mlflow
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
"""

def create_docker_files():
    """Create Docker configuration files"""
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile_content)
    
    with open('docker-compose.yml', 'w') as f:
        f.write(docker_compose_content)
    
    print("Docker files created successfully")

# create_docker_files()
```

### 3. Pipeline Monitoring and Alerting
```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List

class PipelineMonitor:
    """Monitor pipeline execution and send alerts"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alerts = []
    
    def check_pipeline_health(self, pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """Check pipeline health and generate alerts"""
        health_status = {
            'status': 'healthy',
            'issues': [],
            'metrics': {}
        }
        
        # Check execution status
        if pipeline_result['status'] != 'success':
            health_status['status'] = 'unhealthy'
            health_status['issues'].append({
                'type': 'execution_failure',
                'message': f"Pipeline failed: {pipeline_result.get('error', 'Unknown error')}"
            })
        
        # Check execution duration
        duration = pipeline_result.get('duration', 0)
        max_duration = self.config.get('max_pipeline_duration', 3600)  # 1 hour
        
        if duration > max_duration:
            health_status['issues'].append({
                'type': 'long_execution',
                'message': f"Pipeline took {duration:.2f}s (threshold: {max_duration}s)"
            })
        
        # Check model performance
        if pipeline_result['status'] == 'success':
            result = pipeline_result.get('result', {})
            registry_entry = result.get('registry_entry', {})
            metrics = registry_entry.get('metrics', {})
            
            accuracy = metrics.get('accuracy', 0)
            min_accuracy = self.config.get('min_accuracy_threshold', 0.7)
            
            if accuracy < min_accuracy:
                health_status['status'] = 'unhealthy'
                health_status['issues'].append({
                    'type': 'low_performance',
                    'message': f"Model accuracy {accuracy:.4f} below threshold {min_accuracy}"
                })
            
            health_status['metrics'] = metrics
        
        return health_status
    
    def send_alert(self, health_status: Dict[str, Any]):
        """Send alert notification"""
        if health_status['status'] == 'unhealthy':
            # Email alert
            self._send_email_alert(health_status)
            
            # Slack alert (if configured)
            if 'slack_webhook' in self.config:
                self._send_slack_alert(health_status)
    
    def _send_email_alert(self, health_status: Dict[str, Any]):
        """Send email alert"""
        if 'email' not in self.config:
            return
        
        email_config = self.config['email']
        
        msg = MIMEMultipart()
        msg['From'] = email_config['from']
        msg['To'] = ', '.join(email_config['to'])
        msg['Subject'] = f"ML Pipeline Alert - {health_status['status'].upper()}"
        
        # Create email body
        body = f"""
        ML Pipeline Health Alert
        
        Status: {health_status['status'].upper()}
        
        Issues Found:
        """
        
        for issue in health_status['issues']:
            body += f"- {issue['type']}: {issue['message']}\n"
        
        if health_status.get('metrics'):
            body += f"\nModel Metrics:\n"
            for metric, value in health_status['metrics'].items():
                if isinstance(value, (int, float)):
                    body += f"- {metric}: {value:.4f}\n"
        
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            server.quit()
            
            print("Alert email sent successfully")
            
        except Exception as e:
            print(f"Failed to send email alert: {str(e)}")
    
    def _send_slack_alert(self, health_status: Dict[str, Any]):
        """Send Slack alert"""
        import requests
        
        webhook_url = self.config['slack_webhook']
        
        color = "danger" if health_status['status'] == 'unhealthy' else "good"
        
        message = {
            "attachments": [
                {
                    "color": color,
                    "title": f"ML Pipeline Alert - {health_status['status'].upper()}",
                    "fields": [
                        {
                            "title": "Issues",
                            "value": "\n".join([f"• {issue['message']}" for issue in health_status['issues']]),
                            "short": False
                        }
                    ]
                }
            ]
        }
        
        try:
            response = requests.post(webhook_url, json=message)
            response.raise_for_status()
            print("Slack alert sent successfully")
            
        except Exception as e:
            print(f"Failed to send Slack alert: {str(e)}")

# Monitoring example
def setup_pipeline_monitoring():
    """Setup pipeline monitoring configuration"""
    monitor_config = {
        'max_pipeline_duration': 1800,  # 30 minutes
        'min_accuracy_threshold': 0.75,
        'email': {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'from': 'ml-pipeline@company.com',
            'to': ['team@company.com'],
            'username': 'your_email@gmail.com',
            'password': 'your_app_password'
        },
        'slack_webhook': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    }
    
    return PipelineMonitor(monitor_config)
```

## Automated Retraining

### 1. Scheduled Retraining
```python
import schedule
import time
from datetime import datetime, timedelta

class AutomatedRetraining:
    """Automated model retraining system"""
    
    def __init__(self, config: PipelineConfig, monitor: PipelineMonitor):
        self.config = config
        self.monitor = monitor
        self.last_training = None
        self.retraining_log = []
    
    def setup_schedule(self):
        """Setup retraining schedule"""
        # Daily retraining at 2 AM
        schedule.every().day.at("02:00").do(self.scheduled_retrain, trigger='daily')
        
        # Weekly full retraining on Sunday
        schedule.every().sunday.at("03:00").do(self.full_retrain, trigger='weekly')
        
        # Performance-based retraining check every 6 hours
        schedule.every(6).hours.do(self.check_performance_trigger, trigger='performance')
    
    def scheduled_retrain(self, trigger: str):
        """Execute scheduled retraining"""
        print(f"Starting scheduled retraining (trigger: {trigger})")
        
        try:
            # Run training pipeline
            pipeline = TrainingPipeline(self.config)
            result = pipeline.run()
            
            # Monitor results
            health_status = self.monitor.check_pipeline_health(result)
            
            # Log retraining
            self.retraining_log.append({
                'timestamp': datetime.now().isoformat(),
                'trigger': trigger,
                'status': result['status'],
                'health': health_status['status'],
                'model_id': result.get('result', {}).get('model_id'),
                'metrics': health_status.get('metrics', {})
            })
            
            # Send alerts if necessary
            if health_status['status'] == 'unhealthy':
                self.monitor.send_alert(health_status)
            
            self.last_training = datetime.now()
            print(f"Retraining completed: {result['status']}")
            
        except Exception as e:
            print(f"Retraining failed: {str(e)}")
            
            # Log failure
            self.retraining_log.append({
                'timestamp': datetime.now().isoformat(),
                'trigger': trigger,
                'status': 'failed',
                'error': str(e)
            })
    
    def full_retrain(self, trigger: str):
        """Execute full retraining with extended validation"""
        print(f"Starting full retraining (trigger: {trigger})")
        
        # Update configuration for full retraining
        full_config = self.config
        full_config.model_params.update({
            'n_estimators': 200,  # More trees for weekly training
            'max_depth': 15
        })
        
        self.scheduled_retrain(trigger)
    
    def check_performance_trigger(self, trigger: str):
        """Check if performance-based retraining is needed"""
        # This would typically check recent model performance metrics
        # from a monitoring system or database
        
        # Placeholder logic - in practice, you'd query your monitoring system
        current_performance = self._get_current_model_performance()
        
        if current_performance < self.config.validation_threshold:
            print(f"Performance drop detected: {current_performance:.4f}")
            self.scheduled_retrain(f"{trigger}_performance_drop")
    
    def _get_current_model_performance(self) -> float:
        """Get current model performance from monitoring system"""
        # Placeholder - implement based on your monitoring setup
        # This might query MLflow, database, or monitoring service
        return np.random.uniform(0.7, 0.9)  # Simulated performance
    
    def run_scheduler(self):
        """Run the retraining scheduler"""
        print("Starting automated retraining scheduler...")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def get_retraining_history(self) -> List[Dict[str, Any]]:
        """Get retraining history"""
        return self.retraining_log

# Example usage
def setup_automated_retraining():
    """Setup automated retraining system"""
    config = create_sample_config()
    monitor = setup_pipeline_monitoring()
    
    retrainer = AutomatedRetraining(config, monitor)
    retrainer.setup_schedule()
    
    return retrainer

# Auto-retrainer setup
# retrainer = setup_automated_retraining()
# retrainer.run_scheduler()  # This would run indefinitely
```

### 2. Data Drift Detection
```python
from scipy import stats
from sklearn.metrics import wasserstein_distance

class DataDriftDetector:
    """Detect data drift to trigger retraining"""
    
    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.1):
        self.reference_data = reference_data
        self.threshold = threshold
        self.drift_history = []
    
    def detect_drift(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect data drift between reference and new data"""
        drift_results = {
            'overall_drift': False,
            'feature_drifts': {},
            'drift_score': 0.0,
            'timestamp': datetime.now().isoformat()
        }
        
        numeric_features = self.reference_data.select_dtypes(include=[np.number]).columns
        
        feature_drift_scores = []
        
        for feature in numeric_features:
            if feature in new_data.columns:
                # Statistical tests for drift detection
                drift_score = self._calculate_drift_score(
                    self.reference_data[feature],
                    new_data[feature]
                )
                
                feature_drifted = drift_score > self.threshold
                
                drift_results['feature_drifts'][feature] = {
                    'drift_score': drift_score,
                    'drifted': feature_drifted
                }
                
                feature_drift_scores.append(drift_score)
        
        # Overall drift assessment
        if feature_drift_scores:
            drift_results['drift_score'] = np.mean(feature_drift_scores)
            drift_results['overall_drift'] = drift_results['drift_score'] > self.threshold
        
        # Log drift detection
        self.drift_history.append(drift_results)
        
        return drift_results
    
    def _calculate_drift_score(self, reference: pd.Series, new: pd.Series) -> float:
        """Calculate drift score between two distributions"""
        # Remove missing values
        ref_clean = reference.dropna()
        new_clean = new.dropna()
        
        if len(ref_clean) == 0 or len(new_clean) == 0:
            return 0.0
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p_value = stats.ks_2samp(ref_clean, new_clean)
        
        # Wasserstein distance (normalized)
        wasserstein_dist = wasserstein_distance(ref_clean, new_clean)
        
        # Combine metrics (you can adjust weights)
        drift_score = 0.7 * ks_stat + 0.3 * min(wasserstein_dist / np.std(ref_clean), 1.0)
        
        return drift_score
    
    def should_retrain(self, new_data: pd.DataFrame) -> bool:
        """Determine if retraining is needed based on drift"""
        drift_results = self.detect_drift(new_data)
        return drift_results['overall_drift']

# Integration with automated retraining
class DriftTriggeredRetraining(AutomatedRetraining):
    """Automated retraining with drift detection"""
    
    def __init__(self, config: PipelineConfig, monitor: PipelineMonitor, reference_data: pd.DataFrame):
        super().__init__(config, monitor)
        self.drift_detector = DataDriftDetector(reference_data)
    
    def check_drift_and_retrain(self, new_data: pd.DataFrame):
        """Check for data drift and trigger retraining if needed"""
        if self.drift_detector.should_retrain(new_data):
            print("Data drift detected, triggering retraining...")
            self.scheduled_retrain("data_drift")
        else:
            print("No significant data drift detected")
```

## Summary

Automated training pipelines provide:

1. **Reproducible Workflows**: Consistent training processes across environments
2. **Continuous Integration**: Automated testing and validation of ML models
3. **Scalable Architecture**: Handle large datasets and complex workflows
4. **Monitoring & Alerting**: Track pipeline health and model performance
5. **Automated Retraining**: Schedule-based and performance-triggered retraining

### Key Takeaways
- Design modular pipeline components for reusability
- Implement comprehensive validation at each step
- Use orchestration tools for complex workflows
- Monitor pipeline execution and model performance
- Integrate with CI/CD systems for automation
- Implement data drift detection for intelligent retraining

### Next Steps
- Explore advanced orchestration tools (Airflow, Kubeflow, MLflow)
- Implement A/B testing for model deployments
- Add more sophisticated monitoring and alerting
- Integrate with cloud-based ML platforms
- Implement model governance and compliance features