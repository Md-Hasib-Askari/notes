# CI/CD for Machine Learning

## Learning Objectives
- Understand CI/CD concepts adapted for machine learning workflows
- Implement automated testing strategies for ML models and data
- Build deployment pipelines for ML models across environments
- Set up model versioning and artifact management
- Integrate ML workflows with traditional software development practices
- Implement monitoring and rollback strategies for ML deployments

## Introduction

CI/CD (Continuous Integration/Continuous Deployment) for Machine Learning extends traditional software development practices to handle the unique challenges of ML systems, including data versioning, model validation, and gradual deployment strategies.

### Key Differences from Traditional CI/CD
- **Data Dependencies**: Models depend on training data that changes over time
- **Model Validation**: Beyond code tests, models need performance validation
- **Gradual Deployment**: A/B testing and canary deployments are crucial
- **Reproducibility**: Ensuring consistent results across environments
- **Model Drift**: Monitoring and handling performance degradation

## Core CI/CD Components for ML

### 1. Repository Structure for ML Projects
```
ml-project/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── cd-staging.yml
│       └── cd-production.yml
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestion.py
│   │   ├── validation.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── training.py
│   │   ├── evaluation.py
│   │   └── prediction.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── logging.py
├── tests/
│   ├── unit/
│   │   ├── test_data.py
│   │   ├── test_models.py
│   │   └── test_features.py
│   ├── integration/
│   │   ├── test_pipeline.py
│   │   └── test_api.py
│   └── model/
│       ├── test_model_performance.py
│       └── test_model_validation.py
├── configs/
│   ├── config.yaml
│   ├── model_config.yaml
│   └── deployment_config.yaml
├── scripts/
│   ├── train_model.py
│   ├── deploy_model.py
│   └── run_tests.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── setup.py
└── README.md
```

### 2. Configuration Management
```python
# src/utils/config.py
import yaml
import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path

@dataclass
class DataConfig:
    """Data configuration"""
    source_path: str
    validation_rules: Dict[str, Any]
    preprocessing_steps: List[str]
    feature_columns: List[str]
    target_column: str
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

@dataclass
class ModelConfig:
    """Model configuration"""
    model_type: str
    hyperparameters: Dict[str, Any]
    training_config: Dict[str, Any]
    evaluation_metrics: List[str]
    performance_thresholds: Dict[str, float]

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    environment: str
    deployment_strategy: str  # blue-green, canary, rolling
    resource_requirements: Dict[str, Any]
    scaling_config: Dict[str, Any]
    monitoring_config: Dict[str, Any]

@dataclass
class MLConfig:
    """Main ML configuration"""
    project_name: str
    version: str
    environment: str
    data: DataConfig
    model: ModelConfig
    deployment: DeploymentConfig
    artifact_store: str
    model_registry: str
    
    @classmethod
    def from_yaml(cls, config_path: str, environment: str = "development"):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Get environment-specific config
        env_config = config_dict.get(environment, config_dict.get("default", {}))
        
        return cls(
            project_name=env_config["project_name"],
            version=env_config["version"],
            environment=environment,
            data=DataConfig(**env_config["data"]),
            model=ModelConfig(**env_config["model"]),
            deployment=DeploymentConfig(**env_config["deployment"]),
            artifact_store=env_config["artifact_store"],
            model_registry=env_config["model_registry"]
        )
    
    def save(self, output_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            self.environment: {
                "project_name": self.project_name,
                "version": self.version,
                "data": self.data.__dict__,
                "model": self.model.__dict__,
                "deployment": self.deployment.__dict__,
                "artifact_store": self.artifact_store,
                "model_registry": self.model_registry
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

# Example configuration YAML
CONFIG_YAML = """
development:
  project_name: "ml-classifier"
  version: "1.0.0"
  data:
    source_path: "data/train.csv"
    validation_rules:
      min_rows: 1000
      max_missing_ratio: 0.1
      required_columns: ["feature1", "feature2", "target"]
    preprocessing_steps: ["normalize", "handle_missing"]
    feature_columns: ["feature1", "feature2", "feature3"]
    target_column: "target"
  model:
    model_type: "random_forest"
    hyperparameters:
      n_estimators: 100
      max_depth: 10
      random_state: 42
    training_config:
      cross_validation: 5
      early_stopping: true
    evaluation_metrics: ["accuracy", "precision", "recall", "f1"]
    performance_thresholds:
      accuracy: 0.85
      f1: 0.80
  deployment:
    environment: "development"
    deployment_strategy: "blue-green"
    resource_requirements:
      cpu: "500m"
      memory: "1Gi"
    scaling_config:
      min_replicas: 1
      max_replicas: 3
    monitoring_config:
      enable_metrics: true
      log_predictions: true
  artifact_store: "s3://ml-artifacts"
  model_registry: "mlflow"

staging:
  project_name: "ml-classifier"
  version: "1.0.0"
  data:
    source_path: "s3://data-bucket/staging/train.csv"
    validation_rules:
      min_rows: 5000
      max_missing_ratio: 0.05
      required_columns: ["feature1", "feature2", "target"]
    preprocessing_steps: ["normalize", "handle_missing", "feature_selection"]
    feature_columns: ["feature1", "feature2", "feature3", "feature4"]
    target_column: "target"
  model:
    model_type: "random_forest"
    hyperparameters:
      n_estimators: 200
      max_depth: 15
      random_state: 42
    training_config:
      cross_validation: 10
      early_stopping: true
    evaluation_metrics: ["accuracy", "precision", "recall", "f1", "auc"]
    performance_thresholds:
      accuracy: 0.88
      f1: 0.85
      auc: 0.90
  deployment:
    environment: "staging"
    deployment_strategy: "canary"
    resource_requirements:
      cpu: "1000m"
      memory: "2Gi"
    scaling_config:
      min_replicas: 2
      max_replicas: 5
    monitoring_config:
      enable_metrics: true
      log_predictions: true
      alert_thresholds:
        accuracy_drop: 0.05
        latency_p95: 500
  artifact_store: "s3://ml-artifacts-staging"
  model_registry: "mlflow-staging"

production:
  project_name: "ml-classifier"
  version: "1.0.0"
  data:
    source_path: "s3://data-bucket/production/train.csv"
    validation_rules:
      min_rows: 10000
      max_missing_ratio: 0.02
      required_columns: ["feature1", "feature2", "target"]
    preprocessing_steps: ["normalize", "handle_missing", "feature_selection", "outlier_removal"]
    feature_columns: ["feature1", "feature2", "feature3", "feature4", "feature5"]
    target_column: "target"
  model:
    model_type: "random_forest"
    hyperparameters:
      n_estimators: 500
      max_depth: 20
      random_state: 42
    training_config:
      cross_validation: 10
      early_stopping: true
      hyperparameter_tuning: true
    evaluation_metrics: ["accuracy", "precision", "recall", "f1", "auc"]
    performance_thresholds:
      accuracy: 0.90
      f1: 0.88
      auc: 0.92
  deployment:
    environment: "production"
    deployment_strategy: "blue-green"
    resource_requirements:
      cpu: "2000m"
      memory: "4Gi"
    scaling_config:
      min_replicas: 3
      max_replicas: 10
    monitoring_config:
      enable_metrics: true
      log_predictions: true
      alert_thresholds:
        accuracy_drop: 0.03
        latency_p95: 200
        error_rate: 0.01
  artifact_store: "s3://ml-artifacts-production"
  model_registry: "mlflow-production"
"""

def save_config_yaml():
    """Save example configuration to file"""
    with open('configs/config.yaml', 'w') as f:
        f.write(CONFIG_YAML)
```

### 3. Automated Testing for ML
```python
# tests/unit/test_data.py
import pytest
import pandas as pd
import numpy as np
from src.data.validation import DataValidator
from src.data.preprocessing import DataPreprocessor

class TestDataValidation:
    """Test data validation functionality"""
    
    def setup_method(self):
        """Setup test data"""
        self.valid_data = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'feature3': np.random.randn(1000),
            'target': np.random.randint(0, 2, 1000)
        })
        
        self.invalid_data = pd.DataFrame({
            'feature1': [np.nan] * 500 + list(np.random.randn(500)),
            'feature2': np.random.randn(1000),
            'target': np.random.randint(0, 2, 1000)
        })
    
    def test_data_schema_validation(self):
        """Test data schema validation"""
        validator = DataValidator()
        
        # Valid data should pass
        assert validator.validate_schema(self.valid_data, required_columns=['feature1', 'feature2', 'target'])
        
        # Invalid data should fail
        assert not validator.validate_schema(self.invalid_data, required_columns=['feature1', 'feature2', 'feature3', 'target'])
    
    def test_data_quality_checks(self):
        """Test data quality validation"""
        validator = DataValidator()
        
        # Check missing value threshold
        quality_report = validator.check_data_quality(self.invalid_data, max_missing_ratio=0.1)
        assert not quality_report['passed']
        assert 'high_missing_values' in quality_report['issues']
    
    def test_data_drift_detection(self):
        """Test data drift detection"""
        # Create reference and new data
        reference_data = self.valid_data
        new_data = self.valid_data.copy()
        new_data['feature1'] = new_data['feature1'] + 2  # Introduce drift
        
        validator = DataValidator()
        drift_report = validator.detect_drift(reference_data, new_data)
        
        assert drift_report['drift_detected']
        assert 'feature1' in drift_report['drifted_features']

# tests/unit/test_models.py
class TestModelTraining:
    """Test model training functionality"""
    
    def setup_method(self):
        """Setup test data and model"""
        from src.models.training import ModelTrainer
        
        # Create synthetic data
        self.X = np.random.randn(1000, 5)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(int)
        
        self.trainer = ModelTrainer(model_type='random_forest')
    
    def test_model_training(self):
        """Test model training process"""
        model, metrics = self.trainer.train(self.X, self.y)
        
        # Check model is trained
        assert hasattr(model, 'predict')
        
        # Check metrics are calculated
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert metrics['accuracy'] > 0.5  # Should be better than random
    
    def test_model_validation(self):
        """Test model validation"""
        model, metrics = self.trainer.train(self.X, self.y)
        
        # Test prediction shape
        predictions = model.predict(self.X[:10])
        assert len(predictions) == 10
        
        # Test prediction range for classification
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_model_serialization(self):
        """Test model saving and loading"""
        import tempfile
        import joblib
        
        model, _ = self.trainer.train(self.X, self.y)
        
        with tempfile.NamedTemporaryFile(suffix='.joblib') as f:
            # Save model
            joblib.dump(model, f.name)
            
            # Load model
            loaded_model = joblib.load(f.name)
            
            # Test predictions are identical
            original_pred = model.predict(self.X[:10])
            loaded_pred = loaded_model.predict(self.X[:10])
            
            np.testing.assert_array_equal(original_pred, loaded_pred)

# tests/model/test_model_performance.py
class TestModelPerformance:
    """Test model performance requirements"""
    
    def setup_method(self):
        """Setup test model and data"""
        from src.models.training import ModelTrainer
        from src.utils.config import MLConfig
        
        # Load configuration
        self.config = MLConfig.from_yaml('configs/config.yaml', 'development')
        
        # Create test data
        self.X_test = np.random.randn(200, 5)
        self.y_test = (self.X_test[:, 0] + self.X_test[:, 1] > 0).astype(int)
        
        # Train model
        trainer = ModelTrainer(model_type=self.config.model.model_type)
        self.model, self.training_metrics = trainer.train(self.X_test, self.y_test)
    
    def test_model_accuracy_threshold(self):
        """Test model meets accuracy threshold"""
        accuracy_threshold = self.config.model.performance_thresholds['accuracy']
        
        predictions = self.model.predict(self.X_test)
        accuracy = (predictions == self.y_test).mean()
        
        assert accuracy >= accuracy_threshold, f"Model accuracy {accuracy:.4f} below threshold {accuracy_threshold}"
    
    def test_model_f1_threshold(self):
        """Test model meets F1 score threshold"""
        from sklearn.metrics import f1_score
        
        f1_threshold = self.config.model.performance_thresholds['f1']
        
        predictions = self.model.predict(self.X_test)
        f1 = f1_score(self.y_test, predictions, average='weighted')
        
        assert f1 >= f1_threshold, f"Model F1 score {f1:.4f} below threshold {f1_threshold}"
    
    def test_prediction_latency(self):
        """Test model prediction latency"""
        import time
        
        # Single prediction latency
        start_time = time.time()
        _ = self.model.predict(self.X_test[:1])
        single_latency = time.time() - start_time
        
        assert single_latency < 0.1, f"Single prediction latency {single_latency:.4f}s too high"
        
        # Batch prediction latency
        start_time = time.time()
        _ = self.model.predict(self.X_test)
        batch_latency = time.time() - start_time
        
        avg_latency = batch_latency / len(self.X_test)
        assert avg_latency < 0.01, f"Average prediction latency {avg_latency:.4f}s too high"
    
    def test_model_memory_usage(self):
        """Test model memory footprint"""
        import sys
        import pickle
        
        # Serialize model to measure size
        model_bytes = pickle.dumps(self.model)
        model_size_mb = len(model_bytes) / (1024 * 1024)
        
        # Model should be under 100MB for this example
        assert model_size_mb < 100, f"Model size {model_size_mb:.2f}MB too large"

# tests/integration/test_pipeline.py
class TestMLPipeline:
    """Integration tests for ML pipeline"""
    
    def test_end_to_end_pipeline(self):
        """Test complete ML pipeline"""
        from src.data.ingestion import DataIngester
        from src.data.preprocessing import DataPreprocessor
        from src.models.training import ModelTrainer
        from src.models.evaluation import ModelEvaluator
        
        # Create test data file
        test_data = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'feature3': np.random.randn(1000),
            'target': np.random.randint(0, 2, 1000)
        })
        test_data.to_csv('test_data.csv', index=False)
        
        # Run pipeline
        try:
            # Data ingestion
            ingester = DataIngester()
            raw_data = ingester.load_data('test_data.csv')
            
            # Data preprocessing
            preprocessor = DataPreprocessor()
            processed_data = preprocessor.preprocess(raw_data)
            
            # Model training
            trainer = ModelTrainer(model_type='random_forest')
            X = processed_data.drop(columns=['target'])
            y = processed_data['target']
            model, metrics = trainer.train(X, y)
            
            # Model evaluation
            evaluator = ModelEvaluator()
            evaluation_report = evaluator.evaluate(model, X, y)
            
            # Assertions
            assert model is not None
            assert 'accuracy' in metrics
            assert evaluation_report['status'] == 'passed'
            
        finally:
            # Cleanup
            import os
            if os.path.exists('test_data.csv'):
                os.remove('test_data.csv')
```

### 4. GitHub Actions Workflows
```yaml
# .github/workflows/ci.yml
name: CI - Continuous Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  PYTHON_VERSION: '3.9'
  POETRY_VERSION: '1.4.2'

jobs:
  code-quality:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install flake8 black isort mypy pytest
        pip install -r requirements.txt
    
    - name: Code formatting check
      run: |
        black --check src/ tests/
        isort --check-only src/ tests/
    
    - name: Linting
      run: |
        flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503
    
    - name: Type checking
      run: |
        mypy src/ --ignore-missing-imports

  unit-tests:
    runs-on: ubuntu-latest
    needs: code-quality
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  data-validation:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Download test data
      run: |
        # In real scenarios, download from data source
        python scripts/generate_test_data.py
    
    - name: Validate data schema
      run: |
        python scripts/validate_data_schema.py
    
    - name: Check data quality
      run: |
        python scripts/check_data_quality.py

  model-training:
    runs-on: ubuntu-latest
    needs: data-validation
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Train model
      run: |
        python scripts/train_model.py --config configs/config.yaml --env development
    
    - name: Run model tests
      run: |
        pytest tests/model/ -v
    
    - name: Upload model artifacts
      uses: actions/upload-artifact@v3
      with:
        name: model-artifacts
        path: artifacts/

  integration-tests:
    runs-on: ubuntu-latest
    needs: model-training
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Download model artifacts
      uses: actions/download-artifact@v3
      with:
        name: model-artifacts
        path: artifacts/
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
    
    - name: API tests
      run: |
        # Start model API
        python scripts/start_api.py &
        sleep 10
        
        # Run API tests
        pytest tests/api/ -v
        
        # Stop API
        pkill -f "start_api.py"

# .github/workflows/cd-staging.yml
name: CD - Deploy to Staging

on:
  push:
    branches: [ develop ]
  workflow_run:
    workflows: ["CI - Continuous Integration"]
    types:
      - completed
    branches: [ develop ]

jobs:
  deploy-staging:
    runs-on: ubuntu-latest
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    environment: staging
    
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
    
    - name: Build Docker image
      run: |
        docker build -t ml-model:staging .
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-west-2
    
    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1
    
    - name: Push image to ECR
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        ECR_REPOSITORY: ml-model
        IMAGE_TAG: staging-${{ github.sha }}
      run: |
        docker tag ml-model:staging $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
    
    - name: Deploy to EKS
      run: |
        aws eks update-kubeconfig --region us-west-2 --name staging-cluster
        
        # Update deployment
        kubectl set image deployment/ml-model-staging \
          ml-model=$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
        
        # Wait for rollout
        kubectl rollout status deployment/ml-model-staging
    
    - name: Run smoke tests
      run: |
        python scripts/smoke_tests.py --environment staging
    
    - name: Notify Slack
      if: always()
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        text: "Staging deployment ${{ job.status }}"
      env:
        SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}

# .github/workflows/cd-production.yml
name: CD - Deploy to Production

on:
  push:
    branches: [ main ]
  release:
    types: [ published ]

jobs:
  deploy-production:
    runs-on: ubuntu-latest
    environment: production
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Manual approval
      uses: trstringer/manual-approval@v1
      with:
        secret: ${{ github.TOKEN }}
        approvers: team-leads,ml-engineers
        minimum-approvals: 2
        issue-title: "Deploy to Production"
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Train production model
      run: |
        python scripts/train_model.py --config configs/config.yaml --env production
    
    - name: Model validation
      run: |
        python scripts/validate_production_model.py
    
    - name: Build production image
      run: |
        docker build -t ml-model:production .
    
    - name: Blue-green deployment
      run: |
        python scripts/blue_green_deploy.py
    
    - name: Canary deployment
      run: |
        python scripts/canary_deploy.py --traffic-percentage 10
        
        # Monitor for 10 minutes
        sleep 600
        
        # If metrics are good, increase traffic
        python scripts/canary_deploy.py --traffic-percentage 50
        
        # Monitor for 10 minutes
        sleep 600
        
        # Complete deployment
        python scripts/canary_deploy.py --traffic-percentage 100
    
    - name: Post-deployment tests
      run: |
        python scripts/production_tests.py
    
    - name: Update model registry
      run: |
        python scripts/update_model_registry.py \
          --model-version ${{ github.sha }} \
          --status production \
          --environment production
```

### 5. Model Versioning and Registry
```python
# src/models/registry.py
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import mlflow
import boto3

@dataclass
class ModelMetadata:
    """Model metadata for registry"""
    model_id: str
    name: str
    version: str
    framework: str
    algorithm: str
    training_data_hash: str
    feature_schema: Dict[str, str]
    performance_metrics: Dict[str, float]
    training_config: Dict[str, Any]
    created_at: str
    created_by: str
    status: str  # development, staging, production, archived
    deployment_config: Optional[Dict[str, Any]] = None
    tags: Optional[Dict[str, str]] = None

class ModelRegistry:
    """Model registry for versioning and lifecycle management"""
    
    def __init__(self, backend: str = "mlflow", **kwargs):
        self.backend = backend
        
        if backend == "mlflow":
            self.client = mlflow.tracking.MlflowClient(
                tracking_uri=kwargs.get("tracking_uri", "http://localhost:5000")
            )
        elif backend == "s3":
            self.s3_client = boto3.client('s3')
            self.bucket = kwargs.get("bucket", "ml-model-registry")
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    def register_model(self, model_path: str, metadata: ModelMetadata) -> str:
        """Register a new model version"""
        
        if self.backend == "mlflow":
            return self._register_mlflow_model(model_path, metadata)
        elif self.backend == "s3":
            return self._register_s3_model(model_path, metadata)
    
    def _register_mlflow_model(self, model_path: str, metadata: ModelMetadata) -> str:
        """Register model in MLflow"""
        
        with mlflow.start_run():
            # Log model
            mlflow.sklearn.log_model(
                sk_model=model_path,
                artifact_path="model",
                registered_model_name=metadata.name
            )
            
            # Log metadata
            mlflow.log_params(metadata.training_config)
            mlflow.log_metrics(metadata.performance_metrics)
            
            # Log additional metadata
            mlflow.set_tag("model_id", metadata.model_id)
            mlflow.set_tag("algorithm", metadata.algorithm)
            mlflow.set_tag("status", metadata.status)
            mlflow.set_tag("training_data_hash", metadata.training_data_hash)
            
            if metadata.tags:
                for key, value in metadata.tags.items():
                    mlflow.set_tag(key, value)
            
            run_id = mlflow.active_run().info.run_id
        
        # Register model version
        model_version = self.client.create_model_version(
            name=metadata.name,
            source=f"runs:/{run_id}/model",
            description=f"Model version {metadata.version}"
        )
        
        return model_version.version
    
    def _register_s3_model(self, model_path: str, metadata: ModelMetadata) -> str:
        """Register model in S3"""
        import joblib
        import tempfile
        
        # Upload model file
        model_key = f"{metadata.name}/{metadata.version}/model.joblib"
        self.s3_client.upload_file(model_path, self.bucket, model_key)
        
        # Upload metadata
        metadata_key = f"{metadata.name}/{metadata.version}/metadata.json"
        metadata_json = json.dumps(asdict(metadata), indent=2)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
            f.write(metadata_json)
            f.flush()
            self.s3_client.upload_file(f.name, self.bucket, metadata_key)
        
        return metadata.version
    
    def get_model(self, name: str, version: str = "latest", stage: str = None) -> Dict[str, Any]:
        """Retrieve model from registry"""
        
        if self.backend == "mlflow":
            return self._get_mlflow_model(name, version, stage)
        elif self.backend == "s3":
            return self._get_s3_model(name, version)
    
    def _get_mlflow_model(self, name: str, version: str, stage: str) -> Dict[str, Any]:
        """Get model from MLflow"""
        
        if stage:
            model_version = self.client.get_latest_versions(
                name, stages=[stage]
            )[0]
        elif version == "latest":
            model_version = self.client.get_latest_versions(name)[0]
        else:
            model_version = self.client.get_model_version(name, version)
        
        # Download model
        model_uri = f"models:/{name}/{model_version.version}"
        model = mlflow.sklearn.load_model(model_uri)
        
        # Get run metadata
        run = self.client.get_run(model_version.run_id)
        
        return {
            "model": model,
            "version": model_version.version,
            "metadata": {
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags
            }
        }
    
    def _get_s3_model(self, name: str, version: str) -> Dict[str, Any]:
        """Get model from S3"""
        import joblib
        import tempfile
        
        # Download model
        model_key = f"{name}/{version}/model.joblib"
        
        with tempfile.NamedTemporaryFile(suffix='.joblib') as f:
            self.s3_client.download_file(self.bucket, model_key, f.name)
            model = joblib.load(f.name)
        
        # Download metadata
        metadata_key = f"{name}/{version}/metadata.json"
        
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as f:
            self.s3_client.download_file(self.bucket, metadata_key, f.name)
            f.seek(0)
            metadata = json.load(f)
        
        return {
            "model": model,
            "version": version,
            "metadata": metadata
        }
    
    def promote_model(self, name: str, version: str, stage: str) -> bool:
        """Promote model to a new stage"""
        
        if self.backend == "mlflow":
            self.client.transition_model_version_stage(
                name=name,
                version=version,
                stage=stage
            )
            return True
        elif self.backend == "s3":
            # Update metadata with new stage
            metadata_key = f"{name}/{version}/metadata.json"
            
            # Download current metadata
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as f:
                self.s3_client.download_file(self.bucket, metadata_key, f.name)
                f.seek(0)
                metadata = json.load(f)
                
                # Update stage
                metadata['status'] = stage
                
                # Upload updated metadata
                f.seek(0)
                f.truncate()
                json.dump(metadata, f, indent=2)
                f.flush()
                
                self.s3_client.upload_file(f.name, self.bucket, metadata_key)
            
            return True
    
    def list_models(self, name: str = None) -> List[Dict[str, Any]]:
        """List models in registry"""
        
        if self.backend == "mlflow":
            if name:
                versions = self.client.search_model_versions(f"name='{name}'")
            else:
                versions = self.client.search_model_versions()
            
            return [
                {
                    "name": v.name,
                    "version": v.version,
                    "stage": v.current_stage,
                    "created_at": v.creation_timestamp
                }
                for v in versions
            ]
        
        elif self.backend == "s3":
            # List objects in S3 bucket
            if name:
                prefix = f"{name}/"
            else:
                prefix = ""
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix,
                Delimiter="/"
            )
            
            models = []
            for obj in response.get('CommonPrefixes', []):
                model_name = obj['Prefix'].rstrip('/').split('/')[-1]
                
                # Get versions for this model
                version_response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket,
                    Prefix=obj['Prefix'],
                    Delimiter="/"
                )
                
                for version_obj in version_response.get('CommonPrefixes', []):
                    version = version_obj['Prefix'].rstrip('/').split('/')[-1]
                    models.append({
                        "name": model_name,
                        "version": version
                    })
            
            return models

# Model registry usage example
def model_registry_example():
    """Example of using model registry"""
    
    # Initialize registry
    registry = ModelRegistry(backend="mlflow", tracking_uri="http://localhost:5000")
    
    # Create model metadata
    metadata = ModelMetadata(
        model_id="classifier_v1",
        name="customer_churn_classifier",
        version="1.0.0",
        framework="scikit-learn",
        algorithm="random_forest",
        training_data_hash="abc123",
        feature_schema={
            "age": "float",
            "income": "float",
            "tenure": "int"
        },
        performance_metrics={
            "accuracy": 0.89,
            "f1_score": 0.85,
            "auc": 0.92
        },
        training_config={
            "n_estimators": 100,
            "max_depth": 10
        },
        created_at=datetime.now().isoformat(),
        created_by="ml-engineer",
        status="development",
        tags={"experiment": "baseline"}
    )
    
    # Register model
    version = registry.register_model("path/to/model.joblib", metadata)
    print(f"Registered model version: {version}")
    
    # Get model
    model_info = registry.get_model("customer_churn_classifier", "latest")
    print(f"Retrieved model version: {model_info['version']}")
    
    # Promote model
    registry.promote_model("customer_churn_classifier", version, "staging")
    print("Model promoted to staging")
    
    # List models
    models = registry.list_models("customer_churn_classifier")
    for model in models:
        print(f"Model: {model['name']}, Version: {model['version']}")
```

### 6. Deployment Strategies
```python
# scripts/deployment_strategies.py
import time
import requests
import numpy as np
from typing import Dict, Any, List
import kubernetes
from kubernetes import client, config

class BlueGreenDeployment:
    """Blue-green deployment strategy"""
    
    def __init__(self, k8s_config_path: str = None):
        if k8s_config_path:
            config.load_kube_config(k8s_config_path)
        else:
            config.load_incluster_config()
        
        self.v1 = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()
    
    def deploy(self, deployment_config: Dict[str, Any]) -> bool:
        """Execute blue-green deployment"""
        
        service_name = deployment_config['service_name']
        new_image = deployment_config['new_image']
        namespace = deployment_config.get('namespace', 'default')
        
        # Create green deployment
        green_deployment_name = f"{service_name}-green"
        
        print("Creating green deployment...")
        self._create_green_deployment(
            green_deployment_name, 
            new_image, 
            namespace,
            deployment_config
        )
        
        # Wait for green deployment to be ready
        print("Waiting for green deployment to be ready...")
        if not self._wait_for_deployment_ready(green_deployment_name, namespace):
            print("Green deployment failed to become ready")
            return False
        
        # Run health checks
        print("Running health checks on green deployment...")
        if not self._run_health_checks(green_deployment_name, namespace):
            print("Health checks failed")
            self._cleanup_green_deployment(green_deployment_name, namespace)
            return False
        
        # Switch traffic to green
        print("Switching traffic to green deployment...")
        self._switch_traffic_to_green(service_name, green_deployment_name, namespace)
        
        # Wait and monitor
        time.sleep(30)
        
        # Run final validation
        print("Running final validation...")
        if not self._run_final_validation(service_name, namespace):
            print("Final validation failed, rolling back...")
            self._rollback_to_blue(service_name, namespace)
            return False
        
        # Cleanup old blue deployment
        print("Cleaning up old blue deployment...")
        self._cleanup_old_blue_deployment(service_name, namespace)
        
        print("Blue-green deployment completed successfully")
        return True
    
    def _create_green_deployment(self, deployment_name: str, image: str, namespace: str, config: Dict[str, Any]):
        """Create green deployment"""
        
        deployment = client.V1Deployment(
            metadata=client.V1ObjectMeta(name=deployment_name),
            spec=client.V1DeploymentSpec(
                replicas=config.get('replicas', 3),
                selector=client.V1LabelSelector(
                    match_labels={"app": deployment_name}
                ),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={"app": deployment_name}
                    ),
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="ml-model",
                                image=image,
                                ports=[client.V1ContainerPort(container_port=8080)],
                                resources=client.V1ResourceRequirements(
                                    requests={
                                        "cpu": config.get('cpu_request', '500m'),
                                        "memory": config.get('memory_request', '1Gi')
                                    },
                                    limits={
                                        "cpu": config.get('cpu_limit', '1000m'),
                                        "memory": config.get('memory_limit', '2Gi')
                                    }
                                ),
                                env=[
                                    client.V1EnvVar(name="MODEL_VERSION", value=config.get('model_version', 'latest'))
                                ],
                                liveness_probe=client.V1Probe(
                                    http_get=client.V1HTTPGetAction(
                                        path="/health",
                                        port=8080
                                    ),
                                    initial_delay_seconds=30,
                                    period_seconds=10
                                ),
                                readiness_probe=client.V1Probe(
                                    http_get=client.V1HTTPGetAction(
                                        path="/ready",
                                        port=8080
                                    ),
                                    initial_delay_seconds=10,
                                    period_seconds=5
                                )
                            )
                        ]
                    )
                )
            )
        )
        
        self.v1.create_namespaced_deployment(namespace, deployment)
    
    def _wait_for_deployment_ready(self, deployment_name: str, namespace: str, timeout: int = 300) -> bool:
        """Wait for deployment to be ready"""
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                deployment = self.v1.read_namespaced_deployment(deployment_name, namespace)
                
                if (deployment.status.ready_replicas and 
                    deployment.status.ready_replicas == deployment.spec.replicas):
                    return True
                
                time.sleep(10)
                
            except Exception as e:
                print(f"Error checking deployment status: {e}")
                time.sleep(10)
        
        return False
    
    def _run_health_checks(self, deployment_name: str, namespace: str) -> bool:
        """Run health checks on green deployment"""
        
        # Get pod IPs
        pods = self.core_v1.list_namespaced_pod(
            namespace, 
            label_selector=f"app={deployment_name}"
        )
        
        for pod in pods.items:
            pod_ip = pod.status.pod_ip
            
            # Health check
            try:
                response = requests.get(f"http://{pod_ip}:8080/health", timeout=10)
                if response.status_code != 200:
                    return False
                
                # Model prediction test
                test_payload = {"features": [1.0, 2.0, 3.0]}
                response = requests.post(f"http://{pod_ip}:8080/predict", json=test_payload, timeout=10)
                if response.status_code != 200:
                    return False
                
            except Exception as e:
                print(f"Health check failed for pod {pod.metadata.name}: {e}")
                return False
        
        return True
    
    def _switch_traffic_to_green(self, service_name: str, green_deployment_name: str, namespace: str):
        """Switch service traffic to green deployment"""
        
        # Update service selector
        service = self.core_v1.read_namespaced_service(service_name, namespace)
        service.spec.selector = {"app": green_deployment_name}
        
        self.core_v1.patch_namespaced_service(service_name, namespace, service)
    
    def _run_final_validation(self, service_name: str, namespace: str) -> bool:
        """Run final validation after traffic switch"""
        
        # Get service endpoint
        service = self.core_v1.read_namespaced_service(service_name, namespace)
        service_ip = service.spec.cluster_ip
        
        # Run validation tests
        try:
            # Health check
            response = requests.get(f"http://{service_ip}:8080/health", timeout=10)
            if response.status_code != 200:
                return False
            
            # Multiple prediction tests
            for _ in range(10):
                test_payload = {"features": np.random.randn(5).tolist()}
                response = requests.post(f"http://{service_ip}:8080/predict", json=test_payload, timeout=10)
                if response.status_code != 200:
                    return False
            
            return True
            
        except Exception as e:
            print(f"Final validation failed: {e}")
            return False
    
    def _rollback_to_blue(self, service_name: str, namespace: str):
        """Rollback to blue deployment"""
        
        # Find blue deployment
        blue_deployment_name = f"{service_name}-blue"
        
        # Switch service back to blue
        service = self.core_v1.read_namespaced_service(service_name, namespace)
        service.spec.selector = {"app": blue_deployment_name}
        
        self.core_v1.patch_namespaced_service(service_name, namespace, service)
    
    def _cleanup_green_deployment(self, green_deployment_name: str, namespace: str):
        """Cleanup failed green deployment"""
        try:
            self.v1.delete_namespaced_deployment(green_deployment_name, namespace)
        except Exception as e:
            print(f"Error cleaning up green deployment: {e}")
    
    def _cleanup_old_blue_deployment(self, service_name: str, namespace: str):
        """Cleanup old blue deployment after successful switch"""
        
        blue_deployment_name = f"{service_name}-blue"
        
        try:
            self.v1.delete_namespaced_deployment(blue_deployment_name, namespace)
        except Exception as e:
            print(f"Error cleaning up blue deployment: {e}")

class CanaryDeployment:
    """Canary deployment strategy"""
    
    def __init__(self, k8s_config_path: str = None):
        if k8s_config_path:
            config.load_kube_config(k8s_config_path)
        else:
            config.load_incluster_config()
        
        self.v1 = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()
    
    def deploy(self, deployment_config: Dict[str, Any]) -> bool:
        """Execute canary deployment"""
        
        service_name = deployment_config['service_name']
        new_image = deployment_config['new_image']
        namespace = deployment_config.get('namespace', 'default')
        
        # Traffic split stages
        traffic_stages = [10, 25, 50, 100]
        
        canary_deployment_name = f"{service_name}-canary"
        
        # Create canary deployment
        print("Creating canary deployment...")
        self._create_canary_deployment(
            canary_deployment_name, 
            new_image, 
            namespace, 
            deployment_config
        )
        
        # Progressive traffic shifting
        for traffic_percentage in traffic_stages:
            print(f"Shifting {traffic_percentage}% traffic to canary...")
            
            self._update_traffic_split(
                service_name, 
                canary_deployment_name, 
                traffic_percentage, 
                namespace
            )
            
            # Monitor metrics
            if not self._monitor_canary_metrics(canary_deployment_name, namespace, traffic_percentage):
                print("Canary metrics failed, rolling back...")
                self._rollback_canary(service_name, canary_deployment_name, namespace)
                return False
            
            # Wait before next stage
            if traffic_percentage < 100:
                time.sleep(300)  # 5 minutes between stages
        
        # Promote canary to stable
        print("Promoting canary to stable...")
        self._promote_canary_to_stable(service_name, canary_deployment_name, namespace)
        
        print("Canary deployment completed successfully")
        return True
    
    def _create_canary_deployment(self, deployment_name: str, image: str, namespace: str, config: Dict[str, Any]):
        """Create canary deployment with smaller replica count"""
        
        # Canary starts with 1 replica
        deployment = client.V1Deployment(
            metadata=client.V1ObjectMeta(
                name=deployment_name,
                labels={"version": "canary"}
            ),
            spec=client.V1DeploymentSpec(
                replicas=1,
                selector=client.V1LabelSelector(
                    match_labels={"app": deployment_name, "version": "canary"}
                ),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={"app": deployment_name, "version": "canary"}
                    ),
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="ml-model",
                                image=image,
                                ports=[client.V1ContainerPort(container_port=8080)],
                                resources=client.V1ResourceRequirements(
                                    requests={
                                        "cpu": config.get('cpu_request', '500m'),
                                        "memory": config.get('memory_request', '1Gi')
                                    }
                                ),
                                env=[
                                    client.V1EnvVar(name="MODEL_VERSION", value="canary")
                                ]
                            )
                        ]
                    )
                )
            )
        )
        
        self.v1.create_namespaced_deployment(namespace, deployment)
    
    def _update_traffic_split(self, service_name: str, canary_deployment_name: str, traffic_percentage: int, namespace: str):
        """Update traffic split between stable and canary"""
        
        # This would typically use Istio, Linkerd, or similar service mesh
        # For simplicity, we'll use basic Kubernetes services
        
        stable_replicas = max(1, int(3 * (100 - traffic_percentage) / 100))
        canary_replicas = max(1, int(3 * traffic_percentage / 100))
        
        # Update stable deployment replicas
        stable_deployment_name = f"{service_name}-stable"
        stable_deployment = self.v1.read_namespaced_deployment(stable_deployment_name, namespace)
        stable_deployment.spec.replicas = stable_replicas
        self.v1.patch_namespaced_deployment(stable_deployment_name, namespace, stable_deployment)
        
        # Update canary deployment replicas
        canary_deployment = self.v1.read_namespaced_deployment(canary_deployment_name, namespace)
        canary_deployment.spec.replicas = canary_replicas
        self.v1.patch_namespaced_deployment(canary_deployment_name, namespace, canary_deployment)
    
    def _monitor_canary_metrics(self, canary_deployment_name: str, namespace: str, traffic_percentage: int) -> bool:
        """Monitor canary metrics and decide if deployment should continue"""
        
        monitoring_duration = 300  # 5 minutes
        check_interval = 30  # 30 seconds
        
        success_count = 0
        total_checks = monitoring_duration // check_interval
        
        for i in range(total_checks):
            try:
                # Get canary pods
                pods = self.core_v1.list_namespaced_pod(
                    namespace,
                    label_selector=f"app={canary_deployment_name},version=canary"
                )
                
                metrics_passed = True
                
                for pod in pods.items:
                    pod_ip = pod.status.pod_ip
                    
                    # Check error rate
                    response = requests.get(f"http://{pod_ip}:8080/metrics", timeout=10)
                    if response.status_code == 200:
                        metrics = response.json()
                        
                        error_rate = metrics.get('error_rate', 0)
                        latency_p95 = metrics.get('latency_p95', 0)
                        
                        # Check thresholds
                        if error_rate > 0.05:  # 5% error rate threshold
                            print(f"High error rate detected: {error_rate}")
                            metrics_passed = False
                        
                        if latency_p95 > 500:  # 500ms latency threshold
                            print(f"High latency detected: {latency_p95}ms")
                            metrics_passed = False
                
                if metrics_passed:
                    success_count += 1
                
                time.sleep(check_interval)
                
            except Exception as e:
                print(f"Error monitoring canary metrics: {e}")
                metrics_passed = False
        
        # Require 80% of checks to pass
        success_rate = success_count / total_checks
        return success_rate >= 0.8
    
    def _rollback_canary(self, service_name: str, canary_deployment_name: str, namespace: str):
        """Rollback canary deployment"""
        
        # Delete canary deployment
        try:
            self.v1.delete_namespaced_deployment(canary_deployment_name, namespace)
        except Exception as e:
            print(f"Error rolling back canary: {e}")
        
        # Ensure stable deployment has full traffic
        stable_deployment_name = f"{service_name}-stable"
        stable_deployment = self.v1.read_namespaced_deployment(stable_deployment_name, namespace)
        stable_deployment.spec.replicas = 3
        self.v1.patch_namespaced_deployment(stable_deployment_name, namespace, stable_deployment)
    
    def _promote_canary_to_stable(self, service_name: str, canary_deployment_name: str, namespace: str):
        """Promote canary to stable deployment"""
        
        # Get canary deployment
        canary_deployment = self.v1.read_namespaced_deployment(canary_deployment_name, namespace)
        
        # Create new stable deployment with canary image
        stable_deployment_name = f"{service_name}-stable"
        
        # Delete old stable deployment
        try:
            self.v1.delete_namespaced_deployment(stable_deployment_name, namespace)
            time.sleep(30)  # Wait for cleanup
        except Exception as e:
            print(f"Error deleting old stable deployment: {e}")
        
        # Rename canary to stable
        canary_deployment.metadata.name = stable_deployment_name
        canary_deployment.metadata.labels["version"] = "stable"
        canary_deployment.spec.replicas = 3
        canary_deployment.spec.selector.match_labels["version"] = "stable"
        canary_deployment.spec.template.metadata.labels["version"] = "stable"
        
        self.v1.create_namespaced_deployment(namespace, canary_deployment)
        
        # Delete canary deployment
        try:
            self.v1.delete_namespaced_deployment(canary_deployment_name, namespace)
        except Exception as e:
            print(f"Error cleaning up canary deployment: {e}")

# Deployment script examples
def deploy_with_strategy():
    """Example deployment script"""
    
    deployment_config = {
        'service_name': 'ml-model',
        'new_image': 'ml-model:v2.0.0',
        'namespace': 'production',
        'replicas': 3,
        'cpu_request': '500m',
        'memory_request': '1Gi',
        'cpu_limit': '1000m',
        'memory_limit': '2Gi',
        'model_version': 'v2.0.0'
    }
    
    # Choose deployment strategy
    strategy = os.getenv('DEPLOYMENT_STRATEGY', 'blue-green')
    
    if strategy == 'blue-green':
        deployer = BlueGreenDeployment()
        success = deployer.deploy(deployment_config)
    elif strategy == 'canary':
        deployer = CanaryDeployment()
        success = deployer.deploy(deployment_config)
    else:
        raise ValueError(f"Unknown deployment strategy: {strategy}")
    
    if success:
        print("Deployment completed successfully")
        return 0
    else:
        print("Deployment failed")
        return 1

if __name__ == "__main__":
    import sys
    import os
    
    exit_code = deploy_with_strategy()
    sys.exit(exit_code)
```

## Summary

CI/CD for Machine Learning provides:

1. **Automated Testing**: Comprehensive testing for data, models, and integration
2. **Version Control**: Systematic model and artifact versioning
3. **Deployment Automation**: Reliable deployment strategies with rollback capabilities
4. **Environment Management**: Consistent environments across development, staging, and production
5. **Monitoring Integration**: Continuous monitoring of model performance and system health

### Key Takeaways
- Adapt traditional CI/CD practices for ML-specific challenges
- Implement comprehensive testing strategies for data and models
- Use proper model versioning and registry systems
- Choose appropriate deployment strategies (blue-green, canary)
- Monitor model performance continuously
- Implement robust rollback mechanisms

### Next Steps
- Explore advanced orchestration tools (Kubeflow, MLflow)
- Implement feature stores for data management
- Add A/B testing frameworks
- Integrate with cloud-native ML platforms
- Implement automated model retraining triggers