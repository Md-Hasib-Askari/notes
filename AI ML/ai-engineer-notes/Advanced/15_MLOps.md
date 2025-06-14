
## ⚙️ 15. MLOps – Notes

### 📌 Overview:

**MLOps** (Machine Learning Operations) is the set of practices to **deploy, monitor, and maintain ML models in production reliably and efficiently**. Think of it as DevOps + ML.

---

### 🔁 15.1 CI/CD for ML

#### ✅ What It Is:

* **CI** (Continuous Integration): Automate testing and validation of code + data.
* **CD** (Continuous Deployment): Automatically deploy models to production.

#### ✅ MLOps-Specific Pipelines:

1. Data validation (e.g., using `Great Expectations`)
2. Model training and evaluation
3. Model versioning
4. Deployment trigger based on metrics

#### ✅ Tools:

* **GitHub Actions**, **GitLab CI**, **Jenkins**
* **Tecton**, **ZenML**, **KubeFlow Pipelines**

---

### 📦 15.2 Model Registry (MLflow, Weights & Biases)

#### ✅ Model Registry:

* Stores trained models with version control
* Includes metadata, metrics, artifacts

#### ✅ Popular Options:

* **MLflow**: Open-source, supports experiment tracking, registry, UI
* **Weights & Biases (W\&B)**: Logging, dashboards, artifacts, sweeps

```python
import mlflow
mlflow.log_metric("accuracy", 0.92)
```

---

### 🐳 15.3 Kubernetes Basics for ML Apps

#### ✅ Why Use Kubernetes:

* Scalable container orchestration
* Auto-healing, load balancing
* Ideal for deploying ML microservices

#### ✅ Key Concepts:

* **Pods**: Smallest deployable unit
* **Services**: Expose pods
* **Deployments**: Manage pod replicas

#### ✅ ML Stack Examples:

* JupyterHub on K8s
* TorchServe + Ingress
* GPU-enabled training clusters

---

### 🚀 15.4 TensorFlow Serving / TorchServe

#### ✅ TensorFlow Serving:

* High-performance, flexible serving system for TensorFlow models

```bash
docker run -p 8501:8501 \
  -v "/models:/models/my_model" \
  -e MODEL_NAME=my_model \
  tensorflow/serving
```

#### ✅ TorchServe:

* Model server for PyTorch
* Supports REST & gRPC
* Custom handlers for preprocessing/postprocessing
