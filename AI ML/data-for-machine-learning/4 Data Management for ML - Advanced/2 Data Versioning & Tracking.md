# 📘 Advanced → Data Versioning & Tracking (for ML)

In ML projects, both **data** and **models** change frequently. Data versioning and tracking ensure that experiments are **reproducible, traceable, and comparable** over time.

---

## **1. Why Data Versioning & Tracking?**

* **Reproducibility** → Know exactly which dataset & preprocessing steps led to which model.
* **Collaboration** → Multiple people can work on the same project without overwriting results.
* **Experiment management** → Keep track of hyperparameters, metrics, and dataset versions.
* **Audit & Compliance** → Required in regulated industries (finance, healthcare).

---

## **2. Data Versioning Tools**

### **(a) DVC (Data Version Control)**

* Git-like tool for data & ML models.
* Stores metadata in Git, but large files in external storage (S3, GCP, Azure).
* Tracks datasets, preprocessing scripts, and models.

📌 Example workflow:

```bash
dvc add data.csv       # track dataset
git add data.csv.dvc   # commit pointer file
dvc push               # push to remote storage
```

---

### **(b) Git LFS (Large File Storage)**

* Git extension for handling big files.
* Suitable for small projects but less powerful than DVC.

---

### **(c) LakeFS / Delta Lake**

* Manage **data lake versioning** at scale.
* Provide commit, rollback, and branching for big data.

---

## **3. Experiment Tracking Tools**

### **(a) MLflow**

* Open-source platform for ML lifecycle.
* Tracks experiments: parameters, metrics, models, artifacts.
* Provides UI to compare runs.

📌 Example (logging with MLflow):

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", 0.87)
    mlflow.sklearn.log_model(model, "random_forest")
```

---

### **(b) Weights & Biases (W\&B)**

* Cloud-based experiment tracking & visualization.
* Logs metrics, hyperparameters, gradients.
* Interactive dashboards for comparing runs.

---

### **(c) Neptune.ai, Comet.ml**

* Similar to W\&B, often used in enterprise settings.

---

## **4. Feature Stores**

* Specialized databases for storing & serving **features consistently** for training and inference.
* Prevents “training–serving skew” (when features differ between offline and online environments).
* Examples: **Feast (open-source), Tecton, Hopsworks**.

---

## **5. Best Practices**

* Always version datasets **along with code**.
* Track **preprocessing steps** (scaling, encoding) to reproduce models.
* Store **metadata**: dataset source, schema, preprocessing, and transformation history.
* Use **experiment tracking tools** to log parameters + metrics.
* Use **feature stores** in production ML pipelines.

---

## ✅ Key Takeaways

1. Data versioning ensures datasets can be **reproduced & rolled back**.
2. DVC + Git is the most popular combo for ML versioning.
3. MLflow, W\&B, Neptune.ai provide **experiment tracking**.
4. Feature stores help maintain **consistency** between training & production.

---