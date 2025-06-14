
## 🏗️ 14. AI System Design – Notes

### 📌 Overview:

AI system design focuses on building **robust, scalable, and maintainable** ML pipelines for real-world deployment. It spans the full lifecycle from data ingestion to post-deployment monitoring.

---

### 🧱 14.1 End-to-End ML Pipelines

#### ✅ Typical Stages:

1. **Data Collection** – from APIs, DBs, user input, sensors
2. **Data Processing** – ETL, feature engineering, handling nulls
3. **Model Training** – batch or online
4. **Evaluation & Validation** – metrics like AUC, F1, MAE
5. **Deployment** – API, stream, edge device
6. **Monitoring** – drift, latency, accuracy

#### ✅ Tools:

* **Airflow, Prefect**: Orchestration
* **MLflow, DVC**: Versioning
* **FastAPI, Docker**: Serving

---

### 📉 14.2 Model Monitoring

#### ✅ Monitor for:

* **Concept Drift**: Input data distribution changes
* **Data Drift**: Training vs live input changes
* **Model Performance**: Drop in metrics like accuracy/F1
* **Latency & Errors**: API response time, failure rate

#### ✅ Tools:

* **Evidently**, **WhyLabs**, **Prometheus + Grafana**

---

### 🔬 14.3 A/B Testing

#### ✅ Purpose:

* Compare two models (A and B) on live users
* Measure improvements statistically

#### ✅ Metrics to Track:

* Conversion rate, click-through rate
* Mean squared error / classification accuracy

#### ✅ Implementation Tips:

* Use consistent randomization
* Ensure sufficient sample size
* Avoid leakage across groups

---

### ⚙️ 14.4 Scalability (Batch vs Real-Time Inference)

#### ✅ Batch Inference:

* Run model on data periodically (e.g., nightly)
* Tools: Spark, Pandas, AWS Batch

#### ✅ Real-Time Inference:

* Serve predictions on-demand (e.g., fraud detection)
* Tools: FastAPI, TensorFlow Serving, TorchServe

#### ✅ Considerations:

| Aspect     | Batch                | Real-Time                  |
| ---------- | -------------------- | -------------------------- |
| Latency    | High (minutes/hours) | Low (milliseconds/seconds) |
| Cost       | Lower                | Higher                     |
| Complexity | Simpler              | More complex               |

