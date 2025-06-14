
## 🚀 4.4: **Deployment**

### 🎯 Goal

Take your trained deep learning model and **deploy it for real-world use** — via APIs, mobile devices, or edge environments.

---

## 📦 1. Model Serialization & Export

### 🔹 PyTorch

* **TorchScript**: Export to `.pt` format using `torch.jit.trace()` or `torch.jit.script()`.
* **ONNX**: Interoperable format for converting models across frameworks.

```python
# TorchScript example
traced_model = torch.jit.trace(model, sample_input)
traced_model.save("model.pt")

# Convert to ONNX
torch.onnx.export(model, sample_input, "model.onnx")
```

### 🔹 TensorFlow/Keras

* Save as `.pb` or `.h5`
* Convert to:

  * **TensorFlow Lite** for mobile
  * **TensorFlow\.js** for browsers

---

## 🌐 2. Serving the Model

### Option A: **FastAPI / Flask** (For REST APIs)

```python
from fastapi import FastAPI
import torch

app = FastAPI()
model = torch.jit.load("model.pt")

@app.post("/predict")
def predict(data: List[float]):
    input_tensor = torch.tensor(data).unsqueeze(0)
    output = model(input_tensor)
    return {"prediction": output.argmax().item()}
```

✅ Use [Uvicorn](https://www.uvicorn.org/) or Gunicorn for production hosting.

### Option B: **TensorFlow Serving**

For high-performance inference:

```bash
docker run -p 8501:8501 --name=tf_model \
  -v "/path/to/saved_model:/models/model" \
  -e MODEL_NAME=model -t tensorflow/serving
```

### Option C: **TorchServe**

For deploying PyTorch models in production (supports REST & gRPC APIs).

---

## 📱 3. Mobile & Edge Deployment

| Platform            | Format                      | Tools                    |
| ------------------- | --------------------------- | ------------------------ |
| Android/iOS         | TensorFlow Lite (`.tflite`) | TFLite Converter         |
| Android/iOS         | ONNX → CoreML               | `onnx-coreml`, `mlmodel` |
| Jetson/Raspberry Pi | TorchScript / TensorRT      | NVIDIA TensorRT, PyTorch |

---

## ☁️ 4. Cloud Deployment Options

| Provider     | Product                    | Use Case                    |
| ------------ | -------------------------- | --------------------------- |
| AWS          | SageMaker, Lambda, ECS     | Scalable inference/API      |
| Google Cloud | AI Platform, Cloud Run     | Managed serving + autoscale |
| Azure        | Azure ML, App Service      | Enterprise integrations     |
| Hugging Face | Inference API              | Instant deployment for NLP  |
| Vercel       | Frontend + FastAPI backend | Lightweight hosting         |

---

## 🛡️ 5. Performance, Monitoring & Scaling

* **Model quantization**: Reduce size + improve speed (e.g., INT8)
* **Monitoring tools**: Prometheus + Grafana, Weights & Biases, MLflow
* **Auto-scaling**: Use Kubernetes or Serverless functions (e.g., Cloud Run)

---

## 🧪 Exercises

### ✅ Theory

* What’s the difference between TorchScript and ONNX?
* How does model quantization help mobile apps?

### ✅ Practical

* Convert a PyTorch model to TorchScript and serve with FastAPI
* Deploy a Keras model to TensorFlow Lite for Android
* Use Docker to containerize and run your model API

