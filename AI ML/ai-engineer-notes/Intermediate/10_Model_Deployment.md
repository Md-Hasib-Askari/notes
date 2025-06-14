

## 🚀 10. Model Deployment – Notes

### 📌 Overview:

Model deployment is the process of **exposing a trained ML model to real-world users** via APIs, frontends, or applications. It turns your `.pkl`, `.pt`, or `.h5` file into a **live service**.

---

### 🧱 10.1 Flask / FastAPI for Serving Models

#### ✅ Flask (Lightweight, simple)

```python
from flask import Flask, request, jsonify
import joblib

model = joblib.load('model.pkl')
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction.tolist()})
```

#### ✅ FastAPI (Faster, async support, better docs)

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

class Input(BaseModel):
    features: list

app = FastAPI()
model = joblib.load('model.pkl')

@app.post('/predict')
def predict(input: Input):
    prediction = model.predict([input.features])
    return {'prediction': prediction.tolist()}
```

---

### 📦 10.2 Docker Basics

#### ✅ Why Docker?

* Packages your app + environment
* Makes deployment platform-independent

#### ✅ Example `Dockerfile`:

```Dockerfile
FROM python:3.10
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t ml-app .
docker run -p 8000:8000 ml-app
```

---

### 🔁 10.3 REST APIs for Inference

#### ✅ Flow:

Client → HTTP POST/GET → Server (Flask/FastAPI) → Model → Prediction → JSON response

#### ✅ Tools:

* `requests` for client
* Swagger UI (auto-generated in FastAPI)
* Postman for testing endpoints

---

### 🎛️ 10.4 Streamlit / Gradio for Simple Frontends

#### ✅ Streamlit (Great for internal tools)

```python
import streamlit as st
import joblib

model = joblib.load("model.pkl")
features = st.text_input("Enter features (comma separated)")
if st.button("Predict"):
    x = [float(i) for i in features.split(",")]
    st.write("Prediction:", model.predict([x])[0])
```

#### ✅ Gradio (More plug-and-play)

```python
import gradio as gr
import joblib

model = joblib.load("model.pkl")

def predict(features):
    return model.predict([features])

gr.Interface(fn=predict, inputs="text", outputs="label").launch()
```
