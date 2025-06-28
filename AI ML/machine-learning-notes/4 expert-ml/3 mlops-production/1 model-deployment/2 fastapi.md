# FastAPI for Model Deployment

## Overview
FastAPI is a modern, high-performance web framework for building APIs with Python. It's particularly well-suited for ML model deployment due to its automatic documentation generation, type hints, async support, and excellent performance.

## Why FastAPI for ML Deployment?
- **High Performance**: One of the fastest Python frameworks
- **Automatic Documentation**: Interactive API docs with Swagger UI
- **Type Safety**: Built-in data validation using Pydantic
- **Async Support**: Handle concurrent requests efficiently
- **Easy Testing**: Built-in testing capabilities
- **Standards-based**: Based on OpenAPI and JSON Schema

## Basic FastAPI App for Model Deployment

### Project Structure
```
ml-fastapi-app/
├── main.py
├── models/
│   ├── __init__.py
│   ├── ml_model.py
│   └── trained_model.pkl
├── schemas/
│   ├── __init__.py
│   └── predictions.py
├── requirements.txt
├── tests/
│   └── test_main.py
└── Dockerfile
```

### Simple Model Serving Example
```python
# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from typing import List, Optional
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="ML Model API",
    description="API for machine learning model predictions",
    version="1.0.0"
)

# Load model at startup
model = None
scaler = None

@app.on_event("startup")
async def load_model():
    global model, scaler
    try:
        with open('models/trained_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: str
    feature4: int
    
    class Config:
        schema_extra = {
            "example": {
                "feature1": 1.5,
                "feature2": 2.3,
                "feature3": "category_a",
                "feature4": 10
            }
        }

class PredictionResponse(BaseModel):
    prediction: int
    probability: Optional[List[float]] = None
    confidence: float
    status: str

@app.get("/")
async def root():
    return {"message": "ML Model API is running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to DataFrame
        input_data = pd.DataFrame([request.dict()])
        
        # Preprocess data
        if scaler:
            input_scaled = scaler.transform(input_data.select_dtypes(include=[np.number]))
            input_data[input_data.select_dtypes(include=[np.number]).columns] = input_scaled
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Get probability if available
        probability = None
        confidence = 0.0
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_data)[0]
            probability = proba.tolist()
            confidence = float(max(proba))
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=probability,
            confidence=confidence,
            status="success"
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Advanced Model Serving

### Image Classification API
```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
from typing import Dict

app = FastAPI(title="Image Classification API")

# Load model
model = None
transform = None
class_names = ['cat', 'dog', 'bird', 'fish']

@app.on_event("startup")
async def load_model():
    global model, transform
    try:
        model = torch.load('models/image_classifier.pth', map_location='cpu')
        model.eval()
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        print("Image classification model loaded")
    except Exception as e:
        print(f"Error loading model: {e}")

class ImagePredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]

@app.post("/predict_image", response_model=ImagePredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class_idx].item()
        
        # Prepare response
        all_probs = {
            class_names[i]: float(probabilities[i]) 
            for i in range(len(class_names))
        }
        
        return ImagePredictionResponse(
            predicted_class=class_names[predicted_class_idx],
            confidence=confidence,
            all_probabilities=all_probs
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

class Base64ImageRequest(BaseModel):
    image_data: str  # Base64 encoded image
    
class Base64ImageResponse(BaseModel):
    predicted_class: str
    confidence: float

@app.post("/predict_image_base64", response_model=Base64ImageResponse)
async def predict_image_base64(request: Base64ImageRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(request.image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class_idx].item()
        
        return Base64ImageResponse(
            predicted_class=class_names[predicted_class_idx],
            confidence=confidence
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
```

### Batch Prediction API
```python
from typing import List

class BatchPredictionRequest(BaseModel):
    instances: List[PredictionRequest]

class BatchPredictionResponse(BaseModel):
    predictions: List[Dict]
    status: str
    processed_count: int

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.instances) > 100:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
    
    try:
        results = []
        
        for i, instance in enumerate(request.instances):
            # Convert to DataFrame
            input_data = pd.DataFrame([instance.dict()])
            
            # Preprocess
            if scaler:
                input_scaled = scaler.transform(input_data.select_dtypes(include=[np.number]))
                input_data[input_data.select_dtypes(include=[np.number]).columns] = input_scaled
            
            # Predict
            prediction = model.predict(input_data)[0]
            
            result = {
                "instance_id": i,
                "prediction": int(prediction)
            }
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_data)[0]
                result["probability"] = proba.tolist()
                result["confidence"] = float(max(proba))
            
            results.append(result)
        
        return BatchPredictionResponse(
            predictions=results,
            status="success",
            processed_count=len(results)
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")
```

## Async Model Serving

### Async Prediction with Background Tasks
```python
from fastapi import BackgroundTasks
import asyncio
import uuid
from typing import Dict

# In-memory store for async results (use Redis in production)
prediction_results: Dict[str, Dict] = {}

class AsyncPredictionRequest(BaseModel):
    callback_url: Optional[str] = None
    **PredictionRequest.__fields__

class AsyncPredictionResponse(BaseModel):
    task_id: str
    status: str
    message: str

async def process_prediction_async(task_id: str, request_data: dict):
    """Background task for processing predictions"""
    try:
        # Simulate processing time
        await asyncio.sleep(2)
        
        # Convert to DataFrame and predict
        input_data = pd.DataFrame([request_data])
        prediction = model.predict(input_data)[0]
        
        # Store result
        prediction_results[task_id] = {
            "status": "completed",
            "prediction": int(prediction),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # If callback URL provided, send result (implement HTTP POST)
        # await send_callback(request_data.get('callback_url'), result)
        
    except Exception as e:
        prediction_results[task_id] = {
            "status": "failed",
            "error": str(e),
            "timestamp": pd.Timestamp.now().isoformat()
        }

@app.post("/predict_async", response_model=AsyncPredictionResponse)
async def predict_async(request: AsyncPredictionRequest, background_tasks: BackgroundTasks):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    task_id = str(uuid.uuid4())
    
    # Add background task
    background_tasks.add_task(
        process_prediction_async, 
        task_id, 
        request.dict(exclude={'callback_url'})
    )
    
    # Store initial status
    prediction_results[task_id] = {
        "status": "processing",
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    return AsyncPredictionResponse(
        task_id=task_id,
        status="accepted",
        message="Prediction task started"
    )

@app.get("/prediction_status/{task_id}")
async def get_prediction_status(task_id: str):
    if task_id not in prediction_results:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return prediction_results[task_id]
```

## Advanced Features

### Model Versioning
```python
from enum import Enum

class ModelVersion(str, Enum):
    v1 = "v1"
    v2 = "v2"
    latest = "latest"

# Store multiple models
models = {}

@app.on_event("startup")
async def load_models():
    global models
    try:
        models["v1"] = pickle.load(open('models/model_v1.pkl', 'rb'))
        models["v2"] = pickle.load(open('models/model_v2.pkl', 'rb'))
        models["latest"] = models["v2"]  # Point to latest
        print("All model versions loaded")
    except Exception as e:
        print(f"Error loading models: {e}")

@app.post("/predict/{version}", response_model=PredictionResponse)
async def predict_with_version(version: ModelVersion, request: PredictionRequest):
    if version.value not in models:
        raise HTTPException(status_code=404, detail=f"Model version {version} not found")
    
    model = models[version.value]
    
    # Prediction logic (same as before but using versioned model)
    # ...
```

### Caching with Redis
```python
import redis
import json
from functools import wraps

# Redis connection
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_prediction(expiration_time: int = 3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from request data
            request_data = kwargs.get('request')
            cache_key = f"prediction:{hash(str(request_data.dict()))}"
            
            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # If not in cache, compute result
            result = await func(*args, **kwargs)
            
            # Store in cache
            redis_client.setex(
                cache_key, 
                expiration_time, 
                json.dumps(result.dict())
            )
            
            return result
        return wrapper
    return decorator

@app.post("/predict_cached", response_model=PredictionResponse)
@cache_prediction(expiration_time=1800)  # 30 minutes
async def predict_cached(request: PredictionRequest):
    # Same prediction logic as before
    # ...
```

## Error Handling and Validation

### Custom Exception Handlers
```python
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import traceback

class ModelError(Exception):
    def __init__(self, message: str, error_code: str = "MODEL_ERROR"):
        self.message = message
        self.error_code = error_code

@app.exception_handler(ModelError)
async def model_error_handler(request: Request, exc: ModelError):
    return JSONResponse(
        status_code=500,
        content={
            "error": exc.message,
            "error_code": exc.error_code,
            "type": "model_error"
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "details": exc.errors(),
            "type": "validation_error"
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "type": "internal_error",
            "traceback": traceback.format_exc() if app.debug else None
        }
    )
```

### Advanced Validation
```python
from pydantic import validator, Field
from typing import Union

class AdvancedPredictionRequest(BaseModel):
    feature1: float = Field(..., ge=0.0, le=100.0, description="Feature 1 (0-100)")
    feature2: float = Field(..., gt=0.0, description="Feature 2 (positive)")
    feature3: str = Field(..., regex="^(A|B|C)$", description="Category A, B, or C")
    feature4: int = Field(..., ge=1, le=1000, description="Feature 4 (1-1000)")
    
    @validator('feature1')
    def validate_feature1(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Feature1 must be between 0 and 100')
        return v
    
    @validator('feature3')
    def validate_feature3(cls, v):
        if v not in ['A', 'B', 'C']:
            raise ValueError('Feature3 must be A, B, or C')
        return v.upper()
    
    class Config:
        schema_extra = {
            "example": {
                "feature1": 50.0,
                "feature2": 25.5,
                "feature3": "A",
                "feature4": 100
            }
        }
```

## Testing FastAPI Apps

### Unit Tests
```python
# tests/test_main.py
from fastapi.testclient import TestClient
import pytest
import json
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data

def test_prediction_endpoint():
    test_data = {
        "feature1": 1.0,
        "feature2": 2.0,
        "feature3": "A",
        "feature4": 10
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "status" in data

def test_invalid_input():
    test_data = {"invalid": "data"}
    response = client.post("/predict", json=test_data)
    assert response.status_code == 422  # Validation error

def test_batch_prediction():
    test_data = {
        "instances": [
            {"feature1": 1.0, "feature2": 2.0, "feature3": "A", "feature4": 10},
            {"feature1": 2.0, "feature2": 3.0, "feature3": "B", "feature4": 20}
        ]
    }
    response = client.post("/predict_batch", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 2

@pytest.fixture
def mock_model(monkeypatch):
    """Mock model for testing"""
    class MockModel:
        def predict(self, X):
            return [1]
        
        def predict_proba(self, X):
            return [[0.3, 0.7]]
    
    monkeypatch.setattr("main.model", MockModel())
```

### Load Testing
```python
# load_test.py
import asyncio
import aiohttp
import time

async def make_request(session, url, data):
    async with session.post(url, json=data) as response:
        return await response.json()

async def load_test(num_requests=100):
    url = "http://localhost:8000/predict"
    test_data = {
        "feature1": 1.0,
        "feature2": 2.0,
        "feature3": "A",
        "feature4": 10
    }
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = [make_request(session, url, test_data) for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    
    print(f"Completed {num_requests} requests in {end_time - start_time:.2f} seconds")
    print(f"Average response time: {(end_time - start_time) / num_requests:.4f} seconds")

if __name__ == "__main__":
    asyncio.run(load_test())
```

## Monitoring and Logging

### Structured Logging
```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        
        return json.dumps(log_entry)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    logger.info(
        f"Request processed",
        extra={
            "method": request.method,
            "url": str(request.url),
            "status_code": response.status_code,
            "process_time": process_time
        }
    )
    
    return response
```

### Metrics Collection
```python
from prometheus_client import Counter, Histogram, generate_latest
import time

# Metrics
REQUEST_COUNT = Counter('ml_api_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('ml_api_request_duration_seconds', 'Request duration')
PREDICTION_COUNT = Counter('ml_api_predictions_total', 'Total predictions')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    # Record metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    
    return response

@app.get("/metrics")
async def get_metrics():
    return Response(generate_latest(), media_type="text/plain")
```

## Deployment

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Configuration
```python
# config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "ML Model API"
    debug: bool = False
    model_path: str = "models/trained_model.pkl"
    redis_url: str = "redis://localhost:6379"
    log_level: str = "INFO"
    max_batch_size: int = 100
    cache_ttl: int = 3600
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### Kubernetes Deployment
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api
    spec:
      containers:
      - name: ml-api
        image: ml-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: ml-api-service
spec:
  selector:
    app: ml-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Best Practices

1. **Use Pydantic Models**: Always define request/response schemas
2. **Async Operations**: Use async/await for I/O operations
3. **Error Handling**: Implement comprehensive error handling
4. **Validation**: Validate all inputs with appropriate constraints
5. **Documentation**: Leverage automatic API documentation
6. **Testing**: Write comprehensive tests for all endpoints
7. **Monitoring**: Implement logging, metrics, and health checks
8. **Security**: Add authentication, rate limiting, and input sanitization
9. **Performance**: Use caching, connection pooling, and efficient preprocessing
10. **Deployment**: Use production-ready servers (Gunicorn, Uvicorn)

## Tools and Extensions

- **Uvicorn**: ASGI server for production
- **Gunicorn**: Process manager with Uvicorn workers
- **Redis**: Caching and session storage
- **Prometheus**: Metrics collection
- **Sentry**: Error tracking
- **Celery**: Background task processing
- **SQLAlchemy**: Database ORM
- **Alembic**: Database migrations

## Resources

- **FastAPI Documentation**: Official FastAPI documentation
- **Pydantic**: Data validation and serialization
- **Uvicorn**: ASGI server documentation
- **Testing**: FastAPI testing guide
- **Deployment**: Production deployment guides
- **Security**: FastAPI security best practices
