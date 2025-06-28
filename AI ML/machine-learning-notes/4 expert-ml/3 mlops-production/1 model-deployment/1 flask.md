# Flask for Model Deployment

## Overview
Flask is a lightweight Python web framework that's perfect for deploying machine learning models as REST APIs. It provides a simple way to expose model predictions through HTTP endpoints.

## Why Flask for ML Deployment?
- **Lightweight**: Minimal overhead and easy to set up
- **Flexible**: Easy to customize and extend
- **Python Native**: Seamless integration with ML libraries
- **REST API**: Standard interface for model serving
- **Quick Prototyping**: Fast development and testing

## Basic Flask App for Model Deployment

### Project Structure
```
ml-flask-app/
├── app.py
├── model/
│   ├── trained_model.pkl
│   └── preprocessing.py
├── requirements.txt
├── templates/
│   └── index.html
└── static/
    └── style.css
```

### Simple Model Serving Example
```python
# app.py
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load trained model
with open('model/trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load preprocessor
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Preprocess data
        df_scaled = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(df_scaled)
        probability = model.predict_proba(df_scaled) if hasattr(model, 'predict_proba') else None
        
        # Prepare response
        response = {
            'prediction': int(prediction[0]),
            'probability': probability[0].tolist() if probability is not None else None,
            'status': 'success'
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

## Advanced Model Serving

### Image Classification API
```python
from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load model
model = torch.load('model/image_classifier.pth', map_location='cpu')
model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

class_names = ['cat', 'dog', 'bird', 'fish']  # Example classes

@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        # Get image from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        # Process image
        image = Image.open(file.stream).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
        
        return jsonify({
            'predicted_class': class_names[predicted_class],
            'confidence': float(confidence),
            'all_probabilities': {
                class_names[i]: float(probabilities[i]) 
                for i in range(len(class_names))
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_image_base64', methods=['POST'])
def predict_image_base64():
    try:
        data = request.json
        image_data = data['image']
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Process and predict (same as above)
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
        
        return jsonify({
            'predicted_class': class_names[predicted_class],
            'confidence': float(confidence)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

### Batch Prediction API
```python
@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        data = request.json
        batch_data = data['instances']
        
        # Convert to DataFrame
        df = pd.DataFrame(batch_data)
        
        # Preprocess
        df_scaled = scaler.transform(df)
        
        # Make predictions
        predictions = model.predict(df_scaled)
        probabilities = model.predict_proba(df_scaled) if hasattr(model, 'predict_proba') else None
        
        # Prepare response
        results = []
        for i, pred in enumerate(predictions):
            result = {
                'prediction': int(pred),
                'instance_id': i
            }
            if probabilities is not None:
                result['probability'] = probabilities[i].tolist()
            results.append(result)
        
        return jsonify({
            'predictions': results,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

## Error Handling and Validation

### Input Validation
```python
from functools import wraps
import jsonschema

# Define input schema
prediction_schema = {
    "type": "object",
    "properties": {
        "feature1": {"type": "number"},
        "feature2": {"type": "number"},
        "feature3": {"type": "string", "enum": ["A", "B", "C"]},
        "feature4": {"type": "number", "minimum": 0, "maximum": 100}
    },
    "required": ["feature1", "feature2", "feature3", "feature4"]
}

def validate_json(schema):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                jsonschema.validate(request.json, schema)
            except jsonschema.ValidationError as e:
                return jsonify({'error': f'Invalid input: {e.message}'}), 400
            except Exception as e:
                return jsonify({'error': 'Invalid JSON format'}), 400
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route('/predict', methods=['POST'])
@validate_json(prediction_schema)
def predict():
    # Prediction logic here
    pass
```

### Custom Error Handlers
```python
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'error'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500

@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        'error': 'Bad request',
        'status': 'error'
    }), 400
```

## Configuration and Environment Management

### Configuration Class
```python
# config.py
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    MODEL_PATH = os.environ.get('MODEL_PATH') or 'model/trained_model.pkl'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
class DevelopmentConfig(Config):
    DEBUG = True
    
class ProductionConfig(Config):
    DEBUG = False
    
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
```

### Using Configuration
```python
# app.py
from config import config

def create_app(config_name='default'):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Load model based on config
    model_path = app.config['MODEL_PATH']
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return app

app = create_app(os.environ.get('FLASK_ENV', 'default'))
```

## Logging and Monitoring

### Logging Setup
```python
import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging(app):
    if not app.debug:
        # File logging
        if not os.path.exists('logs'):
            os.mkdir('logs')
        
        file_handler = RotatingFileHandler(
            'logs/ml_api.log', 
            maxBytes=10240000, 
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        
        app.logger.setLevel(logging.INFO)
        app.logger.info('ML API startup')

# Request logging
@app.before_request
def log_request_info():
    app.logger.info('Request: %s %s', request.method, request.url)

@app.after_request
def log_response_info(response):
    app.logger.info('Response: %s', response.status)
    return response
```

### Performance Monitoring
```python
import time
from functools import wraps

def monitor_performance(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        
        # Log performance metrics
        app.logger.info(f'Function {f.__name__} took {end_time - start_time:.4f} seconds')
        
        return result
    return decorated_function

@app.route('/predict', methods=['POST'])
@monitor_performance
def predict():
    # Prediction logic
    pass
```

## Testing Flask APIs

### Unit Tests
```python
# test_app.py
import unittest
import json
from app import create_app

class MLAPITestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app('testing')
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
    
    def tearDown(self):
        self.app_context.pop()
    
    def test_health_check(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
    
    def test_prediction_endpoint(self):
        test_data = {
            'feature1': 1.0,
            'feature2': 2.0,
            'feature3': 'A',
            'feature4': 50
        }
        response = self.client.post('/predict',
                                  data=json.dumps(test_data),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('prediction', data)
    
    def test_invalid_input(self):
        test_data = {'invalid': 'data'}
        response = self.client.post('/predict',
                                  data=json.dumps(test_data),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()
```

## Security Considerations

### Rate Limiting
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    # Prediction logic
    pass
```

### API Key Authentication
```python
from functools import wraps

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != app.config['API_KEY']:
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    # Prediction logic
    pass
```

## Deployment Strategies

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

### Production Server (Gunicorn)
```python
# gunicorn_config.py
bind = "0.0.0.0:5000"
workers = 4
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
timeout = 30
keepalive = 2
preload_app = True
```

### Nginx Configuration
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Best Practices

1. **Model Loading**: Load model once at startup, not per request
2. **Input Validation**: Always validate and sanitize inputs
3. **Error Handling**: Provide clear, informative error messages
4. **Logging**: Log requests, responses, and errors for debugging
5. **Security**: Implement authentication, rate limiting, and input validation
6. **Performance**: Use caching, connection pooling, and efficient preprocessing
7. **Testing**: Write comprehensive tests for all endpoints
8. **Documentation**: Provide clear API documentation

## Common Pitfalls

1. **Memory Leaks**: Not properly managing model instances
2. **Blocking Operations**: Synchronous processing for large inputs
3. **No Input Validation**: Accepting any input without validation
4. **Poor Error Handling**: Generic error messages
5. **Security Issues**: No authentication or rate limiting
6. **Performance Issues**: Loading model per request

## Tools and Extensions

- **Flask-RESTful**: For building REST APIs
- **Flask-Limiter**: Rate limiting
- **Flask-JWT-Extended**: JWT authentication
- **Flask-CORS**: Cross-origin resource sharing
- **Flask-Caching**: Response caching
- **Marshmallow**: Serialization and validation

## Resources

- **Flask Documentation**: Official Flask documentation
- **Deployment Guides**: Production deployment best practices
- **Testing**: Flask testing documentation
- **Security**: OWASP security guidelines
- **Performance**: Flask performance optimization
