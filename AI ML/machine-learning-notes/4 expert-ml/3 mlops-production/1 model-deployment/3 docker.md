# Docker for ML Model Deployment

## Overview
Docker is a containerization platform that packages applications and their dependencies into lightweight, portable containers. For ML model deployment, Docker ensures consistent environments across development, testing, and production.

## Why Docker for ML Deployment?
- **Consistency**: Same environment across all stages
- **Portability**: Runs anywhere Docker is supported
- **Isolation**: Dependencies don't conflict with host system
- **Scalability**: Easy to scale horizontally
- **Version Control**: Container images are versioned
- **Reproducibility**: Exact same environment every time

## Basic Concepts
- **Image**: Blueprint for creating containers
- **Container**: Running instance of an image
- **Dockerfile**: Instructions to build an image
- **Registry**: Storage for Docker images (Docker Hub, ECR, etc.)
- **Volume**: Persistent data storage
- **Network**: Communication between containers

## Simple ML Model Dockerfile

### Basic Python ML App
```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "app.py"]
```

### Multi-stage Build for Optimization
```dockerfile
# Multi-stage Dockerfile
# Stage 1: Build environment
FROM python:3.9 as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++

# Copy and install requirements
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Stage 2: Production environment
FROM python:3.9-slim

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Update PATH to include user packages
ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "app.py"]
```

## Optimized Dockerfiles for Different Frameworks

### PyTorch Model Deployment
```dockerfile
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /app

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and application
COPY models/ ./models/
COPY src/ ./src/
COPY app.py .

# Set environment variables
ENV PYTHONPATH=/app
ENV TORCH_HOME=/app/models

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### TensorFlow Model Deployment
```dockerfile
FROM tensorflow/tensorflow:2.10.0

WORKDIR /app

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install requirements
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Set environment variables
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PYTHONPATH=/app

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "app.py"]
```

### Lightweight Model with Alpine
```dockerfile
FROM python:3.9-alpine

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache \
    gcc \
    musl-dev \
    linux-headers \
    curl

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Remove build dependencies
RUN apk del gcc musl-dev linux-headers

# Copy application
COPY . .

# Create non-root user
RUN adduser -D -s /bin/sh appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "app.py"]
```

## Docker Compose for Multi-Service Setup

### ML API with Redis and Database
```yaml
# docker-compose.yml
version: '3.8'

services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:password@postgres:5432/mldb
    depends_on:
      - redis
      - postgres
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  postgres:
    image: postgres:14-alpine
    environment:
      - POSTGRES_DB=mldb
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d mldb"]
      interval: 10s
      timeout: 5s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - ml-api
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
```

### Development vs Production Compose
```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  ml-api:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - .:/app
      - /app/__pycache__
    environment:
      - DEBUG=True
      - LOG_LEVEL=DEBUG
    ports:
      - "8000:8000"
    command: ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client", "app.py"]

# docker-compose.prod.yml
version: '3.8'

services:
  ml-api:
    image: my-ml-api:latest
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
    environment:
      - DEBUG=False
      - LOG_LEVEL=INFO
```

## Advanced Docker Techniques

### Model Loading Optimization
```dockerfile
# Dockerfile with model optimization
FROM python:3.9-slim

WORKDIR /app

# Install dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy and optimize model loading
COPY models/ ./models/
RUN python -c "import pickle; import torch; \
    model = torch.load('models/model.pth', map_location='cpu'); \
    torch.jit.save(torch.jit.script(model), 'models/model_optimized.pt')"

# Copy application code
COPY src/ ./src/
COPY app.py .

# Warm up the model (optional)
RUN python -c "from src.model_loader import load_model; load_model()"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### GPU Support
```dockerfile
# GPU-enabled Dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install ML libraries with GPU support
COPY requirements-gpu.txt .
RUN pip3 install --no-cache-dir -r requirements-gpu.txt

# Copy application
COPY . .

# Set CUDA environment
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8000

CMD ["python3", "app.py"]
```

### Docker Build Arguments and Secrets
```dockerfile
# Dockerfile with build arguments
ARG PYTHON_VERSION=3.9
FROM python:${PYTHON_VERSION}-slim

ARG MODEL_VERSION=latest
ARG ENVIRONMENT=production

WORKDIR /app

# Use build-time secrets (requires Docker BuildKit)
RUN --mount=type=secret,id=aws_credentials \
    pip install awscli && \
    aws configure set aws_access_key_id $(cat /run/secrets/aws_credentials | head -1) && \
    aws s3 cp s3://my-models/model-${MODEL_VERSION}.pkl ./models/

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV ENVIRONMENT=${ENVIRONMENT}

CMD ["python", "app.py"]
```

## Docker Build and Run Commands

### Building Images
```bash
# Basic build
docker build -t ml-api:latest .

# Build with arguments
docker build --build-arg PYTHON_VERSION=3.9 --build-arg MODEL_VERSION=v2.0 -t ml-api:v2.0 .

# Build for multiple architectures
docker buildx build --platform linux/amd64,linux/arm64 -t ml-api:latest --push .

# Build with secrets
echo "your-secret-key" | docker build --secret id=api_key,src=- -t ml-api:latest .
```

### Running Containers
```bash
# Basic run
docker run -p 8000:8000 ml-api:latest

# Run with environment variables
docker run -p 8000:8000 -e DEBUG=true -e LOG_LEVEL=DEBUG ml-api:latest

# Run with volumes
docker run -p 8000:8000 -v $(pwd)/models:/app/models -v $(pwd)/logs:/app/logs ml-api:latest

# Run with resource limits
docker run -p 8000:8000 --memory=512m --cpus=0.5 ml-api:latest

# Run with GPU support
docker run --gpus all -p 8000:8000 ml-api:latest

# Run in detached mode
docker run -d -p 8000:8000 --name ml-api-container ml-api:latest
```

## Container Management

### Logging and Monitoring
```bash
# View logs
docker logs ml-api-container

# Follow logs
docker logs -f ml-api-container

# View container stats
docker stats ml-api-container

# Execute commands in running container
docker exec -it ml-api-container bash

# Copy files to/from container
docker cp model.pkl ml-api-container:/app/models/
```

### Health Checks and Debugging
```dockerfile
# Enhanced health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' ml-api-container

# View health check logs
docker inspect --format='{{range .State.Health.Log}}{{.Output}}{{end}}' ml-api-container
```

## Production Deployment Strategies

### Docker Swarm
```yaml
# docker-stack.yml
version: '3.8'

services:
  ml-api:
    image: ml-api:latest
    ports:
      - "8000:8000"
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
        failure_action: rollback
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
    networks:
      - ml-network

networks:
  ml-network:
    driver: overlay
```

### Kubernetes Deployment
```yaml
# k8s-deployment.yaml
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
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
          readOnly: true
      volumes:
      - name: model-storage
        configMap:
          name: ml-models
```

## Security Best Practices

### Secure Dockerfile
```dockerfile
FROM python:3.9-slim

# Update and install security patches
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create non-root user with specific UID/GID
RUN groupadd -r appgroup --gid=1000 && \
    useradd -r -g appgroup --uid=1000 --home-dir=/app --shell=/sbin/nologin appuser

# Copy application files and set ownership
COPY --chown=appuser:appgroup . .

# Remove write permissions from sensitive files
RUN chmod -R 755 /app && \
    chmod -R 644 /app/*.py

# Switch to non-root user
USER appuser

# Use specific port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use exec form for proper signal handling
CMD ["python", "app.py"]
```

### Container Security Scanning
```bash
# Scan image for vulnerabilities
docker scan ml-api:latest

# Use Trivy for security scanning
trivy image ml-api:latest

# Scan during build
docker build --target security-scan .
```

## Performance Optimization

### Image Size Optimization
```dockerfile
# Multi-stage build for minimal size
FROM python:3.9 as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.9-slim
WORKDIR /app

# Copy only necessary files
COPY --from=builder /root/.local /root/.local
COPY app.py .
COPY models/ ./models/

# Remove unnecessary files
RUN find /usr/local -name '*.pyc' -delete && \
    find /usr/local -name '__pycache__' -delete

ENV PATH=/root/.local/bin:$PATH

CMD ["python", "app.py"]
```

### .dockerignore File
```dockerignore
# .dockerignore
**/__pycache__
**/*.pyc
**/*.pyo
**/*.pyd
**/.pytest_cache
**/.coverage
**/.git
**/.gitignore
**/README.md
**/Dockerfile*
**/docker-compose*
**/.dockerignore
**/node_modules
**/.env
**/tests
**/docs
**/*.log
```

## Monitoring and Observability

### Container Metrics
```yaml
# docker-compose with monitoring
version: '3.8'

services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    labels:
      - "prometheus.io/scrape=true"
      - "prometheus.io/port=8000"
      - "prometheus.io/path=/metrics"

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  grafana_data:
```

## Best Practices

1. **Use Multi-stage Builds**: Reduce image size and improve security
2. **Run as Non-root User**: Enhance container security
3. **Use Specific Tags**: Avoid `latest` tag in production
4. **Minimize Layers**: Combine RUN commands when possible
5. **Use .dockerignore**: Exclude unnecessary files
6. **Health Checks**: Implement proper health checks
7. **Resource Limits**: Set memory and CPU limits
8. **Security Scanning**: Regularly scan images for vulnerabilities
9. **Logging**: Use structured logging and proper log levels
10. **Environment Variables**: Use env vars for configuration

## Common Issues and Solutions

### Large Image Sizes
```dockerfile
# Problem: Large images
FROM python:3.9  # ~900MB

# Solution: Use slim variants
FROM python:3.9-slim  # ~150MB

# Or use Alpine for smallest size
FROM python:3.9-alpine  # ~50MB
```

### Model Loading Performance
```python
# Optimize model loading
import torch

# Load model once at startup
@app.on_event("startup")
async def load_model():
    global model
    model = torch.jit.load('model_optimized.pt')
    model.eval()
```

### Memory Issues
```dockerfile
# Set memory limits
docker run -m 512m ml-api:latest

# In Dockerfile
LABEL memory_limit="512m"
```

## Tools and Utilities

- **Docker Desktop**: GUI for Docker management
- **Portainer**: Web-based Docker management
- **Watchtower**: Automatic container updates
- **Dive**: Analyze Docker image layers
- **Hadolint**: Dockerfile linter
- **Docker Bench**: Security benchmarking

## Resources

- **Docker Documentation**: Official Docker documentation
- **Best Practices**: Docker development best practices
- **Security**: Docker security guidelines
- **Optimization**: Container optimization techniques
- **Orchestration**: Kubernetes and Docker Swarm guides
