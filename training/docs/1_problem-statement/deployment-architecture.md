# Deployment Architecture

**Document Version:** v1.0
**Author(s):** Guevarra
**Date:** 2026-07-17
**Status:** Draft
**Purpose:** Define the deployment architecture for Odin ML modules

---

## 1. Overview

This document specifies the deployment architecture for the three Odin ML modules. The design follows the **separate containers** recommendation from the architecture discussion, with each module running as an independent microservice.

---

## 2. Deployment Strategy

### 2.1 Containerization

Each module is packaged as a **Docker container**:

| Module | Container | Port | Image |
|--------|-----------|------|-------|
| PFP Classifier | pfp-classifier | 8001 | odin/pfp-classifier:v1.0 |
| Forecaster | forecaster | 8002 | odin/forecaster:v1.0 |
| Anomaly Detector | anomaly-detector | 8003 | odin/anomaly-detector:v1.0 |
| API Gateway | api-gateway | 8000 | odin/api-gateway:v1.0 |
| Transaction Service | transaction-service | 8004 | odin/transaction-service:v1.0 |

### 2.2 Why Separate Containers

| Benefit | Description |
|---------|-------------|
| Independent scaling | Scale anomaly detector (high QPS) separately from forecaster (CPU-intensive) |
| Independent deployment | Deploy PFP changes without restarting forecaster |
| Fault isolation | Anomaly detector crash doesn't affect PFP classification |
| Resource optimization | Give more memory to LSTM forecaster, more CPU to XGBoost PFP |
| Team specialization | Different developers can own different modules |

---

## 3. Infrastructure

### 3.1 Cloud Provider (TBD)

Options under consideration:

| Provider | Pros | Cons |
|----------|------|------|
| AWS | Most mature, ECS/Lambda | Complex pricing |
| GCP | Good ML support, Cloud Run | Smaller ecosystem |
| Azure | Enterprise integration | Steeper learning curve |
| DigitalOcean | Simple, affordable | Limited ML services |

### 3.2 Container Orchestration

**Development:** Docker Compose
```yaml
# docker-compose.dev.yml
version: '3.8'
services:
  pfp-classifier:
    build: ./pfp-classifier
    ports:
      - "8001:8001"
    volumes:
      - ./models/pfp:/app/models
    environment:
      - MODEL_PATH=/app/models/pfp_v1.pkl
  
  forecaster:
    build: ./forecaster
    ports:
      - "8002:8002"
    volumes:
      - ./models/forecaster:/app/models
    environment:
      - MODEL_PATH=/app/models/lstm_v1.pt
  
  anomaly-detector:
    build: ./anomaly-detector
    ports:
      - "8003:8003"
    volumes:
      - ./models/anomaly:/app/models
    environment:
      - MODEL_PATH=/app/models/iforest_v1.pkl
  
  api-gateway:
    build: ./api-gateway
    ports:
      - "8000:8000"
    depends_on:
      - pfp-classifier
      - forecaster
      - anomaly-detector
```

**Production:** Kubernetes (AWS EKS / GCP GKE)

```yaml
# k8s/pfp-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pfp-classifier
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pfp-classifier
  template:
    metadata:
      labels:
        app: pfp-classifier
    spec:
      containers:
      - name: pfp-classifier
        image: odin/pfp-classifier:v1.0
        ports:
        - containerPort: 8001
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8001
          initialDelaySeconds: 5
          periodSeconds: 5
```

---

## 4. Resource Requirements

### 4.1 Per-Module Resources

| Module | CPU | Memory | Disk | GPU |
|--------|-----|--------|------|-----|
| PFP Classifier | 500m | 1 Gi | 1 Gi | None |
| Forecaster | 1000m | 2 Gi | 2 Gi | Optional (LSTM) |
| Anomaly Detector | 500m | 1 Gi | 1 Gi | None |
| API Gateway | 250m | 512 Mi | 512 Mi | None |
| Transaction Service | 500m | 1 Gi | 5 Gi | None |

**Total (without GPU):** 2.75 CPU, 5.5 Gi RAM, 9.5 Gi Disk
**Total (with GPU):** Add 1 GPU for LSTM training

### 4.2 Scaling Rules

```yaml
# Horizontal Pod Autoscaler for Anomaly Detector
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: anomaly-detector-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: anomaly-detector
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## 5. Model Serving

### 5.1 Model Storage

Models are stored in cloud object storage:

```
s3://odin-models/
├── pfp/
│   ├── v1.0/
│   │   ├── model.pkl
│   │   ├── scaler.pkl
│   │   ├── thresholds.json
│   │   └── metadata.json
│   └── latest -> v1.0
├── forecaster/
│   ├── v1.0/
│   │   ├── model.pt
│   │   ├── tokenizer.json
│   │   └── metadata.json
│   └── latest -> v1.0
└── anomaly/
    ├── v1.0/
    │   ├── model.pkl
    │   ├── baseline.json
    │   └── metadata.json
    └── latest -> v1.0
```

### 5.2 Model Loading

```python
import boto3
import pickle
from pathlib import Path

class ModelLoader:
    def __init__(self, bucket='odin-models'):
        self.s3 = boto3.client('s3')
        self.bucket = bucket
        self.cache_dir = Path('/tmp/models')
        self.cache_dir.mkdir(exist_ok=True)
    
    def load(self, module, version='latest'):
        cache_path = self.cache_dir / module / version
        if cache_path.exists():
            return self._load_from_cache(cache_path)
        
        # Download from S3
        self._download(module, version)
        return self._load_from_cache(cache_path)
    
    def _download(self, module, version):
        prefix = f'{module}/{version}/'
        objects = self.s3.list_objects(Bucket=self.bucket, Prefix=prefix)
        
        for obj in objects.get('Contents', []):
            local_path = self.cache_dir / obj['Key']
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file(self.bucket, obj['Key'], str(local_path))
    
    def _load_from_cache(self, path):
        if path.suffix == '.pkl':
            return pickle.loads(path.read_bytes())
        elif path.suffix == '.pt':
            import torch
            return torch.load(path)
```

### 5.3 Model Versioning

```python
# Model metadata schema
{
    "model_id": "pfp_v1.0.0",
    "module": "pfp",
    "version": "1.0.0",
    "created_at": "2026-07-15T10:00:00Z",
    "trained_on": "synthetic_personas_14k",
    "metrics": {
        "accuracy": 0.87,
        "macro_f1": 0.85,
        "cohen_kappa": 0.82
    },
    "features": ["income_cv", "obligation_ratio", ...],
    "dependencies": {
        "python": "3.10",
        "sklearn": "1.3.0",
        "xgboost": "2.0.0"
    }
}
```

---

## 6. Monitoring & Observability

### 6.1 Health Checks

```python
# Each module exposes these endpoints
@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.get("/ready")
def ready():
    # Check model is loaded
    if model is None:
        return {"status": "not ready", "reason": "model not loaded"}
    return {"status": "ready"}

@app.get("/metrics")
def metrics():
    return {
        "requests_total": request_counter,
        "avg_latency_ms": avg_latency,
        "error_rate": error_rate,
        "model_version": current_version
    }
```

### 6.2 Logging

```python
import logging
import json

class StructuredLogger:
    def __init__(self, module_name):
        self.logger = logging.getLogger(module_name)
        self.handler = logging.StreamHandler()
        self.handler.setFormatter(
            logging.Formatter('%(message)s')
        )
        self.logger.addHandler(self.handler)
    
    def log_prediction(self, user_id, input_features, output, latency_ms):
        self.logger.info(json.dumps({
            "event": "prediction",
            "module": self.module_name,
            "user_id": user_id,
            "input_features": input_features,
            "output": output,
            "latency_ms": latency_ms,
            "timestamp": datetime.utcnow().isoformat()
        }))
```

### 6.3 Alerting

| Alert | Condition | Action |
|-------|-----------|--------|
| High Error Rate | > 5% errors in 5 min | Page on-call |
| High Latency | p99 > 500ms for 5 min | Investigate |
| Model Stale | No model update in 30 days | Trigger retraining |
| Memory High | > 80% utilization | Scale up |
| Disk Full | > 90% disk usage | Clean cache |

---

## 7. CI/CD Pipeline

### 7.1 Build Stage

```yaml
# .github/workflows/build.yml
name: Build and Test
on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: pip install -r requirements.txt
    
    - name: Run tests
      run: pytest tests/ -v
    
    - name: Build Docker image
      run: docker build -t odin/${{ matrix.module }}:${{ github.sha }} .
    
    - name: Push to ECR
      run: |
        aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_URL
        docker push odin/${{ matrix.module }}:${{ github.sha }}
```

### 7.2 Deploy Stage

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production
on:
  workflow_run:
    workflows: ["Build and Test"]
    types: [completed]
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - name: Update Kubernetes manifest
      run: |
        kubectl set image deployment/${{ matrix.module }} \
          ${{ matrix.module }}=odin/${{ matrix.module }}:${{ github.sha }}
    
    - name: Rollout status
      run: kubectl rollout status deployment/${{ matrix.module }}
```

---

## 8. Cost Estimation

### 8.1 Development (Google Colab)

| Resource | Cost |
|----------|------|
| Colab GPU (L4) | Free (limited hours) |
| Colab T4 | Free (limited hours) |
| Storage (Google Drive) | Free (15 GB) |

**Total:** $0 (within limits)

### 8.2 Production (AWS)

| Resource | Monthly Cost |
|----------|--------------|
| ECS Fargate (2 tasks) | ~$50 |
| RDS PostgreSQL | ~$30 |
| ElastiCache Redis | ~$25 |
| S3 Storage | ~$5 |
| CloudWatch | ~$10 |
| Data Transfer | ~$10 |
| **Total** | **~$130/month** |

### 8.3 Production (GCP)

| Resource | Monthly Cost |
|----------|--------------|
| Cloud Run (2 instances) | ~$40 |
| Cloud SQL | ~$30 |
| Memorystore Redis | ~$25 |
| Cloud Storage | ~$5 |
| Cloud Logging | ~$10 |
| **Total** | **~$110/month** |

---

## 9. Security

### 9.1 API Authentication

```python
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.get("/api/v1/pfp/classify")
async def classify(request: PFPRequest, token = Security(security)):
    # Verify token
    user = verify_token(token.credentials)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    # Check permissions
    if not user.has_permission("pfp:classify"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Process request
    return pfp_classifier.classify(request)
```

### 9.2 Data Encryption

- **In transit:** TLS 1.3 for all API calls
- **At rest:** AES-256 for model files and user data
- **In memory:** Optional encryption for sensitive features

### 9.3 Access Control

| Role | Permissions |
|------|-------------|
| User | Read own data, trigger predictions |
| Admin | Read all data, manage models |
| Model Trainer | Upload models, view metrics |
| Auditor | Read logs, view metrics |

---

## 10. Expected Outputs

| Output | Description | Location |
|--------|-------------|----------|
| `docker-compose.yml` | Development setup | `Odin-ML/` |
| `Dockerfile` | Per module | `Odin-ML/{module}/` |
| `k8s/` | Kubernetes manifests | `Odin-ML/deploy/` |
| `.github/workflows/` | CI/CD pipelines | `Odin-ML/.github/` |

---

## 11. RRL Justifications

| Concept | RRL Support | Topic |
|---------|-------------|-------|
| Containerization | Standard deployment practice | 12.B.I |
| Microservices | Scalable architecture | 12.B.I |
| Model versioning | ML best practice | 12.B.II |
| Health checks | Fault tolerance | 12.B.II |
| Structured logging | Observability | 12.B.III |

---

*Document created: 2026-07-17*
