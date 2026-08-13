# Deployment Architecture

**Document Version:** v1.1
**Author(s):** Guevarra; Group 4, III-DCSAD
**Date:** 2026-08-10
**Status:** Draft
**Purpose:** Define the deployment architecture for Odin ML modules

---

## 1. Overview

This document specifies the deployment architecture for the four Odin ML modules. The design follows the **separate containers** recommendation from the architecture discussion, with each module running as an independent microservice.

---

## 2. Deployment Strategy

### 2.1 Containerization

Each module is packaged as a **Docker container**:

| Module | Container | Port | Image |
|--------|-----------|------|-------|
| PFP Classifier | pfp-classifier | 8001 | odin/pfp-classifier:v1.0 |
| Forecaster | forecaster | 8002 | odin/forecaster:v1.0 |
| Anomaly Detector | anomaly-detector | 8003 | odin/anomaly-detector:v1.0 |
| Budget Optimizer | budget-optimizer | 8005 | odin/budget-optimizer:v1.0 |
| API Gateway | api-gateway | 8000 | odin/api-gateway:v1.0 |
| Transaction Service | transaction-service | 8004 | odin/transaction-service:v1.0 |

> Ports match the Odin-Paper system spec v0.3.0 deployment diagram (PFM 8001, forecaster 8002, anomaly 8003, gateway 8000, transaction 8004, budget optimizer 8005).

### 2.2 Why Separate Containers

| Benefit | Description |
|---------|-------------|
| Independent scaling | Scale anomaly detector (high QPS) separately from forecaster (CPU-intensive) |
| Independent deployment | Deploy PFP changes without restarting forecaster |
| Fault isolation | Anomaly detector crash doesn't affect PFP classification |
| Resource optimization | Give more memory to LSTM forecaster, more CPU to Random Forest PFP |
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
  
  budget-optimizer:
    build: ./budget-optimizer
    ports:
      - "8005:8005"
    volumes:
      - ./models/budget:/app/models
    environment:
      - MODEL_PATH=/app/models/budget_v1.0.json
  
  api-gateway:
    build: ./api-gateway
    ports:
      - "8000:8000"
    depends_on:
      - pfp-classifier
      - forecaster
      - anomaly-detector
      - budget-optimizer
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
| Budget Optimizer | 500m | 1 Gi | 1 Gi | None |
| API Gateway | 250m | 512 Mi | 512 Mi | None |
| Transaction Service | 500m | 1 Gi | 5 Gi | None |

**Total (without GPU):** 3.25 CPU, 6.5 Gi RAM, 11.5 Gi Disk
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

Models are trained locally into `training/models/` (gitignored, regenerable from `training/scripts/`) and published to cloud object storage:

```
<object-store>/odin-models/
├── pfp/
│   ├── v1.0.0/
│   │   ├── model.joblib
│   │   ├── feature_columns.json
│   │   ├── thresholds.json
│   │   └── metadata.json
│   └── latest -> v1.0.0
├── forecaster/
│   ├── v1.0.0/
│   │   ├── model.joblib
│   │   ├── feature_columns.json
│   │   └── metadata.json
│   └── latest -> v1.0.0
├── anomaly/
│   ├── v1.0.0/
│   │   ├── model.joblib
│   │   ├── baseline.json
│   │   └── metadata.json
│   └── latest -> v1.0.0
└── budget/
    ├── v1.0.0/
    │   ├── constraints.json
    │   └── metadata.json
    └── latest -> v1.0.0
```

### 5.2 Model Loading

Models are loaded from the local cache first, falling back to object storage (S3-compatible / GCS / Azure Blob). Pickle-free artifact loading keeps the runtime dependency surface minimal:

```python
import json
import joblib
from pathlib import Path

class ModelLoader:
    """Loads model artifacts by module + version.

    Artifacts live in `training/models/{module}/{version}/` locally; the
    deployment environment mounts the same layout into `/app/models/`.
    """

    def __init__(self, models_dir: Path = Path('/app/models')):
        self.models_dir = models_dir

    def load(self, module: str, version: str = 'latest'):
        root = self.models_dir / module / version
        if not root.exists():
            # resolve the `latest` symlink or fail fast
            root = self.models_dir / module / 'latest'
        return self._load_from_cache(root)

    def load_joblib(self, module: str, name: str, version: str = 'latest'):
        root = self._resolve(module, version)
        return joblib.load(root / name)

    def load_json(self, module: str, name: str, version: str = 'latest'):
        root = self._resolve(module, version)
        return json.loads((root / name).read_text())

    def _resolve(self, module: str, version: str) -> Path:
        root = self.models_dir / module / version
        if not root.exists():
            root = self.models_dir / module / 'latest'
        if not root.exists():
            raise FileNotFoundError(f'model artifacts missing: {module}/{version}')
        return root
```

### 5.3 Artifact Versioning

Every published artifact carries a metadata file that pins the training data, metrics, and dependency versions:

```json
{
    "model_id": "pfp_v1.3.0",
    "module": "pfp",
    "version": "1.3.0",
    "created_at": "2026-08-10T10:00:00Z",
    "trained_on": "synthetic_personas_12k",
    "training_data_hash": "sha256:4f9c2d1e8b6a3f7c9e0d1b2a3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d",
    "metrics": {
        "accuracy": 0.4857,
        "macro_f1": 0.4759
    },
    "features": ["income_stability_cv", "obligation_ratio", "savings_rate", ...],
    "dependencies": {
        "python": "3.14.4",
        "sklearn": "1.9.0",
        "pandas": "3.0.3"
    }
}
```

> **Consistency rule:** a model may only serve predictions against the exact `feature_columns.json` it was trained on. The serving API loads `feature_columns.json` alongside the model and rejects requests whose feature set does not match.

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
| Drift Detected | PSI or ADWIN/CUSUM flag | Flag retraining |

### 6.4 Drift Monitoring

Each module monitors drift on its input distribution and prediction distribution:

| Method | Use |
|--------|-----|
| **PSI (Population Stability Index)** | Categorical/feature-distribution shift (e.g., PFP label mix, spend ratios) |
| **ADWIN / CUSUM** | Online detection of gradual or abrupt performance drift on live prediction outcomes |

Drift checks run on a rolling window (e.g., 30 days of live predictions vs. the training-time baseline distribution) and publish a `drift_alert` event when the metric crosses its pre-registered threshold. A drift alert does **not** auto-deploy; it flags the artifact for evaluation in the CI/CD pipeline.

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
        python-version: '3.14.4'
    
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
*Updated: 2026-08-10 (v1.1 — added Budget Optimizer container, artifact versioning, drift monitoring)*
