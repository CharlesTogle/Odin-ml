# Module Integration Spec

**Document Version:** v1.1
**Author(s):** Guevarra; Group 4, III-DCSAD
**Date:** 2026-08-10
**Status:** Draft
**Purpose:** Define how the four Odin modules (PFP, Forecaster, Anomaly Detector, Budget Optimizer) integrate

---

## 1. Overview

This document specifies the integration architecture for the four Odin modules:

1. **PFP Classifier** — Personal Financial Profile classification
2. **Forecaster** — Monthly Spending Prediction
3. **Anomaly Detector** — Unusual Transaction Detection
4. **Budget Optimizer** — Budget allocation recommendation (constraint optimization)

> **Note:** The **Budget Optimizer** was defined in this version (v1.1) following the Odin-Paper system spec v0.3.0, which added it to the deployment and integration architecture. Its model design is defined in the Budget Optimizer MDD v1.0 within `module-design-document.md`.

The integration design follows these principles:
- **Loose coupling** — Modules communicate via APIs, not shared state
- **Event-driven** — Modules react to transaction events
- **Fail-safe** — Module failures don't crash the system
- **Composable** — Outputs can be combined or used independently

---

## 2. System Architecture

### 2.1 High-Level Flow

```
[User App] → [API Gateway] → [Transaction Service]
                                    ↓
                            [Event Bus (Kafka/RabbitMQ)]
                                    ↓
                    ┌───────────────┼───────────────┬───────────────┐
                    ↓               ↓               ↓               ↓
            [PFP Classifier] [Forecaster]  [Anomaly Detector] [Budget Optimizer]
                    ↓               ↓               ↓               ↓
                    └───────────────┼───────────────┴───────────────┘
                                    ↓
                            [Response Aggregator]
                                    ↓
                            [User App (Response)]
```

### 2.2 Module Responsibilities

| Module | Input | Output | Trigger |
|--------|-------|--------|---------|
| PFP Classifier | Transaction history (3 months) or questionnaire answers | PFP class (8 classes) + 3 dimension scores + confidence + status | On classification request |
| Forecaster | Transaction history (6 months) | Predicted next month expenses (+ per-category) | On forecast request |
| Anomaly Detector | Transaction + baseline | Anomaly score + explanation | On every transaction |
| Budget Optimizer | Available funds, allocations, restrictions, forecast, preferences | Recommended allocations + feasibility + explanation | On request or periodic refresh |

---

## 3. Data Contracts

### 3.1 Transaction Event

```json
{
  "event_id": "txn_123456",
  "user_id": "user_789",
  "timestamp": "2026-07-15T10:30:00Z",
  "amount": 1500.00,
  "currency": "PHP",
  "category": "food",
  "description": "Grocery shopping",
  "account_id": "acc_001",
  "balance_after": 25000.00,
  "metadata": {
    "merchant": "SM Supermarket",
    "payment_method": "debit_card"
  }
}
```

### 3.2 PFP Classification Request

```json
{
  "request_id": "pfp_req_001",
  "user_id": "user_789",
  "transactions": [
    {
      "month": "2026-04",
      "total_income": 25000.00,
      "total_expenses": 20000.00,
      "categories": {
        "food": 8000.00,
        "housing": 5000.00,
        "transport": 3000.00,
        "health": 1500.00,
        "education": 2500.00
      }
    }
  ],
  "options": {
    "mode": "STANDARD",
    "include_confidence": true
  }
}
```

### 3.3 PFP Classification Response

```json
{
  "response_id": "pfp_resp_001",
  "request_id": "pfp_req_001",
  "user_id": "user_789",
  "classification": {
    "prediction": "Stable/Obligated/Tolerant",
    "financial_stability_score": 0.88,
    "financial_weight_score": 0.71,
    "financial_tolerance_score": 0.64,
    "confidence": 0.87,
    "status": "SUCCESS",
    "tier_used": 3,
    "model_name": "random_forest"
  },
  "metadata": {
    "processing_time_ms": 45,
    "model_version": "v1.3.0"
  }
}
```

> The PFP response follows the system-spec §5.5 output contract: `prediction` (one of eight PFP classes), three calibrated dimension scores (0–1), `confidence` (0–1), and `status` (`SUCCESS`/`FAILURE`/`FALLBACK`). The module never throws; on failure it returns `FALLBACK` values.

### 3.4 Forecast Request

```json
{
  "request_id": "fcst_req_001",
  "user_id": "user_789",
  "horizon_months": 1,
  "transactions": [
    {
      "month": "2026-01",
      "total_income": 25000.00,
      "total_expenses": 20000.00
    }
  ],
  "options": {
    "include_breakdown": true,
    "confidence_intervals": true
  }
}
```

### 3.5 Forecast Response

```json
{
  "response_id": "fcst_resp_001",
  "request_id": "fcst_req_001",
  "user_id": "user_789",
  "forecast": {
    "next_month_total": 21500.00,
    "confidence_interval": {
      "lower": 18000.00,
      "upper": 25000.00
    },
    "category_breakdown": {
      "food": 8500.00,
      "housing": 5000.00,
      "transport": 3200.00,
      "health": 1800.00,
      "education": 3000.00
    }
  },
  "metadata": {
    "processing_time_ms": 120,
    "model_version": "v1.0.0"
  }
}
```

### 3.6 Anomaly Detection Response

```json
{
  "response_id": "anom_resp_001",
  "transaction_id": "txn_123456",
  "user_id": "user_789",
  "anomaly": {
    "is_anomalous": true,
    "score": 0.92,
    "threshold": 0.75,
    "explanation": [
      "Transaction amount 3x higher than average",
      "Unusual merchant category",
      "Out-of-pattern timing"
    ]
  },
  "metadata": {
    "processing_time_ms": 30,
    "model_version": "v1.0.0"
  }
}
```

### 3.7 Budget Optimization Request

```json
{
  "request_id": "bgt_req_001",
  "user_id": "user_789",
  "payload": {
    "available_funds": 25000.00,
    "period": {
      "start": "2026-08-01",
      "end": "2026-08-31"
    },
    "categories": [
      {
        "category_id": "essentials_food",
        "restriction_level": "PROTECTED",
        "floor": 6000.0,
        "ceiling": 9000.0,
        "priority_weight": 0.4,
        "current_spend": 6800.0
      },
      {
        "category_id": "discretionary_leisure",
        "restriction_level": "FREE",
        "floor": 0.0,
        "ceiling": 4000.0,
        "priority_weight": 0.2,
        "current_spend": 2200.0
      }
    ],
    "target_ratios": {
      "essentials_food": 0.30,
      "discretionary_leisure": 0.10
    },
    "forecast": {
      "expected_income": 26000.0,
      "expected_total_spend": 21500.0
    }
  },
  "options": {
    "include_reasoning": true
  }
}
```

### 3.8 Budget Optimization Response

```json
{
  "response_id": "bgt_resp_001",
  "request_id": "bgt_req_001",
  "user_id": "user_789",
  "recommendation": {
    "allocations": {
      "essentials_food": 7800.0,
      "discretionary_leisure": 2500.0
    },
    "utilization_rate": 0.96,
    "constraint_satisfaction": 1.0,
    "feasibility": "FEASIBLE"
  },
  "explanations": [
    {
      "category_id": "essentials_food",
      "reason": "Protected category — held at or above floor"
    },
    {
      "category_id": "discretionary_leisure",
      "reason": "Adjusted toward target ratio; within ceiling"
    }
  ],
  "status": "SUCCESS",
  "metadata": {
    "processing_time_ms": 40,
    "strategy_used": "tier2",
    "model_version": "v1.0.0"
  }
}
```

---

## 4. API Endpoints

### 4.1 PFP Classifier API

```
POST /api/v1/pfp/classify
POST /api/v1/pfp/classify/batch
GET  /api/v1/pfp/user/{user_id}/history
GET  /api/v1/pfp/user/{user_id}/latest
```

### 4.2 Forecaster API

```
POST /api/v1/forecast/predict
POST /api/v1/forecast/predict/batch
GET  /api/v1/forecast/user/{user_id}/history
```

### 4.3 Anomaly Detector API

```
POST /api/v1/anomaly/detect
POST /api/v1/anomaly/detect/batch
GET  /api/v1/anomaly/user/{user_id}/alerts
GET  /api/v1/anomaly/user/{user_id}/baseline
```

### 4.4 Budget Optimizer API

```
POST /api/v1/budget/recommend
POST /api/v1/budget/recommend/batch
GET  /api/v1/budget/user/{user_id}/latest
```

### 4.5 Gateway API

```
POST /api/v1/analyze          # Run all four modules
GET  /api/v1/user/{user_id}   # Get combined insights
POST /api/v1/classify         # Convenience endpoint
POST /api/v1/forecast         # Convenience endpoint
POST /api/v1/budget           # Convenience endpoint
```

---

## 5. Event Flow

### 5.1 Transaction Ingestion

```python
# Transaction Service publishes event
event_bus.publish('transactions', {
    'event_type': 'new_transaction',
    'transaction': transaction_data
})

# Modules subscribe to events
pfp_consumer.subscribe('transactions', handle_transaction)
forecast_consumer.subscribe('transactions', handle_transaction)
anomaly_consumer.subscribe('transactions', handle_transaction)
budget_consumer.subscribe('transactions', handle_transaction)
```

### 5.2 PFP Classification Trigger

```python
def handle_transaction(event):
    transaction = event['transaction']
    user_id = transaction['user_id']
    
    # Check if we have enough history
    history = get_user_history(user_id, months=3)
    if len(history) < 3:
        return  # Not enough data
    
    # Reclassify every 3 months or on significant change
    last_classification = get_last_classification(user_id)
    if should_reclassify(last_classification, history):
        classification = pfp_classifier.classify(history)
        save_classification(user_id, classification)
        
        # Publish classification event
        event_bus.publish('classifications', {
            'user_id': user_id,
            'classification': classification
        })
```

### 5.3 Forecast Trigger

```python
def handle_transaction(event):
    transaction = event['transaction']
    user_id = transaction['user_id']
    
    # Check if we need to update forecast
    last_forecast = get_last_forecast(user_id)
    if should_update_forecast(last_forecast, transaction):
        history = get_user_history(user_id, months=6)
        forecast = forecaster.predict(history)
        save_forecast(user_id, forecast)
        
        # Publish forecast event
        event_bus.publish('forecasts', {
            'user_id': user_id,
            'forecast': forecast
        })
```

### 5.4 Anomaly Detection Trigger

```python
def handle_transaction(event):
    transaction = event['transaction']
    user_id = transaction['user_id']
    
    # Always check every transaction
    baseline = get_user_baseline(user_id)
    if baseline is None:
        # First 6 months: establish baseline, no detection
        update_baseline(user_id, transaction)
        return
    
    # Detect anomaly
    result = anomaly_detector.detect(transaction, baseline)
    
    if result['is_anomalous']:
        # Publish alert
        event_bus.publish('anomaly_alerts', {
            'user_id': user_id,
            'transaction': transaction,
            'anomaly': result
        })
```

### 5.5 Budget Optimization Trigger

```python
def handle_transaction(event):
    transaction = event['transaction']
    user_id = transaction['user_id']

    # Refresh recommendation on significant change or on schedule
    last_recommendation = get_last_recommendation(user_id)
    if should_refresh_budget(last_recommendation, transaction):
        allocations = get_user_allocations(user_id)
        forecast = get_last_forecast(user_id)
        recommendation = budget_optimizer.optimize(user_id, allocations, forecast)
        save_recommendation(user_id, recommendation)

        # Publish recommendation event (never auto-applied — BR-03)
        event_bus.publish('budget_recommendations', {
            'user_id': user_id,
            'recommendation': recommendation
        })
```

---

## 6. Response Aggregation

### 6.1 Combined Analysis

When a user requests a full analysis, the gateway aggregates all module outputs:

```python
async def analyze_user(user_id):
    # Run all modules in parallel
    pfp_task = pfp_client.classify(user_id)
    forecast_task = forecast_client.predict(user_id)
    anomaly_task = anomaly_client.detect(user_id)
    budget_task = budget_client.recommend(user_id)

    pfp_result, forecast_result, anomaly_result, budget_result = await asyncio.gather(
        pfp_task, forecast_task, anomaly_task, budget_task
    )

    # Combine results
    return {
        'user_id': user_id,
        'classification': pfp_result,
        'forecast': forecast_result,
        'recent_anomalies': anomaly_result,
        'budget_recommendation': budget_result,
        'insights': generate_insights(pfp_result, forecast_result, anomaly_result, budget_result)
    }
```

### 6.2 Insight Generation

```python
def generate_insights(pfp, forecast, anomalies, budget):
    insights = []

    # PFP-based insights
    if pfp['prediction'].startswith('Variable/Obligated'):
        insights.append({
            'type': 'warning',
            'message': 'High obligation ratio with variable income. Consider building emergency fund.',
            'priority': 'high'
        })

    # Forecast-based insights
    if forecast['next_month_total'] > pfp['current_expenses'] * 1.2:
        insights.append({
            'type': 'warning',
            'message': 'Spending forecast 20% higher than current. Review budget.',
            'priority': 'medium'
        })

    # Anomaly-based insights
    if anomalies['count'] > 0:
        insights.append({
            'type': 'alert',
            'message': f'{anomalies["count"]} unusual transactions detected.',
            'priority': 'high'
        })

    # Budget-based insights
    if budget['recommendation']['feasibility'] == 'INFEASIBLE':
        insights.append({
            'type': 'warning',
            'message': 'The suggested budget is infeasible; discretionary categories have been reduced.',
            'priority': 'high'
        })

    return insights
```

---

## 7. Error Handling

### 7.1 Module Failure

```python
async def safe_classify(user_id):
    try:
        result = await pfp_client.classify(user_id)
        return result
    except PFPServiceError as e:
        # Fallback to rule-based classification
        history = get_user_history(user_id, months=3)
        return rule_based_classify(history)
    except Exception as e:
        # Return uncertain classification
        return {
            'label': 'Unknown',
            'confidence': 0.0,
            'error': str(e)
        }
```

### 7.2 Graceful Degradation

```python
async def analyze_user_safe(user_id):
    results = {}

    # Try each module independently
    try:
        results['pfp'] = await pfp_client.classify(user_id)
    except Exception:
        results['pfp'] = None

    try:
        results['forecast'] = await forecast_client.predict(user_id)
    except Exception:
        results['forecast'] = None

    try:
        results['anomalies'] = await anomaly_client.detect(user_id)
    except Exception:
        results['anomalies'] = None

    try:
        results['budget'] = await budget_client.recommend(user_id)
    except Exception:
        results['budget'] = None

    # Return whatever succeeded
    return results
```

---

## 8. Expected Outputs

| Output | Description | Location |
|--------|-------------|----------|
| `api-spec.yaml` | OpenAPI 3.0 specification | `1_problem-statement/` |
| `event-schema.json` | Event bus message schemas | `1_problem-statement/` |
| `integration-diagram.png` | System architecture diagram | `1_problem-statement/` |

---

## 9. RRL Justifications

| Concept | RRL Support | Topic |
|---------|-------------|-------|
| Event-driven architecture | Microservices best practice | 12.B.I |
| API-first design | Standard integration pattern | 12.B.I |
| Graceful degradation | Fault tolerance | 12.B.II |
| Response aggregation | Service composition | 12.B.I |

---

*Document created: 2026-07-17*
*Updated: 2026-08-10 (v1.1 — added Budget Optimizer module)*
