# 14 — Ensemble in Real-Time Systems

Goal: Build ensembles that learn and predict under streaming constraints.

## Constraints
- Latency budgets, memory limits, concept drift, non-stationary data.

## Online Bagging/Boosting
- Replace bootstrap sampling with Poisson(1) draws per instance (Oza & Russell).

## River examples
```python
from river import ensemble, tree, metrics, evaluate, stream

model = ensemble.LeveragingBaggingClassifier(
    model=tree.HoeffdingTreeClassifier(),
    n_models=10,
    random_state=42
)
metric = metrics.F1()

for x, y in stream.iter_array(X, y):
    y_pred = model.predict_one(x)
    metric.update(y, y_pred)
    model.learn_one(x, y)
print('F1:', metric.get())
```

### Adaptive Random Forest (drift-aware)
```python
from river.ensemble import AdaptiveRandomForestClassifier
arf = AdaptiveRandomForestClassifier(n_models=10, random_state=42)
# Same partial learn loop as above
```

## Engineering tips
- Use sliding windows or fading factors for metrics.
- Monitor drift detectors; reinitialize weak learners when needed.
- Pre-allocate feature encoders for categorical streams.
- For serving, prefer lightweight models and batch predictions when possible.
