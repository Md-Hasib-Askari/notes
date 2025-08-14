# 13 — Ensemble with Model Selection

Goal: Choose and tune base learners automatically with Bayesian optimization or AutoML.

## Bayesian optimization (Optuna example)
```python
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    model_type = trial.suggest_categorical('model', ['rf', 'xgb', 'lgbm'])
    if model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int('n_estimators', 200, 1000),
            max_depth=trial.suggest_int('max_depth', 3, 20),
            max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8]),
            random_state=42, n_jobs=-1)
    elif model_type == 'xgb':
        model = XGBClassifier(
            n_estimators=2000, learning_rate=trial.suggest_float('lr', 0.01, 0.2, log=True),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            subsample=trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree=trial.suggest_float('colsample', 0.6, 1.0),
            reg_lambda=trial.suggest_float('l2', 1e-3, 10.0, log=True),
            tree_method='hist', random_state=42)
    else:
        model = LGBMClassifier(
            n_estimators=2000, learning_rate=trial.suggest_float('lr', 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int('leaves', 31, 255),
            subsample=trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree=trial.suggest_float('colsample', 0.6, 1.0),
            reg_lambda=trial.suggest_float('l2', 1e-3, 10.0, log=True),
            random_state=42)
    score = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1).mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print('Best:', study.best_trial.params)
```

## AutoML
- TPOT: genetic programming that explores pipelines and ensembles.
- auto-sklearn: Bayesian optimization + meta-learning + ensembling.

## Tips
- Set realistic time/compute budgets; cache data; use early stopping within models.
- Enforce reproducibility (random seeds) and track trials (MLflow/W&B).
