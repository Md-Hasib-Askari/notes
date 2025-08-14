# 06 — Advanced Boosting — GBM, XGBoost, LightGBM, CatBoost

Goal: Understand modern gradient boosting variants, regularization, early stopping, and categorical handling.

## Landscape
- scikit-learn GBM: solid baseline; `HistGradientBoosting*` is faster and supports early stopping.
- XGBoost: highly optimized, strong regularization, widespread use.
- LightGBM: histogram-based, fast on large/tabular data; leaf-wise growth.
- CatBoost: excellent with categorical features, strong defaults, minimal preprocessing.

## Regularization toolbox (common ideas)
- learning_rate: smaller with more trees (e.g., 0.03–0.1 with 500–2000 trees).
- max_depth / num_leaves: model complexity; leaf-wise methods (LightGBM) use `num_leaves`.
- subsample (row) and colsample_bytree/feature_fraction (col): randomness for robustness.
- min_child_weight / min_data_in_leaf: minimum data per leaf (strong regularizer).
- l1 (alpha) / l2 (lambda): weight regularization.
- gamma (XGB): min loss reduction to split; increases conservatism.

## Early stopping
- Split off a validation set; stop when no improvement after N rounds.
- Prevents overfitting and speeds tuning.

### Example: scikit-learn HistGradientBoosting with early stopping
```python
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401 (for older sklearn)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=5000, n_features=50, n_informative=10, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

hgb = HistGradientBoostingClassifier(
    learning_rate=0.06,
    max_depth=None,
    max_bins=255,
    l2_regularization=0.0,
    early_stopping=True,
    n_iter_no_change=30,
    random_state=42
)
hgb.fit(X_tr, y_tr)
print('Val score (holdout):', hgb.score(X_val, y_val))
```

### Example: XGBoost with early stopping
```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=8000, n_features=40, n_informative=12, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

xgb = XGBClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method='hist',
    random_state=42
)
xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='auc', early_stopping_rounds=50, verbose=False)
print('Best ntree limit:', xgb.best_ntree_limit)
```

### Example: LightGBM with early stopping
```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=8000, n_features=40, n_informative=12, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

lgbm = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42
)
lgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=[lgb.early_stopping(50, verbose=False)])
print('Best iteration:', lgbm.best_iteration_)
```

### Example: CatBoost with categorical features
```python
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

# Toy dataframe with categoricals
df = pd.DataFrame({
    'cat1': pd.Series(['A','B','A','C','B']*300, dtype='category'),
    'cat2': pd.Series(['X','Y','X','Z','Z']*300, dtype='category'),
    'num1': range(1500),
    'y':   [0,1,0,1,0]*300
})
cat_features = ['cat1', 'cat2']
X = df.drop(columns=['y'])
y = df['y']
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

cb = CatBoostClassifier(
    depth=6,
    learning_rate=0.1,
    n_estimators=1000,
    loss_function='Logloss',
    random_state=42,
    verbose=False
)
cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), cat_features=cat_features, use_best_model=True)
print('CatBoost val accuracy:', cb.score(X_val, y_val))
```

## Exercise
- On the same dataset, compare XGBoost, LightGBM, and CatBoost with early stopping. Track AUC/accuracy, best iteration, and fit time.

## Tips
- Start with modest depth/leaves and a small learning_rate; rely on early stopping to select n_estimators.
- For heavy class imbalance, use scale_pos_weight (XGB) or is_unbalance/class_weight (LightGBM/CatBoost).
- CatBoost often wins when many categorical features with high cardinality are present.
