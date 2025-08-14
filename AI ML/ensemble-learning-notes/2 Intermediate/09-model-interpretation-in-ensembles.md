# 09 — Model Interpretation in Ensembles

Goal: Use SHAP and LIME to understand feature contributions in tree ensembles.

## Why interpret
- Trust and debugging: detect leakage, understand spurious correlations.
- Communication: explain predictions to stakeholders.

## Global vs Local
- Global: which features matter overall (permutation importance, mean SHAP values).
- Local: why a specific prediction happened (SHAP values, LIME explanations).

## SHAP with tree ensembles (fast TreeExplainer)
```python
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=500, random_state=42).fit(X_tr, y_tr)

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_te)
# Summary plot (requires matplotlib backend)
shap.summary_plot(shap_values, X_te, show=False)
```

## LIME (model-agnostic local explanations)
```python
from lime.lime_tabular import LimeTabularExplainer
import numpy as np

explainer = LimeTabularExplainer(
    training_data=X_tr,
    feature_names=[f'f{i}' for i in range(X_tr.shape[1])],
    class_names=['class0', 'class1'],
    mode='classification'
)
exp = explainer.explain_instance(X_te[0], rf.predict_proba, num_features=5)
# exp.show_in_notebook()  # or exp.as_list()
```

## Best practices
- Validate insights with permutation importance.
- Beware correlated features; interpret groups, not single columns.
- For XGBoost/LightGBM, built-in SHAP is efficient; avoid LIME on very high-dimensional data.

## Exercise
- Train RF/XGBoost on a tabular dataset; produce permutation importance, SHAP summary, and a few local explanations; compare narratives.
