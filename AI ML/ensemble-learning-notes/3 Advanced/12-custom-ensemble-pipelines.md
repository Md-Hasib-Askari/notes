# 12 — Custom Ensemble Pipelines

Goal: Implement bespoke bagging/boosting and scale with parallel processing.

## From-scratch Bagging (sketch)
```python
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import resample
from joblib import Parallel, delayed
import numpy as np

class SimpleBaggingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, n_estimators=50, bootstrap=True, random_state=None, n_jobs=-1):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.estimators_ = []

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n = len(X)
        def fit_one(seed):
            est = clone(self.base_estimator)
            idx = rng.randint(0, n, n) if self.bootstrap else np.arange(n)
            est.fit(X[idx], y[idx])
            return est
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(delayed(fit_one)(i) for i in range(self.n_estimators))
        return self

    def predict_proba(self, X):
        P = np.mean([est.predict_proba(X) for est in self.estimators_], axis=0)
        return P

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
```

## Tiny Gradient Boosting (squared loss, conceptual)
```python
from sklearn.tree import DecisionTreeRegressor
import numpy as np

class TinyGBM:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees_ = []
        self.init_ = 0.0

    def fit(self, X, y):
        self.init_ = np.mean(y)
        pred = np.full_like(y, self.init_, dtype=float)
        for _ in range(self.n_estimators):
            residual = y - pred
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residual)
            update = tree.predict(X)
            pred += self.learning_rate * update
            self.trees_.append(tree)
        return self

    def predict(self, X):
        pred = np.full(X.shape[0], self.init_)
        for tree in self.trees_:
            pred += self.learning_rate * tree.predict(X)
        return pred
```

## Parallelization
- Use joblib.Parallel for independent fits (bagging, CV).
- For large data, persist intermediate artifacts (joblib.dump) and stream I/O.

## Tips
- Keep APIs sklearn-compatible to leverage pipelines and CV.
- Add random_state parameters everywhere for reproducibility.
