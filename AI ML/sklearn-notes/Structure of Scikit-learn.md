## 🔧 1. **Core Components**

| Component         | Description                                                 |
| ----------------- | ----------------------------------------------------------- |
| `Estimator`       | Any model or transformer with `.fit()` method               |
| `Predictor`       | Estimator with `.predict()` (e.g., classifiers, regressors) |
| `Transformer`     | Estimator with `.transform()` (e.g., scaler, encoder)       |
| `Pipeline`        | Chains preprocessing and model steps                        |
| `Model Selection` | Tools for CV, hyperparameter search                         |
| `Metrics`         | Tools to evaluate models (e.g., accuracy, F1, AUC)          |

---

## 📦 2. **Main Modules Overview**

| Module                    | What it Contains                                          |
| ------------------------- | --------------------------------------------------------- |
| `sklearn.linear_model`    | Linear models (Logistic, Ridge, Lasso, etc.)              |
| `sklearn.tree`            | Decision Trees                                            |
| `sklearn.ensemble`        | RandomForest, GradientBoosting, Voting, etc.              |
| `sklearn.svm`             | Support Vector Machines                                   |
| `sklearn.naive_bayes`     | Naive Bayes models                                        |
| `sklearn.neighbors`       | KNN classifiers, regressors                               |
| `sklearn.cluster`         | KMeans, DBSCAN, Agglomerative                             |
| `sklearn.decomposition`   | PCA, TruncatedSVD, NMF                                    |
| `sklearn.preprocessing`   | Scaling, encoding, normalization                          |
| `sklearn.model_selection` | Train/test split, cross-validation, hyperparameter tuning |
| `sklearn.pipeline`        | Pipeline and FeatureUnion                                 |
| `sklearn.metrics`         | Accuracy, precision, recall, F1, ROC, regression scores   |
| `sklearn.compose`         | `ColumnTransformer`, `make_column_transformer()`          |
| `sklearn.inspection`      | Model explainability (e.g., PDP)                          |
| `sklearn.utils`           | Internal helper functions                                 |

---

## 🔁 3. **Standard Estimator Lifecycle**

Every estimator follows this pattern:

```python
model = Estimator(**params)   # 1. Create
model.fit(X_train, y_train)   # 2. Train
y_pred = model.predict(X_test)  # 3. Predict
score = model.score(X_test, y_test)  # 4. Evaluate
```

---

## 🔄 4. **Pipelines & Transformers**

Pipeline = transformer(s) + estimator
All fit/transform steps are chained:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('clf', LogisticRegression())
])
```

---

## 🧪 5. **Model Selection Tools**

| Tool                 | Purpose                      |
| -------------------- | ---------------------------- |
| `train_test_split`   | Split into training/test     |
| `cross_val_score`    | Cross-validation scoring     |
| `GridSearchCV`       | Exhaustive hyperparam search |
| `RandomizedSearchCV` | Random search                |
| `TimeSeriesSplit`    | CV for time-series data      |

---

## 🧠 Summary Structure

```
sklearn/
├── base/                 # Base classes like BaseEstimator
├── linear_model/         # Linear and logistic regression
├── tree/                 # Decision trees
├── ensemble/             # RandomForest, Boosting, etc.
├── svm/                  # SVMs
├── neighbors/            # KNN
├── cluster/              # Clustering
├── decomposition/        # PCA, SVD
├── preprocessing/        # Scaling, encoding
├── model_selection/      # Cross-validation, tuning
├── pipeline/             # Pipeline support
├── compose/              # ColumnTransformer
├── metrics/              # Evaluation metrics
├── inspection/           # PDP, SHAP support
└── utils/                # Internal tools
```

