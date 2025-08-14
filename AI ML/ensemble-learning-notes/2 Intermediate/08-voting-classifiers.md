# 08 — Voting Classifiers

Goal: Know when to use hard vs soft voting and how to combine heterogeneous models quickly.

## Hard vs Soft
- Hard voting: majority class among base classifiers’ predicted labels.
- Soft voting: average predicted probabilities and pick argmax; often better calibrated and higher accuracy when models output reliable probabilities.

## When to use
- Quick baseline ensemble over diverse models (linear, tree, kNN, SVM).
- When you have several decent-but-different models and want a fast boost without stacking complexity.

## Example — Hard vs Soft
```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

X, y = load_wine(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression(max_iter=2000, random_state=42)
knn = KNeighborsClassifier(n_neighbors=7)
rf = RandomForestClassifier(n_estimators=400, random_state=42)

hard = VotingClassifier(estimators=[('lr', lr), ('knn', knn), ('rf', rf)], voting='hard')
soft = VotingClassifier(estimators=[('lr', lr), ('knn', knn), ('rf', rf)], voting='soft')

hard.fit(X_tr, y_tr)
soft.fit(X_tr, y_tr)
print('Hard acc:', hard.score(X_te, y_te))
print('Soft acc:', soft.score(X_te, y_te))
```

## Tips
- Ensure base models used in soft voting expose predict_proba; otherwise use decision_function with care.
- Calibrate probabilities if needed (CalibratedClassifierCV) before soft voting.
- Don’t overstuff with many weak models; prefer a few strong diverse ones.

## Exercise
- Compare hard vs soft voting across different base model sets and report accuracy/AUC.
