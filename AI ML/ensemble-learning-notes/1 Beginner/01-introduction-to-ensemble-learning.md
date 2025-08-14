# 01 — Introduction to Ensemble Learning

Learn why and how combining multiple models often beats a single model, what types of ensembles exist, and when to use them.

## Learning objectives
- Explain why ensembles improve generalization (variance/bias reduction and error decorrelation).
- Recognize common ensemble families: bagging, boosting, stacking, voting.
- Know when to prefer each family and how to set up quick baselines.

## Mental model (10,000‑ft view)
- One model = one opinion. Many models = a committee. If members are skilled and make different mistakes, the committee is usually more reliable.
- Ensembles help most when base models are:
  - Reasonably strong (better than random/naive), and
  - Diverse (uncorrelated errors from different data samples, features, or algorithms).

## Core families at a glance
- Bagging (parallel):
  - Train multiple base learners independently on bootstrapped samples; aggregate predictions.
  - Reduces variance. Example: Random Forests.
- Boosting (sequential):
  - Train learners one after another; each focuses on previous errors.
  - Reduces bias. Examples: AdaBoost, Gradient Boosting.
- Stacking (meta-learning):
  - Train diverse base models; a meta-model learns how to combine their outputs.
  - Leverages complementary strengths.
- Voting (simple aggregation):
  - Majority vote (hard) or average probabilities (soft) across different models.

## Key ingredients
- Diversity: created via different samples (bootstraps), features (subsampling), model types (heterogeneous), or hyperparameters.
- Aggregation: majority vote, probability averaging, or meta-model.

## Quick baseline example (Voting)
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

clf1 = LogisticRegression(max_iter=1000, random_state=42)
clf2 = KNeighborsClassifier(n_neighbors=5)
clf3 = DecisionTreeClassifier(random_state=42)

voting = VotingClassifier(
    estimators=[('lr', clf1), ('knn', clf2), ('dt', clf3)],
    voting='hard'
)
voting.fit(X_tr, y_tr)
print('Voting accuracy:', voting.score(X_te, y_te))
```

## When to use which?
- Bagging/Random Forest:
  - Strong default for tabular data, robust to outliers, low tuning burden.
- Boosting (AdaBoost/GBM):
  - When you need high accuracy on structured data and can afford tuning; handles complex patterns.
- Stacking:
  - When you already have strong but different models; risk of leakage must be handled carefully.
- Voting:
  - Quick win to combine complementary models; minimal setup.

## Evaluation tips
- Always use cross-validation to estimate performance and avoid lucky splits.
- For bagging, consider Out-of-Bag (OOB) estimates as a free validation.
- Track not just accuracy but also calibration (probabilities) and class-imbalance metrics (F1, AUC, PR-AUC) when relevant.

## Common pitfalls
- Leakage in stacking (using true labels from the test fold to train the meta-model).
- Overfitting with too-complex base learners or meta-models.
- Ensembles can be slow or heavy; measure latency and memory for production.

## Further reading
- “Ensemble Methods” in Hands-On Machine Learning with Scikit-Learn (for concepts and practice)
- StatQuest: Ensemble Learning (intuitive videos)
