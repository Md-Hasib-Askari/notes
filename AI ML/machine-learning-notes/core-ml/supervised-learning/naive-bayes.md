# Naive Bayes

## Overview
Naive Bayes is a probabilistic classifier based on Bayes' theorem with the "naive" assumption of conditional independence between features. Despite this strong assumption, it performs surprisingly well in many real-world applications.

## Mathematical Foundation

### Bayes' Theorem
```
P(class|features) = P(features|class) × P(class) / P(features)
```

- **P(class|features)**: Posterior probability
- **P(features|class)**: Likelihood
- **P(class)**: Prior probability
- **P(features)**: Evidence

### Naive Independence Assumption
Assumes all features are conditionally independent given the class:
```
P(x₁, x₂, ..., xₙ|y) = P(x₁|y) × P(x₂|y) × ... × P(xₙ|y)
```

## Types of Naive Bayes

### 1. Gaussian Naive Bayes
- For continuous features
- Assumes normal distribution
- Best for: Numerical data

### 2. Multinomial Naive Bayes
- For discrete count data
- Common in text classification
- Best for: Word counts, TF-IDF

### 3. Bernoulli Naive Bayes
- For binary/boolean features
- Best for: Document classification (word presence/absence)

## Quick Implementation

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Gaussian Naive Bayes (numerical data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# Multinomial Naive Bayes (text data)
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X_text_counts = vectorizer.fit_transform(text_data)

mnb = MultinomialNB()
mnb.fit(X_text_counts, y_labels)
```

## Key Advantages
- **Fast**: Linear time complexity
- **Simple**: Easy to implement and understand
- **Memory efficient**: Requires small training dataset
- **Handles multiple classes**: Naturally multiclass
- **Good baseline**: Often surprisingly effective

## Limitations
- **Independence assumption**: Rarely true in practice
- **Zero probability**: Smoothing needed for unseen features
- **Categorical inputs**: Poor estimator for numerical values
- **Feature correlation**: Ignores feature relationships

## Common Applications
- **Email spam filtering**
- **Text classification**
- **Sentiment analysis**
- **Medical diagnosis**
- **Real-time predictions**

## Best Practices
1. **Feature preprocessing**: Scale/normalize for Gaussian NB
2. **Laplace smoothing**: Handle zero probabilities
3. **Feature selection**: Remove highly correlated features
4. **Cross-validation**: Validate performance
5. **Baseline model**: Start with Naive Bayes before complex models

## Performance Tips
- Works well with high-dimensional data
- Excellent for text classification tasks
- Consider ensemble methods to improve accuracy
- Use appropriate variant based on data type