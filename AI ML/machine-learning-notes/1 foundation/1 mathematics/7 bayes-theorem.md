# Bayes' Theorem

## Fundamental Concepts

### Bayes' Theorem Formula
```
P(A|B) = P(B|A) × P(A) / P(B)
```

Where:
- P(A|B): Posterior probability of A given B
- P(B|A): Likelihood of B given A
- P(A): Prior probability of A
- P(B): Marginal probability of B

### Extended Form
```
P(A|B) = P(B|A) × P(A) / [P(B|A) × P(A) + P(B|A') × P(A')]
```

## Key Components

### Prior Probability P(A)
- Initial belief about hypothesis
- Based on previous knowledge
- Before observing new evidence

### Likelihood P(B|A)
- Probability of evidence given hypothesis
- How well hypothesis explains evidence
- Model fitting component

### Posterior Probability P(A|B)
- Updated belief after observing evidence
- Combines prior and likelihood
- Final result of Bayesian inference

### Marginal Probability P(B)
- Total probability of evidence
- Normalizing constant
- Ensures probabilities sum to 1

## Applications in Machine Learning

### Naive Bayes Classifier
- Assumes feature independence
- Text classification
- Spam detection
- Medical diagnosis

### Bayesian Networks
- Probabilistic graphical models
- Causal relationships
- Uncertainty reasoning

### Bayesian Optimization
- Hyperparameter tuning
- Acquisition functions
- Gaussian processes

### A/B Testing
- Bayesian inference for experiments
- Credible intervals
- Decision making under uncertainty

## Practical Examples

### Example 1: Medical Diagnosis
```python
# Disease diagnosis example
def bayes_diagnosis(prior_disease, sensitivity, specificity, test_positive):
    """
    prior_disease: P(Disease)
    sensitivity: P(Test+|Disease) - true positive rate
    specificity: P(Test-|No Disease) - true negative rate
    test_positive: observed test result
    """
    if test_positive:
        likelihood = sensitivity
        marginal = sensitivity * prior_disease + (1 - specificity) * (1 - prior_disease)
    else:
        likelihood = 1 - sensitivity
        marginal = (1 - sensitivity) * prior_disease + specificity * (1 - prior_disease)
    
    posterior = likelihood * prior_disease / marginal
    return posterior
```

### Example 2: Naive Bayes Implementation
```python
import numpy as np
from collections import defaultdict

class NaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.feature_probs = defaultdict(dict)
    
    def fit(self, X, y):
        n_samples = len(y)
        classes = np.unique(y)
        
        # Calculate class priors
        for cls in classes:
            self.class_priors[cls] = np.sum(y == cls) / n_samples
        
        # Calculate feature probabilities
        for cls in classes:
            class_samples = X[y == cls]
            for feature_idx in range(X.shape[1]):
                feature_values = class_samples[:, feature_idx]
                self.feature_probs[cls][feature_idx] = {
                    'mean': np.mean(feature_values),
                    'std': np.std(feature_values)
                }
    
    def predict_proba(self, X):
        predictions = []
        for sample in X:
            class_scores = {}
            for cls in self.class_priors:
                # Start with prior
                score = np.log(self.class_priors[cls])
                
                # Add likelihood for each feature
                for feature_idx, feature_value in enumerate(sample):
                    mean = self.feature_probs[cls][feature_idx]['mean']
                    std = self.feature_probs[cls][feature_idx]['std']
                    
                    # Gaussian likelihood
                    likelihood = (1 / (std * np.sqrt(2 * np.pi))) * \
                               np.exp(-0.5 * ((feature_value - mean) / std) ** 2)
                    score += np.log(likelihood)
                
                class_scores[cls] = score
            
            predictions.append(class_scores)
        return predictions
```

### Example 3: Bayesian Update
```python
def bayesian_update(prior, likelihood_data, likelihood_no_data):
    """
    Update belief based on new evidence
    """
    # Calculate marginal probability
    marginal = likelihood_data * prior + likelihood_no_data * (1 - prior)
    
    # Calculate posterior
    posterior = likelihood_data * prior / marginal
    
    return posterior

# Example: Coin flip experiment
prior_fair = 0.5  # Prior belief coin is fair
heads_observed = 7
total_flips = 10

# Likelihood of observing this many heads
from scipy.stats import binom
likelihood_fair = binom.pmf(heads_observed, total_flips, 0.5)
likelihood_biased = binom.pmf(heads_observed, total_flips, 0.7)

posterior_fair = bayesian_update(prior_fair, likelihood_fair, likelihood_biased)
```

## Bayesian vs Frequentist

### Bayesian Approach
- Probability as degree of belief
- Incorporates prior knowledge
- Updates beliefs with evidence
- Provides uncertainty quantification

### Frequentist Approach
- Probability as long-run frequency
- No prior assumptions
- Fixed parameters
- Confidence intervals

## Common Applications

### Text Classification
- Spam filtering
- Sentiment analysis
- Document categorization
- Language detection

### Recommendation Systems
- User preference modeling
- Item similarity
- Collaborative filtering

### Anomaly Detection
- Fraud detection
- Network intrusion
- Quality control

## Learning Objectives
- [ ] Understand Bayes' theorem components
- [ ] Apply to classification problems
- [ ] Implement Naive Bayes
- [ ] Compare Bayesian vs frequentist approaches
- [ ] Use in real-world scenarios

## Key Insights
- Prior knowledge matters
- Evidence updates beliefs
- Independence assumptions simplify computation
- Uncertainty quantification is valuable
- Computational efficiency trade-offs

## Resources
- "Think Bayes" by Allen Downey
- Scikit-learn Naive Bayes
- PyMC3 for Bayesian modeling
- Bayesian statistics textbooks
