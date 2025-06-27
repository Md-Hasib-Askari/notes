# Statistics and Probability

## Probability Fundamentals

### Basic Concepts
- Sample space and events
- Probability axioms
- Conditional probability
- Independence
- Random variables

### Probability Distributions
- Discrete distributions
- Continuous distributions
- Joint distributions
- Marginal distributions

### Common Distributions
- Uniform distribution
- Normal (Gaussian) distribution
- Binomial distribution
- Poisson distribution
- Exponential distribution

## Descriptive Statistics

### Measures of Central Tendency
- Mean (arithmetic, geometric, harmonic)
- Median
- Mode
- When to use each measure

### Measures of Variability
- Range
- Variance
- Standard deviation
- Interquartile range
- Coefficient of variation

### Distribution Shape
- Skewness
- Kurtosis
- Normal distribution properties

## Inferential Statistics

### Sampling
- Population vs sample
- Sampling methods
- Sampling distributions
- Central Limit Theorem

### Confidence Intervals
- Construction and interpretation
- Margin of error
- Confidence levels

### Hypothesis Testing
- Null and alternative hypotheses
- Type I and Type II errors
- p-values
- Statistical significance

## Applications in Machine Learning

### Data Analysis
- Exploratory data analysis
- Distribution fitting
- Outlier detection
- Data quality assessment

### Model Evaluation
- Cross-validation
- Bootstrap sampling
- Statistical significance testing
- Confidence intervals for metrics

### Probabilistic Models
- Naive Bayes
- Gaussian Mixture Models
- Bayesian networks
- Hidden Markov Models

## Key Statistical Concepts for ML

### Bias and Variance
- Bias-variance tradeoff
- Underfitting and overfitting
- Model complexity

### Maximum Likelihood Estimation
- Likelihood functions
- Log-likelihood
- Parameter estimation

### Information Theory
- Entropy
- Cross-entropy
- Mutual information
- KL divergence

## Practical Examples

### Example 1: Data Distribution Analysis
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate sample data
data = np.random.normal(50, 15, 1000)

# Descriptive statistics
mean = np.mean(data)
std = np.std(data)
median = np.median(data)

# Test for normality
statistic, p_value = stats.shapiro(data)
```

### Example 2: Confidence Intervals
```python
from scipy.stats import t

def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    margin_of_error = std_err * t.ppf((1 + confidence) / 2, n - 1)
    return mean - margin_of_error, mean + margin_of_error
```

### Example 3: Hypothesis Testing
```python
# t-test for comparing means
def t_test(sample1, sample2):
    statistic, p_value = stats.ttest_ind(sample1, sample2)
    return statistic, p_value

# Chi-square test for independence
def chi_square_test(contingency_table):
    statistic, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    return statistic, p_value
```

## Common Statistical Tests

### Parametric Tests
- t-test (one-sample, two-sample)
- ANOVA
- Linear regression F-test

### Non-parametric Tests
- Mann-Whitney U test
- Wilcoxon signed-rank test
- Kruskal-Wallis test
- Chi-square test

## Learning Objectives
- [ ] Understand probability distributions
- [ ] Calculate descriptive statistics
- [ ] Perform hypothesis testing
- [ ] Apply statistical concepts to ML
- [ ] Interpret statistical results

## Common Pitfalls
- Correlation vs causation
- Multiple testing problem
- Assumption violations
- Sample size considerations
- Interpretation of p-values

## Resources
- "Think Stats" by Allen Downey
- Khan Academy Statistics
- SciPy documentation
- Statistical inference textbooks
