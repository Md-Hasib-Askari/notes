# Hypothesis Testing

## Statistical Hypothesis Testing Framework

### Basic Concepts
- **Null Hypothesis (H₀)**: Statement of no effect or no difference
- **Alternative Hypothesis (H₁)**: Statement we want to test
- **Test Statistic**: Numerical summary of data
- **p-value**: Probability of observing data given H₀ is true
- **Significance Level (α)**: Threshold for rejecting H₀

### Types of Errors
- **Type I Error**: Rejecting true H₀ (false positive)
- **Type II Error**: Failing to reject false H₀ (false negative)
- **Power**: Probability of correctly rejecting false H₀

## Common Statistical Tests

### t-Tests
- **One-sample t-test**: Compare sample mean to known value
- **Two-sample t-test**: Compare means of two groups
- **Paired t-test**: Compare paired observations

### ANOVA (Analysis of Variance)
- **One-way ANOVA**: Compare means across multiple groups
- **Two-way ANOVA**: Test two factors simultaneously
- **Repeated measures ANOVA**: Within-subjects design

### Non-parametric Tests
- **Mann-Whitney U**: Non-parametric alternative to two-sample t-test
- **Wilcoxon signed-rank**: Non-parametric alternative to paired t-test
- **Kruskal-Wallis**: Non-parametric alternative to one-way ANOVA

### Proportion Tests
- **One-proportion z-test**: Test single proportion
- **Two-proportion z-test**: Compare two proportions
- **Chi-square goodness of fit**: Test distribution fit
- **Chi-square independence**: Test variable independence

## Applications in Machine Learning

### Model Performance Testing
- Comparing algorithm performance
- Statistical significance of improvements
- Cross-validation significance testing

### A/B Testing
- Conversion rate testing
- Feature effectiveness
- Treatment effect analysis

### Feature Selection
- Statistical significance of features
- Correlation testing
- Independence testing

### Data Quality
- Distribution testing
- Outlier detection
- Normality testing

## Practical Examples

### Example 1: Model Comparison
```python
import numpy as np
from scipy import stats

def compare_models(model1_scores, model2_scores, alpha=0.05):
    """
    Compare two models using paired t-test
    """
    # Paired t-test
    statistic, p_value = stats.ttest_rel(model1_scores, model2_scores)
    
    # Interpretation
    if p_value < alpha:
        winner = "Model 1" if np.mean(model1_scores) > np.mean(model2_scores) else "Model 2"
        result = f"Significant difference (p={p_value:.4f}). {winner} performs better."
    else:
        result = f"No significant difference (p={p_value:.4f})"
    
    return statistic, p_value, result

# Example usage
model1_accuracy = [0.85, 0.87, 0.84, 0.86, 0.88]
model2_accuracy = [0.83, 0.85, 0.82, 0.84, 0.86]

stat, p_val, interpretation = compare_models(model1_accuracy, model2_accuracy)
print(interpretation)
```

### Example 2: A/B Testing
```python
def ab_test_proportions(control_conversions, control_total, 
                       treatment_conversions, treatment_total, alpha=0.05):
    """
    A/B test for conversion rates
    """
    # Calculate proportions
    p1 = control_conversions / control_total
    p2 = treatment_conversions / treatment_total
    
    # Pooled proportion
    p_pool = (control_conversions + treatment_conversions) / (control_total + treatment_total)
    
    # Standard error
    se = np.sqrt(p_pool * (1 - p_pool) * (1/control_total + 1/treatment_total))
    
    # Test statistic
    z = (p2 - p1) / se
    
    # p-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    # Effect size
    effect_size = (p2 - p1) / p1 * 100  # Percentage change
    
    return {
        'control_rate': p1,
        'treatment_rate': p2,
        'effect_size_percent': effect_size,
        'z_statistic': z,
        'p_value': p_value,
        'significant': p_value < alpha
    }

# Example usage
result = ab_test_proportions(control_conversions=120, control_total=1000,
                           treatment_conversions=145, treatment_total=1000)
```

### Example 3: Feature Significance Testing
```python
from sklearn.feature_selection import chi2, f_classif

def test_feature_significance(X, y, feature_names, test_type='chi2'):
    """
    Test statistical significance of features
    """
    if test_type == 'chi2':
        # For categorical features
        scores, p_values = chi2(X, y)
    elif test_type == 'f_test':
        # For continuous features
        scores, p_values = f_classif(X, y)
    
    # Create results dataframe
    results = pd.DataFrame({
        'feature': feature_names,
        'score': scores,
        'p_value': p_values,
        'significant': p_values < 0.05
    })
    
    return results.sort_values('p_value')
```

## Multiple Testing Problem

### The Problem
- When testing multiple hypotheses simultaneously
- Increased chance of Type I errors
- Family-wise error rate inflation

### Corrections
- **Bonferroni Correction**: α/n for n tests
- **Holm Method**: Step-down procedure
- **False Discovery Rate (FDR)**: Benjamini-Hochberg procedure

```python
from statsmodels.stats.multitest import multipletests

def multiple_testing_correction(p_values, method='bonferroni', alpha=0.05):
    """
    Apply multiple testing correction
    """
    rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
        p_values, alpha=alpha, method=method
    )
    
    return {
        'rejected': rejected,
        'p_corrected': p_corrected,
        'significant_features': np.sum(rejected)
    }
```

## Power Analysis

### Sample Size Calculation
```python
from statsmodels.stats.power import ttest_power

def calculate_sample_size(effect_size, power=0.8, alpha=0.05):
    """
    Calculate required sample size for t-test
    """
    from statsmodels.stats.power import tt_solve_power
    
    sample_size = tt_solve_power(effect_size=effect_size, 
                                power=power, 
                                alpha=alpha, 
                                alternative='two-sided')
    return int(np.ceil(sample_size))

# Example: detect 20% improvement with 80% power
n_required = calculate_sample_size(effect_size=0.5)
print(f"Required sample size: {n_required}")
```

## Best Practices

### Before Testing
1. Define hypotheses clearly
2. Choose appropriate test
3. Check assumptions
4. Determine sample size

### During Testing
1. Use appropriate significance level
2. Consider multiple testing corrections
3. Check test assumptions
4. Calculate effect sizes

### After Testing
1. Interpret results correctly
2. Consider practical significance
3. Report confidence intervals
4. Validate findings

## Common Pitfalls
- p-hacking and data dredging
- Misinterpreting p-values
- Ignoring effect sizes
- Multiple comparisons
- Assumption violations

## Learning Objectives
- [ ] Understand hypothesis testing framework
- [ ] Choose appropriate statistical tests
- [ ] Interpret p-values correctly
- [ ] Apply multiple testing corrections
- [ ] Calculate power and sample sizes

## Resources
- "Statistics Done Wrong" by Alex Reinhart
- SciPy.stats documentation
- Statsmodels library
- "The Elements of Statistical Learning"
