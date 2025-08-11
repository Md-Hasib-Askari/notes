# Statistical Analysis with SciPy

Statistical analysis is fundamental in microbiology for comparing treatments, analyzing experimental results, and drawing meaningful conclusions from data.

## Introduction to Statistical Testing

### Basic Concepts and Setup

```python
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, ttest_rel, mannwhitneyu, wilcoxon
from scipy.stats import f_oneway, kruskal, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
```

### Sample Data Generation

```python
# Generate sample microbiology data
def generate_sample_data():
    """Generate realistic microbiology experimental data"""
    
    # Growth rates under different conditions
    control_growth = np.random.normal(0.8, 0.15, 30)  # Control condition
    treatment_growth = np.random.normal(0.6, 0.12, 30)  # Treatment condition
    
    # Colony counts from different media
    lb_medium = np.random.poisson(150, 25)
    minimal_medium = np.random.poisson(85, 25)
    rich_medium = np.random.poisson(220, 25)
    
    # Antibiotic susceptibility zones (mm)
    antibiotic_zones = {
        'control': np.random.normal(8, 2, 20),
        'low_dose': np.random.normal(15, 3, 20),
        'medium_dose': np.random.normal(22, 4, 20),
        'high_dose': np.random.normal(28, 3, 20)
    }
    
    return {
        'growth_rates': {'control': control_growth, 'treatment': treatment_growth},
        'colony_counts': {'LB': lb_medium, 'Minimal': minimal_medium, 'Rich': rich_medium},
        'antibiotic_zones': antibiotic_zones
    }

# Generate data for analysis
data = generate_sample_data()
print("Sample data generated for statistical analysis")
```

## Hypothesis Testing and P-values

### Two-Sample T-Tests

```python
def perform_two_sample_ttest(group1, group2, group_names=None, alpha=0.05):
    """Perform two-sample t-test with interpretation"""
    
    if group_names is None:
        group_names = ['Group 1', 'Group 2']
    
    # Check normality assumption
    shapiro_p1 = stats.shapiro(group1)[1]
    shapiro_p2 = stats.shapiro(group2)[1]
    
    print(f"Normality tests (Shapiro-Wilk):")
    print(f"  {group_names[0]}: p = {shapiro_p1:.4f}")
    print(f"  {group_names[1]}: p = {shapiro_p2:.4f}")
    
    # Check equal variances assumption
    levene_stat, levene_p = stats.levene(group1, group2)
    print(f"Equal variances test (Levene): p = {levene_p:.4f}")
    
    # Perform appropriate t-test
    if levene_p > alpha:
        # Equal variances assumed
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=True)
        test_type = "Student's t-test (equal variances)"
    else:
        # Unequal variances (Welch's t-test)
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        test_type = "Welch's t-test (unequal variances)"
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(((len(group1)-1)*np.var(group1, ddof=1) + 
                         (len(group2)-1)*np.var(group2, ddof=1)) / 
                        (len(group1) + len(group2) - 2))
    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    # Results summary
    print(f"\n{test_type} Results:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Cohen's d: {cohens_d:.4f}")
    
    # Interpretation
    if p_value < alpha:
        print(f"  Result: Significant difference (p < {alpha})")
    else:
        print(f"  Result: No significant difference (p ≥ {alpha})")
    
    # Effect size interpretation
    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"
    
    print(f"  Effect size: {effect_size}")
    
    return t_stat, p_value, cohens_d

# Example: Compare growth rates between control and treatment
print("Comparing growth rates: Control vs Treatment")
t_stat, p_val, effect = perform_two_sample_ttest(
    data['growth_rates']['control'], 
    data['growth_rates']['treatment'],
    ['Control', 'Treatment']
)
```

### Paired T-Tests

```python
def perform_paired_ttest(before, after, alpha=0.05):
    """Perform paired t-test for before/after comparisons"""
    
    # Calculate differences
    differences = after - before
    
    # Check normality of differences
    shapiro_stat, shapiro_p = stats.shapiro(differences)
    print(f"Normality test for differences (Shapiro-Wilk): p = {shapiro_p:.4f}")
    
    if shapiro_p > alpha:
        # Use parametric test
        t_stat, p_value = stats.ttest_rel(before, after)
        test_name = "Paired t-test"
    else:
        # Use non-parametric alternative
        w_stat, p_value = stats.wilcoxon(before, after)
        test_name = "Wilcoxon signed-rank test"
        t_stat = w_stat
    
    # Calculate effect size
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    cohens_d = mean_diff / std_diff
    
    print(f"\n{test_name} Results:")
    print(f"  Test statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Mean difference: {mean_diff:.4f}")
    print(f"  Effect size (Cohen's d): {cohens_d:.4f}")
    
    if p_value < alpha:
        print(f"  Result: Significant change (p < {alpha})")
    else:
        print(f"  Result: No significant change (p ≥ {alpha})")
    
    return t_stat, p_value, cohens_d

# Example: Before/after antibiotic treatment
before_treatment = np.random.normal(100, 20, 15)  # Colony counts before
after_treatment = before_treatment * np.random.uniform(0.3, 0.8, 15)  # After treatment

print("Paired analysis: Colony counts before vs after antibiotic treatment")
perform_paired_ttest(before_treatment, after_treatment)
```

### Non-parametric Tests

```python
def perform_mannwhitney_test(group1, group2, group_names=None, alpha=0.05):
    """Perform Mann-Whitney U test (non-parametric alternative to t-test)"""
    
    if group_names is None:
        group_names = ['Group 1', 'Group 2']
    
    # Perform Mann-Whitney U test
    u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    
    # Calculate effect size (rank-biserial correlation)
    n1, n2 = len(group1), len(group2)
    r = 1 - (2 * u_stat) / (n1 * n2)  # Effect size for Mann-Whitney
    
    print(f"Mann-Whitney U Test Results:")
    print(f"  U-statistic: {u_stat:.2f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Effect size (r): {r:.4f}")
    
    if p_value < alpha:
        print(f"  Result: Significant difference (p < {alpha})")
    else:
        print(f"  Result: No significant difference (p ≥ {alpha})")
    
    return u_stat, p_value, r

# Example: Compare non-normally distributed data
skewed_data1 = np.random.exponential(2, 20)
skewed_data2 = np.random.exponential(3, 20)

print("Non-parametric comparison of skewed data:")
perform_mannwhitney_test(skewed_data1, skewed_data2, ['Condition A', 'Condition B'])
```

## ANOVA and Multiple Group Comparisons

### One-Way ANOVA

```python
def perform_one_way_anova(groups, group_names=None, alpha=0.05):
    """Perform one-way ANOVA with post-hoc tests"""
    
    if group_names is None:
        group_names = [f'Group {i+1}' for i in range(len(groups))]
    
    # Check assumptions
    print("ANOVA Assumptions Check:")
    
    # 1. Normality check for each group
    for i, group in enumerate(groups):
        shapiro_stat, shapiro_p = stats.shapiro(group)
        print(f"  Normality ({group_names[i]}): p = {shapiro_p:.4f}")
    
    # 2. Equal variances (Levene's test)
    levene_stat, levene_p = stats.levene(*groups)
    print(f"  Equal variances (Levene): p = {levene_p:.4f}")
    
    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Calculate effect size (eta-squared)
    # SS_between / SS_total
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    
    ss_between = sum([len(group) * (np.mean(group) - grand_mean)**2 for group in groups])
    ss_total = sum([(x - grand_mean)**2 for x in all_data])
    eta_squared = ss_between / ss_total
    
    print(f"\nOne-Way ANOVA Results:")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  eta-squared: {eta_squared:.4f}")
    
    if p_value < alpha:
        print(f"  Result: Significant difference between groups (p < {alpha})")
        
        # Perform post-hoc tests (Tukey HSD)
        print("\nPost-hoc analysis needed - significant ANOVA result")
        perform_posthoc_tests(groups, group_names, alpha)
    else:
        print(f"  Result: No significant difference between groups (p ≥ {alpha})")
    
    return f_stat, p_value, eta_squared

def perform_posthoc_tests(groups, group_names, alpha=0.05):
    """Perform pairwise comparisons with Bonferroni correction"""
    
    n_groups = len(groups)
    n_comparisons = n_groups * (n_groups - 1) // 2
    bonferroni_alpha = alpha / n_comparisons
    
    print(f"Pairwise comparisons (Bonferroni corrected α = {bonferroni_alpha:.4f}):")
    
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            t_stat, p_value = stats.ttest_ind(groups[i], groups[j])
            
            if p_value < bonferroni_alpha:
                significance = "***"
            elif p_value < alpha:
                significance = "*"
            else:
                significance = "ns"
            
            print(f"  {group_names[i]} vs {group_names[j]}: p = {p_value:.4f} {significance}")

# Example: Compare colony counts across different media
media_groups = [data['colony_counts']['LB'], 
                data['colony_counts']['Minimal'], 
                data['colony_counts']['Rich']]
media_names = ['LB Medium', 'Minimal Medium', 'Rich Medium']

print("Comparing colony counts across different growth media:")
perform_one_way_anova(media_groups, media_names)
```

### Kruskal-Wallis Test (Non-parametric ANOVA)

```python
def perform_kruskal_wallis(groups, group_names=None, alpha=0.05):
    """Perform Kruskal-Wallis test (non-parametric ANOVA)"""
    
    if group_names is None:
        group_names = [f'Group {i+1}' for i in range(len(groups))]
    
    # Perform Kruskal-Wallis test
    h_stat, p_value = stats.kruskal(*groups)
    
    # Calculate effect size (epsilon-squared)
    all_data = np.concatenate(groups)
    n_total = len(all_data)
    epsilon_squared = (h_stat - len(groups) + 1) / (n_total - len(groups))
    
    print(f"Kruskal-Wallis Test Results:")
    print(f"  H-statistic: {h_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Effect size (ε²): {epsilon_squared:.4f}")
    
    if p_value < alpha:
        print(f"  Result: Significant difference between groups (p < {alpha})")
    else:
        print(f"  Result: No significant difference between groups (p ≥ {alpha})")
    
    return h_stat, p_value, epsilon_squared

# Example with non-normally distributed data
print("\nNon-parametric comparison of antibiotic zones:")
zone_groups = list(data['antibiotic_zones'].values())
zone_names = list(data['antibiotic_zones'].keys())
perform_kruskal_wallis(zone_groups, zone_names)
```

## Correlation and Regression Analysis

### Correlation Analysis

```python
def analyze_correlations(x, y, labels=None, alpha=0.05):
    """Comprehensive correlation analysis"""
    
    if labels is None:
        labels = ['Variable X', 'Variable Y']
    
    # Pearson correlation (parametric)
    pearson_r, pearson_p = stats.pearsonr(x, y)
    
    # Spearman correlation (non-parametric)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    
    # Kendall's tau (another non-parametric option)
    kendall_tau, kendall_p = stats.kendalltau(x, y)
    
    print(f"Correlation Analysis: {labels[0]} vs {labels[1]}")
    print(f"  Pearson r: {pearson_r:.4f} (p = {pearson_p:.4f})")
    print(f"  Spearman ρ: {spearman_r:.4f} (p = {spearman_p:.4f})")
    print(f"  Kendall τ: {kendall_tau:.4f} (p = {kendall_p:.4f})")
    
    # Interpretation
    if abs(pearson_r) < 0.1:
        strength = "negligible"
    elif abs(pearson_r) < 0.3:
        strength = "weak"
    elif abs(pearson_r) < 0.5:
        strength = "moderate"
    elif abs(pearson_r) < 0.7:
        strength = "strong"
    else:
        strength = "very strong"
    
    direction = "positive" if pearson_r > 0 else "negative"
    print(f"  Relationship: {strength} {direction} correlation")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, alpha=0.6)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(f'Scatter Plot\nr = {pearson_r:.3f}')
    
    # Add regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--", alpha=0.8)
    
    # Residual plot
    plt.subplot(1, 2, 2)
    residuals = y - p(x)
    plt.scatter(p(x), residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    plt.tight_layout()
    plt.show()
    
    return pearson_r, pearson_p, spearman_r, spearman_p

# Example: Temperature vs growth rate correlation
temperature = np.random.normal(37, 5, 50)
growth_rate = 0.02 * temperature + np.random.normal(0, 0.1, 50)

analyze_correlations(temperature, growth_rate, ['Temperature (°C)', 'Growth Rate (h⁻¹)'])
```

### Multiple Testing Corrections

```python
def multiple_testing_correction(p_values, method='bonferroni', alpha=0.05):
    """Apply multiple testing corrections"""
    
    p_values = np.array(p_values)
    n_tests = len(p_values)
    
    if method == 'bonferroni':
        corrected_alpha = alpha / n_tests
        significant = p_values < corrected_alpha
        corrected_p = p_values * n_tests
        corrected_p = np.minimum(corrected_p, 1.0)  # Cap at 1.0
    
    elif method == 'benjamini_hochberg':
        # Benjamini-Hochberg (FDR) procedure
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        m = len(p_values)
        corrected_p = np.zeros_like(p_values)
        
        for i in range(m):
            corrected_p[sorted_indices[i]] = min(1.0, sorted_p[i] * m / (i + 1))
        
        significant = corrected_p < alpha
    
    print(f"Multiple Testing Correction ({method}):")
    print(f"  Number of tests: {n_tests}")
    print(f"  Original α: {alpha}")
    
    if method == 'bonferroni':
        print(f"  Corrected α: {corrected_alpha:.6f}")
    
    print(f"  Significant tests: {np.sum(significant)}/{n_tests}")
    
    return corrected_p, significant

# Example: Multiple pairwise comparisons
p_values_example = [0.001, 0.02, 0.045, 0.08, 0.12, 0.03]
test_names = ['Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5', 'Test 6']

print("Example multiple testing scenario:")
for i, (test, p) in enumerate(zip(test_names, p_values_example)):
    print(f"  {test}: p = {p:.3f}")

# Apply corrections
corrected_p_bonf, sig_bonf = multiple_testing_correction(p_values_example, 'bonferroni')
corrected_p_bh, sig_bh = multiple_testing_correction(p_values_example, 'benjamini_hochberg')

print("\nResults after correction:")
for i, test in enumerate(test_names):
    print(f"  {test}: Bonferroni = {'Sig' if sig_bonf[i] else 'NS'}, BH = {'Sig' if sig_bh[i] else 'NS'}")
```

## Best Practices for Statistical Analysis

1. **Check assumptions** before applying statistical tests
2. **Use appropriate sample sizes** for adequate power
3. **Choose between parametric and non-parametric** tests based on data distribution
4. **Apply multiple testing corrections** when performing multiple comparisons
5. **Report effect sizes** along with p-values
6. **Visualize your data** before and after analysis
7. **Document your statistical approach** and rationale
8. **Consider biological significance** alongside statistical significance
