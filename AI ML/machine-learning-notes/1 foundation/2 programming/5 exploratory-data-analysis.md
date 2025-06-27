# Exploratory Data Analysis (EDA)

## Overview
Exploratory Data Analysis (EDA) is the critical first step in any data science project. It involves examining datasets to summarize their main characteristics, often with visual methods, before applying formal modeling techniques.

## The EDA Process

### 1. Initial Data Inspection
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load and inspect data
df = pd.read_csv('dataset.csv')

# Basic information
print(f"Dataset shape: {df.shape}")
print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")

# Data types and missing values
print("\nData Info:")
df.info()

# First few rows
print("\nFirst 5 rows:")
df.head()

# Statistical summary
print("\nStatistical Summary:")
df.describe(include='all')
```

### 2. Data Quality Assessment
```python
# Missing values analysis
def analyze_missing_data(df):
    missing_data = df.isnull().sum()
    missing_percent = 100 * missing_data / len(df)
    
    missing_table = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent
    })
    
    return missing_table[missing_table['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

missing_analysis = analyze_missing_data(df)
print(missing_analysis)

# Duplicate records
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")

# Data types verification
print("\nData Types:")
for col in df.columns:
    print(f"{col}: {df[col].dtype} - {df[col].nunique()} unique values")
```

## Descriptive Statistics

### 1. Central Tendency Measures
```python
# Measures of central tendency
def central_tendency_analysis(df, numeric_cols):
    stats = pd.DataFrame()
    
    for col in numeric_cols:
        stats[col] = {
            'Mean': df[col].mean(),
            'Median': df[col].median(),
            'Mode': df[col].mode().iloc[0] if not df[col].mode().empty else np.nan,
            'Geometric Mean': np.exp(np.log(df[col][df[col] > 0]).mean()) if (df[col] > 0).any() else np.nan
        }
    
    return stats.T

# Example usage
numeric_columns = df.select_dtypes(include=[np.number]).columns
central_stats = central_tendency_analysis(df, numeric_columns)
print(central_stats)
```

### 2. Variability Measures
```python
# Measures of dispersion
def variability_analysis(df, numeric_cols):
    stats = pd.DataFrame()
    
    for col in numeric_cols:
        stats[col] = {
            'Range': df[col].max() - df[col].min(),
            'Variance': df[col].var(),
            'Std Deviation': df[col].std(),
            'IQR': df[col].quantile(0.75) - df[col].quantile(0.25),
            'CV': df[col].std() / df[col].mean() * 100,  # Coefficient of Variation
            'MAD': np.mean(np.abs(df[col] - df[col].mean()))  # Mean Absolute Deviation
        }
    
    return stats.T

variability_stats = variability_analysis(df, numeric_columns)
print(variability_stats)
```

### 3. Distribution Analysis
```python
# Distribution shape analysis
def distribution_analysis(df, numeric_cols):
    from scipy import stats
    
    dist_stats = pd.DataFrame()
    
    for col in numeric_cols:
        data = df[col].dropna()
        
        dist_stats[col] = {
            'Skewness': stats.skew(data),
            'Kurtosis': stats.kurtosis(data),
            'Shapiro-Wilk p-value': stats.shapiro(data.sample(min(5000, len(data))))[1],
            'Min': data.min(),
            'Q1': data.quantile(0.25),
            'Q3': data.quantile(0.75),
            'Max': data.max()
        }
    
    return dist_stats.T

# Interpretation of skewness and kurtosis
def interpret_distribution(skewness, kurtosis):
    skew_interp = "symmetric" if abs(skewness) < 0.5 else ("moderately skewed" if abs(skewness) < 1 else "highly skewed")
    kurt_interp = "normal" if abs(kurtosis) < 0.5 else ("light-tailed" if kurtosis < 0 else "heavy-tailed")
    return f"{skew_interp}, {kurt_interp}"

distribution_stats = distribution_analysis(df, numeric_columns)
print(distribution_stats)
```

## Data Visualization

### 1. Univariate Analysis
```python
# Univariate plots for numerical variables
def plot_univariate_numerical(df, columns, figsize=(15, 10)):
    n_cols = len(columns)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, col in enumerate(columns):
        # Histogram with KDE
        axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, density=True)
        df[col].plot.kde(ax=axes[i], secondary_y=False)
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Density')
    
    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

# Box plots for outlier detection
def plot_boxplots(df, columns, figsize=(15, 8)):
    plt.figure(figsize=figsize)
    df[columns].boxplot()
    plt.title('Box Plots for Outlier Detection')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Categorical variable analysis
def plot_categorical_variables(df, cat_columns, figsize=(15, 10)):
    n_cols = len(cat_columns)
    n_rows = (n_cols + 1) // 2
    
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, col in enumerate(cat_columns):
        value_counts = df[col].value_counts()
        
        # Bar plot
        value_counts.plot(kind='bar', ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')
        axes[i].tick_params(axis='x', rotation=45)
    
    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

# Usage examples
categorical_columns = df.select_dtypes(include=['object']).columns
plot_univariate_numerical(df, numeric_columns[:6])
plot_categorical_variables(df, categorical_columns[:4])
```

### 2. Bivariate Analysis
```python
# Correlation analysis
def correlation_analysis(df, method='pearson'):
    # Calculate correlation matrix
    corr_matrix = df.select_dtypes(include=[np.number]).corr(method=method)
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                center=0, square=True, linewidths=0.5)
    plt.title(f'{method.capitalize()} Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Find strong correlations
    strong_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:  # Threshold for strong correlation
                strong_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
    
    return corr_matrix, strong_corr

# Scatter plots for relationships
def plot_scatter_relationships(df, target_col, feature_cols, figsize=(15, 10)):
    n_cols = len(feature_cols)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, col in enumerate(feature_cols):
        axes[i].scatter(df[col], df[target_col], alpha=0.6)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel(target_col)
        axes[i].set_title(f'{target_col} vs {col}')
        
        # Add trend line
        z = np.polyfit(df[col].dropna(), df[target_col].dropna(), 1)
        p = np.poly1d(z)
        axes[i].plot(df[col], p(df[col]), "r--", alpha=0.8)
    
    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

# Cross-tabulation for categorical variables
def categorical_relationships(df, cat_col1, cat_col2):
    crosstab = pd.crosstab(df[cat_col1], df[cat_col2])
    
    # Chi-square test
    from scipy.stats import chi2_contingency
    chi2, p_value, dof, expected = chi2_contingency(crosstab)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Cross-tabulation: {cat_col1} vs {cat_col2}')
    plt.show()
    
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Relationship is {'significant' if p_value < 0.05 else 'not significant'}")
    
    return crosstab, chi2, p_value

# Usage
corr_matrix, strong_correlations = correlation_analysis(df)
print("Strong correlations found:")
for col1, col2, corr in strong_correlations:
    print(f"{col1} - {col2}: {corr:.3f}")
```

### 3. Multivariate Analysis
```python
# Pair plots for multiple variables
def create_pairplot(df, columns, hue=None, figsize=(12, 12)):
    if len(columns) > 6:
        print("Warning: Too many columns for pair plot. Using first 6.")
        columns = columns[:6]
    
    subset_df = df[columns + ([hue] if hue else [])]
    g = sns.pairplot(subset_df, hue=hue, diag_kind='kde')
    g.fig.suptitle('Pair Plot Analysis', y=1.02)
    plt.show()

# Principal Component Analysis for dimensionality visualization
def pca_visualization(df, target_col=None):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Prepare data
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col and target_col in numeric_df.columns:
        features = numeric_df.drop(target_col, axis=1)
        target = numeric_df[target_col]
    else:
        features = numeric_df
        target = None
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply PCA
    pca = PCA()
    principal_components = pca.fit_transform(features_scaled)
    
    # Plot explained variance
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             pca.explained_variance_ratio_, 'bo-')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             np.cumsum(pca.explained_variance_ratio_), 'ro-')
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.axhline(y=0.8, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # 2D PCA plot
    if principal_components.shape[1] >= 2:
        plt.figure(figsize=(10, 8))
        if target is not None:
            scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], 
                                c=target, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter)
        else:
            plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.6)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('PCA - First Two Components')
        plt.show()
    
    return pca, principal_components
```

## Pattern Discovery

### 1. Outlier Detection
```python
# Multiple outlier detection methods
def detect_outliers(df, columns, methods=['iqr', 'zscore', 'isolation']):
    outliers_summary = {}
    
    for col in columns:
        data = df[col].dropna()
        outliers_summary[col] = {}
        
        # IQR method
        if 'iqr' in methods:
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
            outliers_summary[col]['IQR'] = len(iqr_outliers)
        
        # Z-score method
        if 'zscore' in methods:
            z_scores = np.abs((data - data.mean()) / data.std())
            zscore_outliers = data[z_scores > 3]
            outliers_summary[col]['Z-Score'] = len(zscore_outliers)
        
        # Isolation Forest
        if 'isolation' in methods:
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(data.values.reshape(-1, 1))
            iso_outliers = data[outlier_labels == -1]
            outliers_summary[col]['Isolation Forest'] = len(iso_outliers)
    
    return pd.DataFrame(outliers_summary).T

# Visualize outliers
def plot_outliers(df, column):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Box plot
    axes[0].boxplot(df[column].dropna())
    axes[0].set_title(f'Box Plot - {column}')
    axes[0].set_ylabel(column)
    
    # Histogram
    axes[1].hist(df[column].dropna(), bins=30, alpha=0.7)
    axes[1].set_title(f'Histogram - {column}')
    axes[1].set_xlabel(column)
    axes[1].set_ylabel('Frequency')
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(df[column].dropna(), dist="norm", plot=axes[2])
    axes[2].set_title(f'Q-Q Plot - {column}')
    
    plt.tight_layout()
    plt.show()

outlier_summary = detect_outliers(df, numeric_columns[:3])
print("Outliers detected by different methods:")
print(outlier_summary)
```

### 2. Trend Analysis
```python
# Time series trend analysis (if date column exists)
def analyze_trends(df, date_col, value_cols):
    if date_col not in df.columns:
        print(f"Date column '{date_col}' not found")
        return
    
    # Convert to datetime
    df[date_col] = pd.to_datetime(df[date_col])
    df_sorted = df.sort_values(date_col)
    
    fig, axes = plt.subplots(len(value_cols), 1, figsize=(12, 4*len(value_cols)))
    if len(value_cols) == 1:
        axes = [axes]
    
    for i, col in enumerate(value_cols):
        axes[i].plot(df_sorted[date_col], df_sorted[col])
        axes[i].set_title(f'Trend Analysis - {col}')
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel(col)
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add trend line
        x_numeric = pd.to_numeric(df_sorted[date_col])
        z = np.polyfit(x_numeric, df_sorted[col].dropna(), 1)
        p = np.poly1d(z)
        axes[i].plot(df_sorted[date_col], p(x_numeric), "r--", alpha=0.8, label='Trend')
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()

# Seasonal decomposition (if applicable)
def seasonal_decomposition(df, date_col, value_col, freq='M'):
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    df_ts = df.set_index(date_col)[value_col].dropna()
    decomposition = seasonal_decompose(df_ts, model='additive', period=12)
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    decomposition.observed.plot(ax=axes[0], title='Original')
    decomposition.trend.plot(ax=axes[1], title='Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
    decomposition.resid.plot(ax=axes[3], title='Residual')
    
    plt.tight_layout()
    plt.show()
```

### 3. Advanced Pattern Discovery
```python
# Clustering for pattern discovery
def discover_clusters(df, n_clusters=3):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # Prepare data
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    
    # Apply K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # Add cluster labels to original data
    result_df = numeric_df.copy()
    result_df['Cluster'] = cluster_labels
    
    # Visualize clusters (using first two components)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(scaled_data[:, 0], scaled_data[:, 1], 
                         c=cluster_labels, cmap='viridis')
    plt.xlabel(f'Feature 1: {numeric_df.columns[0]}')
    plt.ylabel(f'Feature 2: {numeric_df.columns[1]}')
    plt.title('K-Means Clustering Results')
    plt.colorbar(scatter)
    plt.show()
    
    # Cluster summary statistics
    cluster_summary = result_df.groupby('Cluster').agg(['mean', 'std', 'count'])
    return result_df, cluster_summary

# Association rules (for categorical data)
def find_associations(df, categorical_cols, min_support=0.1):
    """
    Simple association rule mining for categorical variables
    """
    associations = []
    
    for col1 in categorical_cols:
        for col2 in categorical_cols:
            if col1 != col2:
                crosstab = pd.crosstab(df[col1], df[col2])
                total_count = len(df)
                
                for val1 in crosstab.index:
                    for val2 in crosstab.columns:
                        # Calculate support, confidence
                        joint_count = crosstab.loc[val1, val2]
                        val1_count = df[df[col1] == val1].shape[0]
                        
                        support = joint_count / total_count
                        confidence = joint_count / val1_count if val1_count > 0 else 0
                        
                        if support >= min_support and confidence > 0.5:
                            associations.append({
                                'Antecedent': f'{col1}={val1}',
                                'Consequent': f'{col2}={val2}',
                                'Support': support,
                                'Confidence': confidence
                            })
    
    return pd.DataFrame(associations).sort_values('Confidence', ascending=False)
```

## Automated EDA Report
```python
def generate_eda_report(df, target_col=None):
    """
    Generate comprehensive EDA report
    """
    print("="*60)
    print("EXPLORATORY DATA ANALYSIS REPORT")
    print("="*60)
    
    # Basic information
    print(f"\n1. DATASET OVERVIEW")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    
    # Missing values
    print(f"\n2. MISSING VALUES")
    missing_data = analyze_missing_data(df)
    if not missing_data.empty:
        print(missing_data)
    else:
        print("No missing values found.")
    
    # Data types
    print(f"\n3. DATA TYPES")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    
    # Statistical summary
    print(f"\n4. STATISTICAL SUMMARY")
    print(df.describe())
    
    # Correlations
    if len(numeric_cols) > 1:
        print(f"\n5. CORRELATIONS")
        corr_matrix, strong_corr = correlation_analysis(df)
        if strong_corr:
            print("Strong correlations found:")
            for col1, col2, corr in strong_corr:
                print(f"  {col1} - {col2}: {corr:.3f}")
        else:
            print("No strong correlations found (|r| > 0.7)")
    
    # Outliers
    if len(numeric_cols) > 0:
        print(f"\n6. OUTLIERS")
        outlier_summary = detect_outliers(df, numeric_cols[:5])  # Limit to first 5 columns
        print(outlier_summary)
    
    print("\n" + "="*60)
    print("END OF REPORT")
    print("="*60)

# Usage
# generate_eda_report(df, target_col='target_column')
```

## Best Practices for EDA

### 1. Systematic Approach
- Start with data overview and quality assessment
- Progress from univariate to multivariate analysis
- Document findings and assumptions
- Iterate based on insights

### 2. Visualization Guidelines
- Choose appropriate plot types for data types
- Use consistent color schemes and styling
- Include clear titles and labels
- Avoid chart junk and unnecessary complexity

### 3. Statistical Considerations
- Check assumptions before applying statistical tests
- Consider multiple methods for robust conclusions
- Be aware of correlation vs. causation
- Account for multiple testing when appropriate

### 4. Documentation
- Record all transformations and decisions
- Save important visualizations
- Create reproducible analysis scripts
- Share insights with stakeholders

## Learning Objectives
- [ ] Conduct systematic data exploration
- [ ] Create effective visualizations for different data types
- [ ] Extract meaningful insights from data
- [ ] Identify data quality issues and patterns
- [ ] Apply statistical analysis techniques
- [ ] Use appropriate tools and libraries
- [ ] Generate comprehensive EDA reports
- [ ] Communicate findings effectively

## Tools and Libraries

**Core Libraries**: pandas, numpy, matplotlib, seaborn
**Statistical Analysis**: scipy, statsmodels
**Machine Learning**: scikit-learn
**Interactive Plots**: plotly, bokeh
**Automated EDA**: pandas-profiling, sweetviz

## Quick Reference

**Data Overview**: `df.info()`, `df.describe()`, `df.head()`
**Missing Values**: `df.isnull().sum()`, `df.isnull().mean()`
**Correlations**: `df.corr()`, `sns.heatmap()`
**Distributions**: `plt.hist()`, `sns.distplot()`, `df.plot.box()`
**Relationships**: `sns.scatterplot()`, `sns.pairplot()`

## Next Steps
- Practice EDA on different types of datasets
- Learn advanced visualization techniques
- Study domain-specific analysis methods
- Explore automated EDA tools
- Develop storytelling skills for data insights