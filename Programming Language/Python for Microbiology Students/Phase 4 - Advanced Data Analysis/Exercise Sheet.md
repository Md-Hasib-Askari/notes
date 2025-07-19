# Phase 4: Advanced Data Analysis - Exercise Sheet

## Exercise 1: Statistical Analysis with SciPy

### 1.1 Hypothesis Testing
```python
# Dataset: Growth rates of bacteria under different pH conditions
# pH 5.5: [0.12, 0.15, 0.13, 0.14, 0.11, 0.16, 0.12, 0.14]
# pH 7.0: [0.18, 0.21, 0.19, 0.20, 0.17, 0.22, 0.18, 0.19]
# pH 8.5: [0.09, 0.11, 0.08, 0.10, 0.07, 0.12, 0.09, 0.08]
```
**Tasks:**
- Perform a one-way ANOVA to test if pH significantly affects growth rate
- Calculate p-values and interpret results (α = 0.05)
- Perform post-hoc pairwise comparisons using Tukey's HSD test

### 1.2 Correlation Analysis
```python
# Dataset: Antibiotic concentration vs. inhibition zone diameter
# Concentration (μg/ml): [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# Zone diameter (mm): [8, 12, 15, 18, 21, 24, 26, 28, 30, 32]
```
**Tasks:**
- Calculate Pearson correlation coefficient
- Test for statistical significance of the correlation
- Create a linear regression model and calculate R²

### 1.3 Non-parametric Tests
```python
# Dataset: Colony counts before and after treatment (paired samples)
# Before: [45, 52, 38, 61, 48, 55, 42, 49, 56, 44]
# After:  [32, 41, 28, 48, 35, 42, 31, 36, 43, 33]
```
**Tasks:**
- Perform Wilcoxon signed-rank test
- Compare with paired t-test results
- Apply Bonferroni correction for multiple comparisons

## Exercise 2: Machine Learning with Scikit-learn

### 2.1 Microbial Classification
```python
# Dataset: Features for bacterial identification
# Features: [cell_length, cell_width, gram_stain, motility, catalase, oxidase]
# Classes: ['E.coli', 'S.aureus', 'B.subtilis', 'P.aeruginosa']
```
**Tasks:**
- Load the provided dataset (create synthetic data if needed)
- Split data into training and testing sets (80/20)
- Train Random Forest, SVM, and Naive Bayes classifiers
- Compare accuracy, precision, recall, and F1-scores
- Create confusion matrices for each model

### 2.2 Community Clustering Analysis
```python
# Dataset: OTU abundance matrix (samples × species)
# 20 samples, 15 microbial species, abundance values 0-100
```
**Tasks:**
- Perform K-means clustering with k=3,4,5
- Use hierarchical clustering with different linkage methods
- Determine optimal number of clusters using elbow method and silhouette analysis
- Visualize clusters using PCA plot

### 2.3 Principal Component Analysis
```python
# Dataset: Multiple environmental parameters affecting microbial communities
# Variables: temperature, pH, salinity, oxygen, nutrients (N, P, K), moisture
```
**Tasks:**
- Standardize the data before PCA
- Perform PCA and determine number of components explaining 90% variance
- Create biplot showing samples and variable loadings
- Interpret the biological meaning of first two components

### 2.4 Cross-validation and Model Evaluation
**Tasks:**
- Implement 5-fold cross-validation for the classification problem from 2.1
- Use stratified sampling to maintain class balance
- Calculate mean and standard deviation of performance metrics
- Perform hyperparameter tuning using GridSearchCV

## Exercise 3: Advanced Pandas Techniques

### 3.1 Merging and Joining Datasets
```python
# Dataset 1: Sample metadata
# Columns: sample_id, location, date_collected, temperature, pH
# Dataset 2: Microbial counts
# Columns: sample_id, species, count, biomass
# Dataset 3: Environmental data
# Columns: location, date, rainfall, humidity, soil_type
```
**Tasks:**
- Merge all three datasets on appropriate keys
- Handle missing values appropriately
- Create summary statistics by location and species
- Identify samples with incomplete data

### 3.2 Time Series Analysis for Growth Curves
```python
# Dataset: Bacterial growth measurements over 24 hours
# Columns: time_hours, strain_A_OD, strain_B_OD, strain_C_OD, temperature, pH
# Measurements every 30 minutes
```
**Tasks:**
- Calculate growth rates for each strain
- Identify lag phase, exponential phase, and stationary phase
- Smooth the data using rolling averages
- Compare growth patterns between strains
- Detect outliers in the time series data

### 3.3 Pivot Tables and Advanced Grouping
```python
# Dataset: Antibiotic susceptibility testing results
# Columns: bacterial_strain, antibiotic, concentration, inhibition_zone, 
#          test_date, laboratory, technician
```
**Tasks:**
- Create pivot table showing mean inhibition zones by strain and antibiotic
- Calculate resistance patterns (zones < 15mm = resistant)
- Group by laboratory and compare inter-lab variability
- Generate multi-level summary statistics

### 3.4 Performance Optimization
```python
# Large dataset: 100,000 microbial sequence records
# Columns: sequence_id, organism, sequence, length, gc_content, annotation
```
**Tasks:**
- Use pandas categorical data types for organism column
- Implement chunked processing for large file reading
- Compare performance of different groupby operations
- Optimize memory usage using appropriate data types

## Exercise 4: Database Interactions

### 4.1 SQL Basics for Biological Data
```sql
-- Database schema:
-- samples (sample_id, location, date_collected, researcher)
-- organisms (organism_id, genus, species, strain)
-- measurements (measurement_id, sample_id, organism_id, count, biomass)
-- environmental (env_id, sample_id, temperature, pH, salinity)
```
**Tasks:**
- Write queries to find samples with pH > 7.5
- Calculate average organism counts per location
- Find the most abundant organism in each sample
- Join tables to get complete sample information
- Create views for commonly used query patterns

### 4.2 API Data Retrieval
**Tasks:**
- Connect to NCBI Entrez API to retrieve bacterial genome information
- Download sequence data for specific bacterial strains
- Parse XML/JSON responses from biological databases
- Implement rate limiting and error handling
- Cache API responses to avoid repeated requests

### 4.3 Data Pipeline Creation
**Tasks:**
- Create automated pipeline for processing daily laboratory results
- Implement data validation and quality checks
- Set up logging for pipeline monitoring
- Handle different input file formats (CSV, Excel, JSON)
- Generate automated reports and alerts for unusual results

## Practical Challenges

### Challenge 1: Antimicrobial Resistance Analysis
Using a provided dataset of clinical isolates:
- Identify resistance patterns across different bacterial species
- Predict resistance based on genomic markers using machine learning
- Analyze temporal trends in resistance development
- Create interactive visualizations for resistance monitoring

### Challenge 2: Microbiome Diversity Study
Using 16S rRNA sequencing data:
- Calculate alpha diversity metrics (Shannon, Simpson, Chao1)
- Perform beta diversity analysis using UniFrac distances
- Identify significant differences between sample groups
- Create publication-ready plots and statistical summaries

### Challenge 3: Laboratory Quality Control Dashboard
Create an automated system that:
- Monitors daily laboratory measurements for outliers
- Tracks equipment calibration schedules
- Generates alerts for unusual results
- Produces weekly quality control reports

## Data Files

Create or download the following datasets for exercises:
1. `growth_rates_ph.csv` - Bacterial growth under different pH conditions
2. `antibiotic_zones.csv` - Antibiotic susceptibility test results
3. `otu_abundance.csv` - Microbial community abundance matrix
4. `environmental_data.csv` - Environmental parameters and microbial counts
5. `time_series_growth.csv` - Growth curve measurements over time
6. `large_sequence_data.csv` - Large dataset for performance optimization

## Assessment Criteria

For each exercise, evaluate:
- **Correctness**: Proper implementation of statistical tests and ML algorithms
- **Code Quality**: Clean, readable, and well-documented code
- **Interpretation**: Correct biological interpretation of results
- **Visualization**: Clear and informative plots and figures
- **Performance**: Efficient handling of large datasets
- **Reproducibility**: Code that can be run by others with consistent results

## Additional Resources

- SciPy documentation for statistical functions
- Scikit-learn user guide for machine learning
- Pandas documentation for advanced data manipulation
- SQLite tutorial for database operations
- Matplotlib/Seaborn galleries for visualization examples
