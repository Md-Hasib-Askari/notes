# Phase 2 Exercises: Scientific Python Ecosystem for Microbiology

## Overview
These exercises focus on the core scientific Python libraries essential for microbiology research: NumPy for numerical computing, Pandas for data analysis, Matplotlib/Seaborn for visualization, and Jupyter Notebooks for interactive analysis.

---

## Section 1: NumPy Fundamentals

### Exercise 1.1: Creating and Manipulating Arrays
```python
import numpy as np

# Bacterial colony counts from different plates
plate_1 = [45, 52, 38, 61, 49, 55, 42, 58]
plate_2 = [38, 44, 35, 55, 41, 48, 39, 52]
plate_3 = [42, 49, 40, 58, 46, 51, 43, 56]

# Convert to NumPy arrays
counts_array = np.array([plate_1, plate_2, plate_3])
print("Colony counts shape:", counts_array.shape)
```

**Tasks**:
1. Create a 4x6 array of random OD600 values between 0.1 and 2.0
2. Generate a time series array from 0 to 24 hours with 0.5-hour intervals
3. Create a 3D array representing pH measurements across different conditions
4. Initialize arrays for bacterial growth rates using `np.zeros()` and `np.ones()`

### Exercise 1.2: Mathematical Operations on Arrays
```python
# Growth rate data (per hour)
growth_rates = np.array([0.5, 0.7, 0.45, 0.8, 0.6, 0.55, 0.65, 0.4])

# Temperature data (Celsius)
temperatures = np.array([25, 30, 35, 37, 40, 42, 45, 50])

# Initial OD600 values
initial_od = np.array([0.05, 0.08, 0.06, 0.07, 0.05, 0.09, 0.04, 0.06])
```

**Tasks**:
1. Calculate doubling times using the formula: `doubling_time = ln(2) / growth_rate`
2. Convert temperatures from Celsius to Kelvin
3. Normalize growth rates to Z-scores
4. Calculate the correlation coefficient between temperature and growth rate
5. Find conditions where growth rate > 0.6 AND temperature < 40°C

### Exercise 1.3: Array Indexing and Slicing
```python
# 16S rRNA gene abundance data (rows: samples, columns: species)
abundance_matrix = np.random.poisson(100, size=(20, 8))  # 20 samples, 8 species
species_names = ['E.coli', 'B.subtilis', 'L.acidophilus', 'S.aureus', 
                 'P.aeruginosa', 'C.difficile', 'B.fragilis', 'E.faecalis']
sample_ids = [f'Sample_{i:03d}' for i in range(1, 21)]
```

**Tasks**:
1. Extract abundance data for the first 5 samples
2. Get all abundance values for 'E.coli' (first column)
3. Find samples with total abundance > 1000
4. Create a subset with only samples 5-15 and species 2-6
5. Replace negative values (if any) with zeros

### Exercise 1.4: Statistical Functions and Random Generation
```python
# Simulating antibiotic effectiveness data
np.random.seed(42)  # For reproducible results

# MIC values for different antibiotics (µg/mL)
mic_values = np.random.lognormal(mean=2, sigma=0.5, size=100)

# Inhibition zone diameters (mm)
zone_diameters = np.random.normal(loc=15, scale=3, size=100)
```

**Tasks**:
1. Calculate basic statistics (mean, median, std, min, max) for both datasets
2. Generate 50 random pH values from a normal distribution (mean=7.0, std=0.3)
3. Create a Boolean mask for MIC values > 10 µg/mL
4. Calculate percentiles (25th, 50th, 75th, 95th) for zone diameters
5. Simulate bacterial counts using Poisson distribution with λ=150

---

## Section 2: Pandas for Data Analysis

### Exercise 2.1: DataFrames and Series Basics
```python
import pandas as pd
import numpy as np

# Create sample microbiology dataset
data = {
    'Sample_ID': ['S001', 'S002', 'S003', 'S004', 'S005', 'S006'],
    'Patient_Age': [25, 34, 45, 67, 23, 56],
    'Sample_Type': ['blood', 'urine', 'sputum', 'blood', 'urine', 'sputum'],
    'Bacteria_Count': [150, 1200, 890, 45, 2300, 670],
    'Pathogenic': [True, True, True, False, True, True],
    'Collection_Date': ['2023-07-01', '2023-07-02', '2023-07-03', 
                       '2023-07-04', '2023-07-05', '2023-07-06']
}

df = pd.DataFrame(data)
df['Collection_Date'] = pd.to_datetime(df['Collection_Date'])
```

**Tasks**:
1. Display basic information about the DataFrame (shape, dtypes, info)
2. Calculate summary statistics for numerical columns
3. Create a new column 'Risk_Level' based on bacteria count thresholds
4. Filter rows where Sample_Type is 'blood' and Pathogenic is True
5. Sort the DataFrame by Bacteria_Count in descending order

### Exercise 2.2: Reading and Processing Different File Formats
```python
# Sample CSV data creation
growth_data = {
    'Time_hours': list(range(0, 25, 2)),
    'E_coli_OD': [0.05, 0.08, 0.15, 0.28, 0.52, 0.95, 1.68, 2.85, 4.12, 4.95, 5.02, 5.01, 5.00],
    'B_subtilis_OD': [0.06, 0.09, 0.16, 0.30, 0.48, 0.82, 1.45, 2.34, 3.45, 4.23, 4.78, 4.85, 4.82],
    'Temperature': [37.0, 37.1, 36.9, 37.2, 37.0, 36.8, 37.1, 37.0, 36.9, 37.1, 37.0, 37.2, 37.1]
}

# Save to CSV for practice
pd.DataFrame(growth_data).to_csv('growth_curve_data.csv', index=False)
```

**Tasks**:
1. Read the CSV file and display the first 5 rows
2. Handle any missing values appropriately
3. Create a 'Growth_Phase' column based on OD600 values
4. Export filtered data (exponential phase only) to a new CSV
5. Read data from an Excel file (create one first) with multiple sheets

### Exercise 2.3: Data Cleaning and Preprocessing
```python
# Create messy dataset for cleaning practice
messy_data = pd.DataFrame({
    'Sample_ID': ['S001', 'S002', 'S003', 'S004', 'S005', 'S006', 'S007'],
    'Species': ['E.coli', 'e. coli', 'E. Coli', 'B.subtilis', 'b. SUBTILIS', 
               'Bacillus subtilis', 'Unknown'],
    'Count': [150, 1200, '890', 'contaminated', 2300, None, 45],
    'pH': [7.2, 6.8, 7.1, None, 6.9, 'error', 7.0],
    'Date': ['2023-07-01', '07/02/2023', '2023.07.03', '2023-07-04', 
            '2023-07-05', 'missing', '2023-07-07']
})
```

**Tasks**:
1. Standardize species names (handle case inconsistencies)
2. Convert 'Count' column to numeric, handling non-numeric values
3. Clean and convert pH values to float
4. Standardize date formats
5. Create a data quality report showing issues found and fixed

### Exercise 2.4: Filtering, Grouping, and Aggregating
```python
# Antibiotic resistance dataset
resistance_data = pd.DataFrame({
    'Isolate_ID': [f'ISO_{i:03d}' for i in range(1, 101)],
    'Species': np.random.choice(['E.coli', 'S.aureus', 'P.aeruginosa', 'K.pneumoniae'], 100),
    'Antibiotic': np.random.choice(['Ampicillin', 'Ciprofloxacin', 'Vancomycin', 'Ceftriaxone'], 100),
    'MIC_ugml': np.random.lognormal(mean=1.5, sigma=1, size=100),
    'Resistant': np.random.choice([True, False], 100, p=[0.3, 0.7]),
    'Hospital_Unit': np.random.choice(['ICU', 'Surgery', 'Pediatrics', 'Emergency'], 100),
    'Collection_Year': np.random.choice([2020, 2021, 2022, 2023], 100)
})
```

**Tasks**:
1. Calculate resistance rates by species and antibiotic
2. Find the average MIC value for each hospital unit
3. Identify trends in resistance over years
4. Create a pivot table showing resistance patterns
5. Filter for high-risk combinations (ICU + resistant + specific antibiotics)

---

## Section 3: Matplotlib and Seaborn Visualization

### Exercise 3.1: Basic Plotting with Matplotlib
```python
import matplotlib.pyplot as plt
import numpy as np

# Growth curve data
time = np.linspace(0, 24, 50)
od_600 = 0.05 * np.exp(0.3 * time) / (1 + (0.05 * np.exp(0.3 * time) - 0.05) / 4.5)  # Logistic growth

# Add some noise
od_600 += np.random.normal(0, 0.1, len(od_600))
```

**Tasks**:
1. Create a line plot of the growth curve with proper labels and title
2. Add error bars using standard deviation
3. Create subplots comparing different growth conditions
4. Generate a scatter plot of OD600 vs time with trend line
5. Customize plot appearance (colors, styles, grid, legend)

### Exercise 3.2: Advanced Matplotlib Visualizations
```python
# Antibiotic susceptibility data
antibiotics = ['Ampicillin', 'Ceftriaxone', 'Ciprofloxacin', 'Vancomycin', 'Tetracycline']
susceptible = [65, 78, 82, 45, 71]
intermediate = [15, 12, 8, 25, 18]
resistant = [20, 10, 10, 30, 11]

# Microbiome composition data
species = ['Bacteroides', 'Firmicutes', 'Proteobacteria', 'Actinobacteria', 'Others']
abundance = [35, 28, 18, 12, 7]
```

**Tasks**:
1. Create a stacked bar chart for antibiotic susceptibility
2. Generate a pie chart for microbiome composition
3. Create a heatmap showing MIC values across species and antibiotics
4. Design a multi-panel figure combining different plot types
5. Make publication-ready figures with proper formatting

### Exercise 3.3: Statistical Visualizations with Seaborn
```python
import seaborn as sns

# Generate comprehensive dataset
np.random.seed(42)
n_samples = 200

microbiome_df = pd.DataFrame({
    'Patient_ID': range(1, n_samples + 1),
    'Age': np.random.normal(45, 15, n_samples),
    'BMI': np.random.normal(25, 4, n_samples),
    'Shannon_Diversity': np.random.gamma(2, 1.5, n_samples),
    'Firmicutes_Ratio': np.random.beta(2, 2, n_samples),
    'Disease_Status': np.random.choice(['Healthy', 'IBD', 'IBS'], n_samples, p=[0.5, 0.3, 0.2]),
    'Antibiotic_Use': np.random.choice(['None', 'Recent', 'Current'], n_samples, p=[0.6, 0.25, 0.15])
})
```

**Tasks**:
1. Create correlation matrix heatmap for numerical variables
2. Generate box plots comparing diversity across disease status
3. Create pair plots to explore relationships between variables
4. Design violin plots for BMI distribution by antibiotic use
5. Build a comprehensive dashboard combining multiple visualizations

### Exercise 3.4: Publication-Ready Figures
```python
# Time series antibiotic resistance data
years = np.arange(2015, 2024)
mrsa_resistance = [15, 18, 22, 26, 29, 32, 28, 25, 23]  # Percentage
vre_resistance = [8, 12, 15, 18, 22, 25, 23, 20, 19]
esbl_resistance = [25, 28, 32, 35, 38, 42, 45, 43, 41]
```

**Tasks**:
1. Create a multi-line plot with confidence intervals
2. Design figures following journal submission guidelines
3. Generate high-resolution figures suitable for publication
4. Create interactive plots using plotly (bonus)
5. Build automated figure generation pipeline

---

## Section 4: Jupyter Notebooks

### Exercise 4.1: Setting Up and Using Jupyter
**Tasks**:
1. Create a new Jupyter notebook for microbiome analysis
2. Set up proper imports and configure display options
3. Create a table of contents using markdown headers
4. Implement notebook organization best practices
5. Configure custom CSS for improved appearance

### Exercise 4.2: Markdown Documentation
Create comprehensive markdown cells covering:

```markdown
# Microbiology Data Analysis Project

## Objective
Analyze bacterial growth patterns and antibiotic resistance trends from clinical isolates.

## Methodology
- **Data Collection**: Clinical isolates from 2020-2023
- **Growth Analysis**: OD600 measurements every 2 hours
- **Susceptibility Testing**: Standard disk diffusion method
- **Statistical Analysis**: ANOVA and post-hoc tests

## Expected Outcomes
1. Identification of optimal growth conditions
2. Resistance pattern characterization
3. Temporal trend analysis

### Mathematical Formulas
Growth rate calculation: $\mu = \frac{\ln(N_t) - \ln(N_0)}{t}$

Doubling time: $t_d = \frac{\ln(2)}{\mu}$
```

**Tasks**:
1. Create section headers with proper markdown syntax
2. Include mathematical formulas using LaTeX
3. Add bullet points and numbered lists
4. Insert images and tables
5. Create hyperlinks to external resources

### Exercise 4.3: Interactive Data Exploration
```python
# Interactive widgets example
from ipywidgets import interact, IntSlider, Dropdown
import matplotlib.pyplot as plt

def plot_growth_curve(growth_rate, carrying_capacity, initial_population):
    time = np.linspace(0, 24, 100)
    population = carrying_capacity / (1 + ((carrying_capacity - initial_population) / initial_population) * np.exp(-growth_rate * time))
    
    plt.figure(figsize=(10, 6))
    plt.plot(time, population, 'b-', linewidth=2)
    plt.xlabel('Time (hours)')
    plt.ylabel('Population (OD600)')
    plt.title(f'Bacterial Growth Curve (r={growth_rate}, K={carrying_capacity})')
    plt.grid(True, alpha=0.3)
    plt.show()

# Interactive widget
interact(plot_growth_curve,
         growth_rate=FloatSlider(min=0.1, max=1.0, step=0.1, value=0.5),
         carrying_capacity=IntSlider(min=1, max=10, step=1, value=5),
         initial_population=FloatSlider(min=0.01, max=0.5, step=0.01, value=0.05))
```

**Tasks**:
1. Create interactive parameter exploration widgets
2. Build dynamic data filtering interfaces
3. Implement real-time plot updates
4. Design interactive dashboards
5. Create guided analysis workflows

### Exercise 4.4: Sharing and Presenting Analyses
**Tasks**:
1. Convert notebook to HTML for sharing
2. Create PDF reports with nbconvert
3. Export specific cells as Python scripts
4. Set up version control for notebooks
5. Create presentation slides using RISE

---

## Comprehensive Project: Microbiome Analysis Pipeline

### Project Overview
Create a complete analysis pipeline that integrates all Phase 2 concepts:

### Dataset
Generate or use real microbiome data including:
- Sample metadata (age, BMI, disease status)
- OTU abundance tables
- Alpha diversity metrics
- Beta diversity distances
- Taxonomic classifications

### Analysis Components

#### 1. Data Import and Cleaning (NumPy/Pandas)
```python
# Load and clean multiple data files
# Handle missing values and outliers
# Merge datasets on sample IDs
# Validate data integrity
```

#### 2. Statistical Analysis (NumPy/Pandas)
```python
# Calculate diversity metrics
# Perform correlation analyses
# Compare groups using appropriate tests
# Handle multiple testing corrections
```

#### 3. Visualization (Matplotlib/Seaborn)
```python
# Create comprehensive figure panels
# Generate publication-ready plots
# Build interactive dashboards
# Export high-quality graphics
```

#### 4. Documentation (Jupyter)
```python
# Document methodology clearly
# Include interactive exploration
# Create reproducible analysis
# Generate automated reports
```

### Deliverables
1. **Jupyter Notebook**: Complete analysis with markdown documentation
2. **Data Processing Scripts**: Reusable Python modules
3. **Visualization Gallery**: Collection of publication-ready figures
4. **Analysis Report**: PDF summary with key findings
5. **Code Repository**: Version-controlled project with proper documentation

---

## Evaluation Criteria

### Technical Skills (60%)
- **NumPy Proficiency**: Array operations, statistical functions, indexing
- **Pandas Mastery**: Data manipulation, cleaning, aggregation
- **Visualization Quality**: Clear, informative, and well-designed plots
- **Jupyter Usage**: Effective documentation and interactive features

### Scientific Application (25%)
- **Biological Relevance**: Appropriate use of microbiology concepts
- **Statistical Rigor**: Proper analysis methods and interpretations
- **Data Quality**: Attention to data validation and cleaning
- **Reproducibility**: Clear methodology and reusable code

### Communication (15%)
- **Documentation**: Clear explanations and code comments
- **Visualization**: Effective communication through graphics
- **Organization**: Logical flow and structure
- **Presentation**: Professional appearance and formatting

## Tips for Success

### NumPy Best Practices
- Use vectorized operations instead of loops
- Understand broadcasting rules
- Choose appropriate data types
- Profile code for performance bottlenecks

### Pandas Efficiency
- Use method chaining for readable code
- Leverage built-in string and datetime methods
- Understand when to use apply() vs vectorized operations
- Use categorical data for memory efficiency

### Visualization Guidelines
- Choose appropriate plot types for data
- Use consistent color schemes and styling
- Include proper labels, titles, and legends
- Consider your audience and purpose

### Jupyter Workflow
- Use meaningful cell organization
- Include explanatory markdown cells
- Test code in small chunks
- Keep notebooks focused and modular

## Next Steps

After mastering Phase 2, you'll be ready for:
- **Phase 3**: Bioinformatics applications with BioPython
- **Advanced Statistics**: Hypothesis testing and experimental design
- **Machine Learning**: Pattern recognition in biological data
- **Specialized Tools**: Domain-specific analysis packages

## Additional Resources

### Documentation
- [NumPy User Guide](https://numpy.org/doc/stable/user/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/)

### Practice Datasets
- [Microbiome datasets from Qiita](https://qiita.ucsd.edu/)
- [Clinical microbiology data](https://www.ncbi.nlm.nih.gov/sra)
- [Antibiotic resistance databases](https://card.mcmaster.ca/)

Good luck with Phase 2! The scientific Python ecosystem will become your foundation for all future microbiology data analysis. 🧬📊🐍
