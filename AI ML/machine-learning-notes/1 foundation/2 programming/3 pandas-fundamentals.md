# Pandas Fundamentals

## Overview
Pandas is the cornerstone library for data manipulation and analysis in Python. It provides powerful, flexible data structures and data analysis tools that are essential for machine learning workflows.

## Learning Objectives
- [ ] Master DataFrame and Series operations
- [ ] Perform efficient data cleaning and preprocessing
- [ ] Handle missing data and outliers effectively
- [ ] Execute complex data transformations
- [ ] Optimize pandas operations for large datasets

## 1. Core Data Structures

### 1.1 Series - One-dimensional labeled array
```python
import pandas as pd
import numpy as np

# Creating Series
s1 = pd.Series([1, 3, 5, np.nan, 6, 8])
s2 = pd.Series({'a': 1, 'b': 2, 'c': 3})
s3 = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])

print(s1.dtype)  # float64
print(s2.index)  # Index(['a', 'b', 'c'], dtype='object')
```

### 1.2 DataFrame - Two-dimensional labeled data structure
```python
# Creating DataFrames
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'Salary': [50000, 60000, 70000, 55000],
    'Department': ['IT', 'Finance', 'IT', 'HR']
}
df = pd.DataFrame(data)

# From arrays
df2 = pd.DataFrame(np.random.randn(4, 3), 
                   columns=['A', 'B', 'C'],
                   index=['Row1', 'Row2', 'Row3', 'Row4'])

# From file
# df = pd.read_csv('data.csv')
# df = pd.read_excel('data.xlsx')
# df = pd.read_json('data.json')
```

## 2. Data Inspection and Information

### 2.1 Basic Information
```python
# Dataset overview
print(df.shape)          # (4, 4) - rows, columns
print(df.info())         # Data types, memory usage
print(df.describe())     # Statistical summary
print(df.head(3))        # First 3 rows
print(df.tail(2))        # Last 2 rows

# Column and index information
print(df.columns.tolist())  # Column names
print(df.index)            # Index information
print(df.dtypes)           # Data types
```

### 2.2 Memory and Performance
```python
# Memory usage
print(df.memory_usage(deep=True))
print(f"Total memory: {df.memory_usage(deep=True).sum()} bytes")

# Optimize data types
df['Age'] = df['Age'].astype('int8')  # Reduce memory
df['Department'] = df['Department'].astype('category')  # Categorical data
```

## 3. Indexing and Selection

### 3.1 Column Selection
```python
# Single column
ages = df['Age']  # Returns Series
ages = df.Age     # Attribute access

# Multiple columns
subset = df[['Name', 'Salary']]  # Returns DataFrame
```

### 3.2 Row Selection
```python
# By position (iloc)
first_row = df.iloc[0]        # First row
first_three = df.iloc[0:3]    # First 3 rows
specific_rows = df.iloc[[0, 2, 3]]  # Specific rows

# By label (loc)
row_by_index = df.loc[0]      # By index label
conditional = df.loc[df['Age'] > 30]  # Conditional selection
```

### 3.3 Boolean Indexing
```python
# Filter data
young_employees = df[df['Age'] < 30]
it_employees = df[df['Department'] == 'IT']
high_earners = df[df['Salary'] > 55000]

# Multiple conditions
young_it = df[(df['Age'] < 30) & (df['Department'] == 'IT')]
young_or_rich = df[(df['Age'] < 30) | (df['Salary'] > 60000)]

# Using isin()
specific_depts = df[df['Department'].isin(['IT', 'Finance'])]
```

## 4. Data Operations

### 4.1 Adding and Modifying Data
```python
# Add new column
df['Bonus'] = df['Salary'] * 0.1
df['Experience'] = [2, 5, 10, 3]  # From list
df['Grade'] = df['Salary'].apply(lambda x: 'A' if x > 60000 else 'B')

# Modify existing data
df.loc[df['Age'] > 30, 'Salary'] *= 1.1  # 10% raise for seniors
df['Department'] = df['Department'].str.upper()
```

### 4.2 String Operations
```python
# String methods
df['Name_Upper'] = df['Name'].str.upper()
df['Name_Length'] = df['Name'].str.len()
df['Name_Contains_A'] = df['Name'].str.contains('a', case=False)

# Extract information
email_data = pd.Series(['alice@gmail.com', 'bob@yahoo.com', 'charlie@hotmail.com'])
df['Email_Domain'] = email_data.str.split('@').str[1]
```

### 4.3 Mathematical Operations
```python
# Arithmetic operations
df['Salary_Thousands'] = df['Salary'] / 1000
df['Age_Squared'] = df['Age'] ** 2

# Statistical operations
mean_salary = df['Salary'].mean()
std_age = df['Age'].std()
correlation = df[['Age', 'Salary']].corr()
```

## 5. Grouping and Aggregation

### 5.1 GroupBy Operations
```python
# Group by single column
dept_stats = df.groupby('Department').agg({
    'Salary': ['mean', 'max', 'min', 'count'],
    'Age': ['mean', 'std']
})

# Multiple grouping
age_groups = pd.cut(df['Age'], bins=[0, 30, 40, 100], labels=['Young', 'Middle', 'Senior'])
df['Age_Group'] = age_groups
multi_group = df.groupby(['Department', 'Age_Group'])['Salary'].mean()
```

### 5.2 Pivot Tables
```python
# Create pivot table
pivot = df.pivot_table(
    values='Salary',
    index='Department',
    columns='Age_Group',
    aggfunc='mean',
    fill_value=0
)

# Cross-tabulation
crosstab = pd.crosstab(df['Department'], df['Age_Group'])
```

## 6. Data Cleaning and Preprocessing

### 6.1 Handling Missing Data
```python
# Create sample data with missing values
data_with_nan = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [5, np.nan, 7, 8, 9],
    'C': [10, 11, 12, np.nan, 14]
}
df_nan = pd.DataFrame(data_with_nan)

# Detect missing values
print(df_nan.isnull().sum())  # Count nulls per column
print(df_nan.isnull().any())  # Any nulls per column

# Handle missing values
df_dropped = df_nan.dropna()  # Remove rows with any null
df_dropped_cols = df_nan.dropna(axis=1)  # Remove columns with any null
df_filled = df_nan.fillna(0)  # Fill with constant
df_ffill = df_nan.fillna(method='ffill')  # Forward fill
df_interpolated = df_nan.interpolate()  # Linear interpolation
```

### 6.2 Duplicate Handling
```python
# Create data with duplicates
duplicate_data = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Alice', 'Charlie'],
    'Score': [85, 90, 85, 95]
})

# Detect and remove duplicates
print(duplicate_data.duplicated())  # Boolean mask
print(duplicate_data.drop_duplicates())  # Remove duplicates
print(duplicate_data.drop_duplicates(subset=['Name']))  # Based on specific column
```

### 6.3 Data Type Conversion
```python
# Convert data types
df['Age'] = df['Age'].astype('float64')
df['Department'] = df['Department'].astype('category')
df['Hire_Date'] = pd.to_datetime(['2020-01-15', '2019-03-20', '2018-06-10', '2021-09-05'])

# Handle categorical data
df['Department_Code'] = pd.Categorical(df['Department']).codes
dept_dummies = pd.get_dummies(df['Department'], prefix='Dept')
```

## 7. Merging and Joining

### 7.1 Concatenation
```python
# Create sample DataFrames
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
df3 = pd.DataFrame({'C': [9, 10], 'D': [11, 12]})

# Concatenate
vertical = pd.concat([df1, df2], axis=0)  # Stack vertically
horizontal = pd.concat([df1, df3], axis=1)  # Side by side
```

### 7.2 Merging DataFrames
```python
# Create related DataFrames
employees = pd.DataFrame({
    'emp_id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'dept_id': [10, 20, 10, 30]
})

departments = pd.DataFrame({
    'dept_id': [10, 20, 30],
    'dept_name': ['IT', 'Finance', 'HR']
})

# Different types of joins
inner_join = pd.merge(employees, departments, on='dept_id', how='inner')
left_join = pd.merge(employees, departments, on='dept_id', how='left')
outer_join = pd.merge(employees, departments, on='dept_id', how='outer')
```

## 8. Time Series Operations

### 8.1 DateTime Handling
```python
# Create time series data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
ts_data = pd.DataFrame({
    'date': dates,
    'value': np.random.randn(100).cumsum()
})

# Set datetime index
ts_data.set_index('date', inplace=True)

# Extract time components
ts_data['year'] = ts_data.index.year
ts_data['month'] = ts_data.index.month
ts_data['day_of_week'] = ts_data.index.dayofweek
```

### 8.2 Resampling and Rolling Windows
```python
# Resample to monthly averages
monthly_avg = ts_data.resample('M').mean()

# Rolling statistics
ts_data['rolling_mean'] = ts_data['value'].rolling(window=7).mean()
ts_data['rolling_std'] = ts_data['value'].rolling(window=7).std()
```

## 9. Performance Optimization

### 9.1 Efficient Operations
```python
# Use vectorized operations instead of loops
# BAD: Using loops
result = []
for value in df['Salary']:
    result.append(value * 1.1)

# GOOD: Vectorized operation
df['Salary_Increased'] = df['Salary'] * 1.1

# Use query() for complex filtering
filtered = df.query('Age > 30 and Salary > 55000')

# Avoid chained indexing
# BAD
df[df['Age'] > 30]['Salary'] = df[df['Age'] > 30]['Salary'] * 1.1

# GOOD
mask = df['Age'] > 30
df.loc[mask, 'Salary'] *= 1.1
```

### 9.2 Memory Optimization
```python
# Optimize data types
def optimize_dtypes(df):
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
    return df
```

## 10. Machine Learning Integration

### 10.1 Feature Engineering
```python
# Create polynomial features
df['Age_Salary_Interaction'] = df['Age'] * df['Salary']
df['Salary_Log'] = np.log(df['Salary'])

# Binning continuous variables
df['Age_Bin'] = pd.cut(df['Age'], bins=3, labels=['Young', 'Middle', 'Senior'])
df['Salary_Quartile'] = pd.qcut(df['Salary'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
```

### 10.2 Data Preparation for ML
```python
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Separate features and target
X = df[['Age', 'Salary']]
y = df['Department']

# Handle categorical variables
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create final dataset
ml_ready_df = pd.DataFrame(X_scaled, columns=['Age_scaled', 'Salary_scaled'])
ml_ready_df['Department_encoded'] = y_encoded
```

## 11. Advanced Techniques

### 11.1 Apply Functions
```python
# Apply custom functions
def categorize_salary(salary):
    if salary < 55000:
        return 'Low'
    elif salary < 65000:
        return 'Medium'
    else:
        return 'High'

df['Salary_Category'] = df['Salary'].apply(categorize_salary)

# Apply to multiple columns
def calculate_score(row):
    return (row['Age'] * 0.3) + (row['Salary'] / 1000 * 0.7)

df['Score'] = df.apply(calculate_score, axis=1)
```

### 11.2 Window Functions
```python
# Ranking
df['Salary_Rank'] = df['Salary'].rank(ascending=False)
df['Salary_Percentile'] = df['Salary'].rank(pct=True)

# Cumulative operations
df = df.sort_values('Salary')
df['Cumulative_Salary'] = df['Salary'].cumsum()
df['Running_Average'] = df['Salary'].expanding().mean()
```

## 12. Best Practices for ML Workflows

### 12.1 Data Validation
```python
# Validate data integrity
def validate_data(df):
    # Check for unexpected nulls
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"Warning: Found null values:\n{null_counts[null_counts > 0]}")
    
    # Check for duplicates
    if df.duplicated().any():
        print(f"Warning: Found {df.duplicated().sum()} duplicate rows")
    
    # Check data ranges
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if (df[col] < 0).any():
            print(f"Warning: Negative values found in {col}")

validate_data(df)
```

### 12.2 Reproducible Data Processing
```python
# Create processing pipeline
class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def fit_transform(self, df):
        # Make a copy to avoid modifying original
        processed_df = df.copy()
        
        # Handle missing values
        processed_df = processed_df.fillna(processed_df.median())
        
        # Encode categorical variables
        categorical_cols = processed_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            processed_df[col] = le.fit_transform(processed_df[col])
            self.label_encoders[col] = le
        
        # Scale numerical features
        numerical_cols = processed_df.select_dtypes(include=[np.number]).columns
        processed_df[numerical_cols] = self.scaler.fit_transform(processed_df[numerical_cols])
        
        return processed_df

# Usage
processor = DataProcessor()
processed_data = processor.fit_transform(df)
```

## 13. Common Patterns and Idioms

### 13.1 Chaining Operations
```python
# Method chaining for cleaner code
result = (df
    .query('Age > 25')
    .groupby('Department')
    .agg({'Salary': 'mean'})
    .sort_values('Salary', ascending=False)
    .head(3)
)
```

### 13.2 Conditional Assignment
```python
# Multiple conditions with np.where
df['Employee_Level'] = np.where(
    df['Age'] < 30, 'Junior',
    np.where(df['Age'] < 40, 'Mid-level', 'Senior')
)

# Using pd.cut for binning
df['Salary_Band'] = pd.cut(
    df['Salary'], 
    bins=[0, 50000, 60000, 70000, np.inf],
    labels=['Low', 'Medium', 'High', 'Very High']
)
```

## 14. Debugging and Troubleshooting

### 14.1 Common Issues
```python
# Debug pandas operations
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)        # Don't wrap lines

# Check for common issues
print("Data types:", df.dtypes)
print("Missing values:", df.isnull().sum())
print("Unique values per column:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()}")
```

### 14.2 Performance Profiling
```python
import time

# Time operations
start_time = time.time()
result = df.groupby('Department')['Salary'].mean()
end_time = time.time()
print(f"Operation took {end_time - start_time:.4f} seconds")

# Use %timeit in Jupyter notebooks
# %timeit df.groupby('Department')['Salary'].mean()
```

## Summary

Pandas is essential for ML data preprocessing and analysis. Key takeaways:

1. **Master the basics**: Series, DataFrames, indexing, and selection
2. **Clean data effectively**: Handle missing values, duplicates, and data types
3. **Use vectorized operations**: Avoid loops for better performance
4. **Chain operations**: Write cleaner, more readable code
5. **Optimize memory usage**: Use appropriate data types
6. **Validate data**: Implement checks for data integrity
7. **Create reusable pipelines**: Build reproducible data processing workflows

These fundamentals form the foundation for effective data manipulation in machine learning projects.