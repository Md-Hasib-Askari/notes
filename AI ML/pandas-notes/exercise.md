**interactive Pandas exercise sheet**. You can copy/paste each section into a Jupyter Notebook or Google Colab and try solving them one by one. Each section starts simple and gets more challenging.

---

## 🧪 **Pandas Interactive Exercise Sheet**

### 📦 Setup

```python
import pandas as pd
import numpy as np
```

---

### ✅ **1. Create and Inspect**

```python
# Create a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'Department': ['HR', 'IT', 'Finance', 'IT'],
    'Salary': [50000, 60000, 70000, 80000]
}
df = pd.DataFrame(data)

# TASKS:
# 1. Show the first two rows
# 2. Display data types
# 3. Check for missing values
```

---

### ✅ **2. Filtering and Selection**

```python
# TASKS:
# 1. Select the 'Name' and 'Salary' columns
# 2. Filter rows where Age > 30
# 3. Filter employees from IT department with salary > 60000
```

---

### ✅ **3. Aggregation and Grouping**

```python
# TASKS:
# 1. Group by 'Department' and get average salary
# 2. Count number of employees in each department
```

---

### ✅ **4. Modifying Data**

```python
# TASKS:
# 1. Create a new column 'Tax' as 10% of salary
# 2. Add a new column 'Seniority' = 'Senior' if Age > 30 else 'Junior'
```

---

### ✅ **5. Missing Data**

```python
# Inject missing data
df.loc[2, 'Salary'] = np.nan

# TASKS:
# 1. Fill missing salary with the mean salary
# 2. Drop rows with any missing values
```

---

### ✅ **6. Merge and Concatenate**

```python
df2 = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Bonus': [5000, 4000, 4500, 3000]
})

# TASKS:
# 1. Merge df and df2 on 'Name'
# 2. Concatenate df and df2 side by side
```

---

### ✅ **7. Pivot Table & Crosstab**

```python
# TASKS:
# 1. Create a pivot table of average salary by department
# 2. Create a crosstab of Department vs Age > 30
```

---

### ✅ **8. Time Series**

```python
# Add date column
df['JoinDate'] = pd.date_range(start='2021-01-01', periods=4, freq='Y')

# TASKS:
# 1. Set JoinDate as index
# 2. Resample by year and sum salaries
```

---

### ✅ **9. Visualization (optional if matplotlib installed)**

```python
import matplotlib.pyplot as plt

# TASKS:
# 1. Plot salary over time
# 2. Plot department vs average salary bar chart
```

