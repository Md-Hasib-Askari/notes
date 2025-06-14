## 🟢 **3. DataFrame Basics**

### ✅ What is a DataFrame?

* A **DataFrame** is a **2-dimensional labeled data structure** with columns of potentially different types.
* Think of it as an in-memory table (like an Excel sheet or SQL table).

---

### ✅ Creating a DataFrame

#### From a dictionary:

```python
import pandas as pd

data = {
    'Name': ['Hasib', 'Ayesha'],
    'Age': [21, 22]
}
df = pd.DataFrame(data)
print(df)
```

#### From a list of dictionaries:

```python
data = [
    {'Name': 'Hasib', 'Age': 21},
    {'Name': 'Ayesha', 'Age': 22}
]
df = pd.DataFrame(data)
```

#### From NumPy arrays:

```python
import numpy as np

df = pd.DataFrame(np.array([[1, 2], [3, 4]]), columns=['A', 'B'])
```

---

### ✅ Accessing Data

#### Columns:

```python
df['Name']          # Returns a Series
df[['Name', 'Age']] # Returns a DataFrame
```

#### Rows:

```python
df.loc[0]      # Access by label/index – returns a Series
df.iloc[0]     # Access by integer position – returns a Series
```

#### Multiple rows:

```python
df.iloc[0:2]   # Slice rows
df.loc[[0, 1]] # Specific row labels
```

---

### ✅ Modifying Data

#### Add a new column:

```python
df['Gender'] = ['Male', 'Female']
```

#### Update values:

```python
df.at[0, 'Age'] = 23
```

#### Drop a column:

```python
df.drop('Gender', axis=1, inplace=True)
```

---

### ✅ Renaming Columns

```python
df.rename(columns={'Name': 'FullName'}, inplace=True)
```

---

### ✅ Summary Functions

```python
df.head()       # First 5 rows
df.tail()       # Last 5 rows
df.shape        # (rows, columns)
df.columns      # Column names
df.dtypes       # Data types
df.info()       # Overview
df.describe()   # Statistical summary
```

