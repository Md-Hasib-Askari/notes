## 🟢 **7. Data Cleaning**

Cleaning data is a crucial step in any data analysis pipeline. Pandas offers powerful tools to handle missing data, incorrect formats, and duplicates.

---

### ✅ Handling Missing Data

#### Detect Missing Values

```python
df.isnull()            # Boolean DataFrame
df.isnull().sum()      # Count of nulls per column
```

#### Drop Missing Data

```python
df.dropna()            # Drop rows with any missing values
df.dropna(axis=1)      # Drop columns with any missing values
df.dropna(thresh=2)    # Drop rows with less than 2 non-NA values
```

#### Fill Missing Data

```python
df.fillna(0)                          # Replace all NaNs with 0
df['Age'].fillna(df['Age'].mean())   # Fill with column mean
df.fillna(method='ffill')            # Forward fill
df.fillna(method='bfill')            # Backward fill
```

---

### ✅ Removing Duplicates

#### Check for Duplicates

```python
df.duplicated()
```

#### Drop Duplicates

```python
df.drop_duplicates(inplace=True)
```

---

### ✅ Replacing Values

```python
df['Gender'].replace({'M': 'Male', 'F': 'Female'}, inplace=True)
```

---

### ✅ Renaming Columns and Index

```python
df.rename(columns={'old_name': 'new_name'}, inplace=True)
df.rename(index={0: 'first_row'}, inplace=True)
```

---

### ✅ Changing Data Types

```python
df['Age'] = df['Age'].astype(int)
df['Date'] = pd.to_datetime(df['Date'])
```

---

### ✅ String Cleaning (useful for messy text data)

```python
df['Name'] = df['Name'].str.strip()           # Remove spaces
df['Name'] = df['Name'].str.lower()           # Lowercase
df['Name'] = df['Name'].str.replace('-', ' ') # Replace characters
```

