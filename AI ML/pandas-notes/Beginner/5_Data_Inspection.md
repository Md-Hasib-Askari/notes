## 🟢 **5. Data Inspection**

### ✅ Overview Functions

#### `df.head(n)`

* Shows the first `n` rows (default is 5).

```python
df.head()
```

#### `df.tail(n)`

* Shows the last `n` rows.

```python
df.tail(3)
```

#### `df.shape`

* Returns a tuple: (number of rows, number of columns)

```python
df.shape
```

#### `df.columns`

* Lists all column names.

```python
df.columns
```

#### `df.dtypes`

* Shows data types of each column.

```python
df.dtypes
```

---

### ✅ Summary and Metadata

#### `df.info()`

* Displays a summary including:

  * Number of entries
  * Column names and data types
  * Memory usage
  * Non-null counts

```python
df.info()
```

#### `df.describe()`

* Statistical summary of numeric columns:

  * count, mean, std, min, 25%, 50%, 75%, max

```python
df.describe()
```

#### `df.value_counts()`

* Counts unique values in a Series.

```python
df['column_name'].value_counts()
```

---

### ✅ Checking for Missing Data

#### `df.isnull()` / `df.notnull()`

* Returns a DataFrame of boolean values.

```python
df.isnull()
df.notnull()
```

#### `df.isnull().sum()`

* Total number of missing values per column.

```python
df.isnull().sum()
```

---

### ✅ Basic Filtering

```python
df[df['Age'] > 21]         # Filter rows where Age > 21
df[df['Gender'] == 'Male'] # Filter rows by condition
```
