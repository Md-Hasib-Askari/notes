## 🟢 **6. Data Selection and Filtering**

### ✅ Selecting Columns

#### Single column (returns Series):

```python
df['Name']
```

#### Multiple columns (returns DataFrame):

```python
df[['Name', 'Age']]
```

---

### ✅ Selecting Rows

#### By index position (`iloc`):

```python
df.iloc[0]        # First row
df.iloc[0:3]      # First three rows
```

#### By label/index name (`loc`):

```python
df.loc[0]         # Row with index 0
df.loc[2:4]       # Rows from index 2 to 4 (inclusive)
```

#### Using conditions:

```python
df[df['Age'] > 21]                    # Filter rows
df[(df['Age'] > 21) & (df['Gender'] == 'Male')]  # Multiple conditions
```

---

### ✅ Conditional Filtering Examples

```python
df[df['Name'].str.contains('Has')]   # String condition
df[df['Age'].isin([21, 23])]         # Check for multiple values
```

---

### ✅ Filtering with `query()`

```python
df.query("Age > 21 and Gender == 'Female'")
```

---

### ✅ Using `at`, `iat`, `loc`, `iloc` for access

* `at[row_label, col_label]`: Fast for scalar access.
* `iat[row_pos, col_pos]`: Fastest access by position.
* `loc[row_label, col_label]`: Label-based access.
* `iloc[row_pos, col_pos]`: Position-based access.

```python
df.at[0, 'Age'] = 25
df.iat[0, 1] = 25
```
