## 🟢 **14. Advanced Tips and Performance Optimization**

Efficient data processing is essential, especially with large datasets. These advanced tips help boost performance and write cleaner code.

---

### ✅ Use Vectorized Operations (Avoid Loops)

❌ Inefficient:

```python
df['double'] = [x*2 for x in df['value']]
```

✅ Efficient:

```python
df['double'] = df['value'] * 2
```

---

### ✅ Use `query()` for Filtering

```python
df.query("Age > 30 and Gender == 'Male'")
```

---

### ✅ Use `eval()` for Expression Evaluation

```python
df.eval("Bonus = Salary * 0.1", inplace=True)
```

---

### ✅ Categorical Data Optimization

```python
df['Category'] = df['Category'].astype('category')
```

Reduces memory usage for columns with repeated string values.

---

### ✅ Memory Usage Summary

```python
df.info(memory_usage='deep')
df.memory_usage(deep=True)
```

---

### ✅ Use `astype()` for Efficient Data Types

```python
df['int_column'] = df['int_column'].astype('int32')
df['float_column'] = df['float_column'].astype('float32')
```

---

### ✅ Use `.loc[]` Instead of `.ix[]` or `.iloc[]` Where Possible

```python
df.loc[df['Age'] > 30, 'Senior'] = True
```

---

### ✅ Chunking Large Files

```python
for chunk in pd.read_csv('data.csv', chunksize=10000):
    process(chunk)
```

---

### ✅ Profile Your DataFrame

```python
!pip install pandas-profiling
from pandas_profiling import ProfileReport

profile = ProfileReport(df)
profile.to_notebook_iframe()
```
