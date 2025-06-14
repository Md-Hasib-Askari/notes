## 🟢 **9. Grouping and Aggregation**

Grouping lets you split data into groups, apply operations on each group, and combine the results. This is great for summarizing, transforming, or filtering data.

---

### ✅ `groupby()` Basics

#### Example:

```python
df.groupby('Gender')
```

This returns a **GroupBy object**. To see actual data, you need to **aggregate**.

---

### ✅ Aggregation Functions

#### Single aggregation:

```python
df.groupby('Gender')['Age'].mean()
```

#### Multiple aggregations:

```python
df.groupby('Gender')['Age'].agg(['mean', 'max', 'min'])
```

#### Aggregation with multiple columns:

```python
df.groupby('Gender')[['Age', 'Salary']].mean()
```

---

### ✅ Group by Multiple Columns

```python
df.groupby(['Department', 'Gender'])['Salary'].mean()
```

---

### ✅ `agg()` Custom Aggregation

```python
df.groupby('Gender').agg({
    'Age': 'mean',
    'Salary': ['min', 'max']
})
```

---

### ✅ Using `size()` and `count()`

* `size()` → counts total records per group (including NaNs)
* `count()` → counts non-null values per group

```python
df.groupby('Gender').size()
df.groupby('Gender').count()
```

---

### ✅ Resetting Index After Grouping

```python
grouped = df.groupby('Gender')['Salary'].mean().reset_index()
```

