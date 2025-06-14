## 🟢 **10. Merging, Joining, and Concatenation**

These operations help you combine multiple DataFrames—like SQL joins, vertical stacking, or appending new columns.

---

### ✅ `concat()` – Concatenating DataFrames

#### Stack vertically (like UNION ALL in SQL):

```python
pd.concat([df1, df2])
```

#### Stack horizontally (aligns by index):

```python
pd.concat([df1, df2], axis=1)
```

#### Ignore index:

```python
pd.concat([df1, df2], ignore_index=True)
```

---

### ✅ `merge()` – Like SQL JOINs

```python
pd.merge(df1, df2, on='id')
```

#### Types of joins:

```python
pd.merge(df1, df2, on='id', how='inner')   # Default
pd.merge(df1, df2, on='id', how='left')
pd.merge(df1, df2, on='id', how='right')
pd.merge(df1, df2, on='id', how='outer')
```

#### Merge on different column names:

```python
pd.merge(df1, df2, left_on='user_id', right_on='id')
```

---

### ✅ `join()` – Join on Index

```python
df1.join(df2, how='left')
```

You can also join on keys:

```python
df1.set_index('id').join(df2.set_index('id'), how='inner')
```

---

### ✅ Combining with `append()` *(deprecated in future pandas)*

```python
df1.append(df2, ignore_index=True)
```

Prefer `pd.concat()` over `append()` moving forward.

