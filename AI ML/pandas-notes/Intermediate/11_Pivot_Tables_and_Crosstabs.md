## 🟢 **11. Pivot Tables and Crosstabs**

These are powerful tools for reshaping and summarizing data, similar to Excel pivot tables.

---

### ✅ `pivot_table()` – Advanced Grouping

#### Basic usage:

```python
df.pivot_table(values='Sales', index='Region', columns='Product')
```

#### Add aggregation:

```python
df.pivot_table(values='Sales', index='Region', columns='Product', aggfunc='sum')
```

#### Multiple aggregation functions:

```python
df.pivot_table(values='Sales', index='Region', columns='Product', aggfunc=['sum', 'mean'])
```

#### Fill missing values:

```python
df.pivot_table(values='Sales', index='Region', columns='Product', fill_value=0)
```

---

### ✅ `crosstab()` – Frequency Tables

#### Basic crosstab:

```python
pd.crosstab(df['Gender'], df['Department'])
```

#### Normalize by rows or columns:

```python
pd.crosstab(df['Gender'], df['Department'], normalize='index')  # Row %
pd.crosstab(df['Gender'], df['Department'], normalize='columns')  # Column %
```

#### Add margins (totals):

```python
pd.crosstab(df['Gender'], df['Department'], margins=True)
```

#### With multiple variables:

```python
pd.crosstab([df['Gender'], df['Region']], df['Department'])
```

