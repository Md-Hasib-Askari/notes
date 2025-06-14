## 🟢 **8. Data Transformation**

Data transformation helps reshape and convert raw data into formats better suited for analysis.

---

### ✅ Applying Functions to Data

#### Using `apply()` on Series:

```python
df['Age'].apply(lambda x: x + 1)
```

#### Using `apply()` on DataFrame:

```python
df.apply(lambda x: x.max() - x.min())  # Column-wise by default
df.apply(lambda x: x.max() - x.min(), axis=1)  # Row-wise
```

#### Using `map()` on Series:

```python
df['Gender'].map({'Male': 1, 'Female': 0})
```

#### Using `replace()`:

```python
df['Grade'].replace(['A', 'B', 'C'], [4, 3, 2])
```

---

### ✅ Binning / Discretization

```python
bins = [0, 18, 35, 60, 100]
labels = ['Teen', 'Adult', 'Mid-Age', 'Senior']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)
```

---

### ✅ One-Hot Encoding (Categorical to Numerical)

```python
pd.get_dummies(df, columns=['Gender'])
```

---

### ✅ Sorting Data

```python
df.sort_values('Age')                      # Ascending by Age
df.sort_values('Age', ascending=False)     # Descending
df.sort_values(by=['Gender', 'Age'])       # Multi-column sort
```

---

### ✅ Renaming Axis Labels

```python
df.rename_axis("RowID", axis='index', inplace=True)
df.rename_axis("Attributes", axis='columns', inplace=True)
```

---

### ✅ Changing Column Order

```python
df = df[['Name', 'Gender', 'Age']]
```

