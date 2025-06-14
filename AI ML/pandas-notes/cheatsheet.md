
## 🐼 **Pandas Cheat Sheet**

---

### 📦 **Import & Setup**

```python
import pandas as pd
import numpy as np
```

---

### 📁 **DataFrame Creation**

```python
df = pd.DataFrame({
    'Name': ['A', 'B'],
    'Age': [25, 30]
})
```

---

### 📄 **Read/Write Files**

```python
pd.read_csv('file.csv')              # Read CSV
df.to_csv('file.csv', index=False)   # Save CSV
```

---

### 📊 **Inspect Data**

```python
df.head(), df.tail()
df.shape, df.columns, df.dtypes
df.describe(), df.info()
```

---

### 🎯 **Selection & Filtering**

```python
df['col'], df[['col1', 'col2']]
df.loc[0], df.loc[:, 'col']
df.iloc[0], df.iloc[0:5, 1:3]
df[df['Age'] > 25]
df.query('Age > 25')
```

---

### 🧹 **Cleaning**

```python
df.dropna(), df.fillna(0)
df.drop(['col'], axis=1)
df.rename(columns={'old': 'new'}, inplace=True)
df.duplicated(), df.drop_duplicates()
df.astype({'col': 'int32'})
```

---

### 🔄 **Modification**

```python
df['new'] = df['old'] * 2
df['col'].replace({'A': 1, 'B': 2})
df['col'].apply(lambda x: x.upper())
```

---

### 📆 **DateTime**

```python
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df.set_index('date', inplace=True)
```

---

### 📚 **Group & Aggregate**

```python
df.groupby('col').mean()
df.groupby(['col1', 'col2'])['val'].agg(['mean', 'sum'])
df['col'].value_counts()
```

---

### 🔁 **Merge / Join / Concat**

```python
pd.merge(df1, df2, on='key', how='inner')
pd.concat([df1, df2], axis=0)
df1.join(df2, how='left')
```

---

### 🔄 **Reshape**

```python
df.pivot(index='A', columns='B', values='C')
df.melt(id_vars=['id'], var_name='variable')
```

---

### 📈 **Visualization**

```python
df.plot(kind='line')
df['col'].plot(kind='hist')
df[['a', 'b']].plot(kind='box')
```

---

### ⚙️ **Performance**

```python
df.memory_usage(deep=True)
df['col'] = df['col'].astype('category')
df.query('col > 5')
```

