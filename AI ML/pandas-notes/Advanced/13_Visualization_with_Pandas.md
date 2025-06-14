## 🟢 **13. Visualization with Pandas**

Pandas has built-in support for quick visualizations using **Matplotlib** under the hood. Great for quick data exploration.

---

### ✅ Setup (Optional)

```python
import matplotlib.pyplot as plt
import seaborn as sns  # for more advanced visuals
```

---

### ✅ Basic Plotting with Pandas

#### Line plot (default):

```python
df['Sales'].plot()
```

#### Bar plot:

```python
df['Sales'].plot(kind='bar')
df.plot(x='Product', y='Sales', kind='bar')
```

#### Histogram:

```python
df['Age'].plot(kind='hist', bins=10)
```

#### Box plot:

```python
df[['Math', 'Science']].plot(kind='box')
```

#### Pie chart:

```python
df['Category'].value_counts().plot(kind='pie', autopct='%1.1f%%')
```

#### Area chart:

```python
df[['Sales1', 'Sales2']].plot(kind='area')
```

---

### ✅ Customizing Plots

```python
df['Sales'].plot(title='Monthly Sales', figsize=(10,5), color='green', grid=True)
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.show()
```

---

### ✅ Correlation Heatmap (with Seaborn)

```python
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```

