## 🟢 **12. Time Series Handling**

Pandas makes working with date and time data seamless—whether you're parsing timestamps, resampling, or indexing by time.

---

### ✅ Converting to DateTime

```python
df['Date'] = pd.to_datetime(df['Date'])
```

---

### ✅ Extracting Date Components

```python
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.day_name()
```

---

### ✅ Setting DateTime as Index

```python
df.set_index('Date', inplace=True)
```

---

### ✅ Date Range Creation

```python
pd.date_range(start='2024-01-01', end='2024-01-10')
pd.date_range(start='2024-01-01', periods=5, freq='D')  # D = day, M = month, H = hour, etc.
```

---

### ✅ Resampling (Time-based Grouping)

#### Resample to monthly frequency:

```python
df.resample('M').sum()
```

#### Other frequencies:

* `'D'` – Day
* `'W'` – Week
* `'M'` – Month
* `'Y'` – Year
* `'H'` – Hour
* `'T'` – Minute

```python
df.resample('W').mean()
```

---

### ✅ Rolling Statistics (Moving Averages)

```python
df['Sales'].rolling(window=3).mean()  # 3-period rolling average
```

---

### ✅ Shifting Data (Lag or Lead)

```python
df['Lag'] = df['Sales'].shift(1)     # Previous row
df['Lead'] = df['Sales'].shift(-1)   # Next row
```

