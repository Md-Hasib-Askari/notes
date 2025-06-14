## 🟢 **4. Importing and Exporting Data**

### ✅ Reading Data from Files

#### CSV File

```python
df = pd.read_csv('file.csv')
```

**Options:**

```python
pd.read_csv('file.csv', delimiter=',', header=0, index_col=0)
```

#### Excel File

```python
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')
```

#### JSON File

```python
df = pd.read_json('file.json')
```

#### SQL Query (requires DB connection)

```python
import sqlite3

conn = sqlite3.connect('data.db')
df = pd.read_sql_query("SELECT * FROM table_name", conn)
```

---

### ✅ Writing Data to Files

#### To CSV

```python
df.to_csv('output.csv', index=False)
```

#### To Excel

```python
df.to_excel('output.xlsx', index=False)
```

#### To JSON

```python
df.to_json('output.json', orient='records')
```

---

### ✅ Common Parameters in `read_csv()`:

* `sep=','`: Define a different delimiter (e.g., tab `\t`)
* `header=None`: Use when file has no headers
* `names=['A', 'B']`: Manually set column names
* `na_values=['NA', '?']`: Set custom missing value identifiers
* `usecols=['col1', 'col2']`: Load only specific columns
* `nrows=100`: Load first 100 rows
* `skiprows=1`: Skip first row

---

### ✅ Useful Tip: Preview File Without Loading All

```python
pd.read_csv('file.csv', nrows=5)   # Only first 5 rows
```
