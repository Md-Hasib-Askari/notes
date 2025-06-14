
## 🟢 **1. Introduction to Pandas**

### ✅ What is Pandas?

* **Pandas** is a powerful open-source Python library used for data manipulation and analysis.
* Built on top of **NumPy**, it provides two primary data structures:

  * **Series**: One-dimensional labeled array.
  * **DataFrame**: Two-dimensional labeled data structure (like a table in SQL or Excel).

---

### ✅ Why Use Pandas?

* Handle and clean data efficiently.
* Load and export datasets easily (CSV, Excel, JSON, SQL).
* Perform operations like filtering, grouping, merging, and aggregation.
* Built-in support for time series analysis.

---

### ✅ Installing Pandas

```bash
pip install pandas
```

---

### ✅ Importing Pandas

```python
import pandas as pd
```

* Common convention is to use `pd` as an alias for `pandas`.

---

### ✅ Basic Example

```python
import pandas as pd

data = {'Name': ['Hasib', 'Ayesha'], 'Age': [21, 22]}
df = pd.DataFrame(data)

print(df)
```

**Output:**

```
    Name  Age
0  Hasib   21
1 Ayesha   22
```

