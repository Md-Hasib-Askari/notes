# pandas-notes

### 🟢 **Beginner Level**

#### 1. **Introduction to Pandas**

* What is Pandas?
* Series vs DataFrame
* Installing Pandas (`pip install pandas`)
* Importing and setting up (`import pandas as pd`)

#### 2. **Series**

* Creating Series (from list, dict, scalar)
* Indexing and slicing
* Basic operations (sum, mean, etc.)
* `.head()`, `.tail()`

#### 3. **DataFrame Basics**

* Creating DataFrames (from list of dicts, dict of lists, CSV, etc.)
* Accessing columns and rows (`df['col']`, `df.loc[]`, `df.iloc[]`)
* Adding/removing columns
* Renaming columns

#### 4. **Importing and Exporting Data**

* Reading from CSV, Excel, JSON

  * `pd.read_csv()`, `read_excel()`, `read_json()`
* Exporting to file formats

  * `.to_csv()`, `.to_excel()`, `.to_json()`

#### 5. **Data Inspection**

* `.info()`, `.describe()`, `.shape`, `.columns`, `.dtypes`
* Checking for nulls: `isnull()`, `notnull()`
* Basic filtering

---

### 🟡 **Intermediate Level**

#### 6. **Data Selection and Filtering**

* Conditional filtering
* `.isin()`, `.between()`, `.query()`
* `loc[]` vs `iloc[]` in depth

#### 7. **Handling Missing Data**

* Detecting missing values
* Filling (`fillna()`), forward/backward fill (`ffill()`, `bfill()`)
* Dropping (`dropna()`)

#### 8. **Data Cleaning & Transformation**

* Renaming columns
* Replacing values (`replace()`)
* String operations with `.str`
* Type conversion (`astype()`)

#### 9. **Sorting and Ranking**

* Sorting values (`sort_values()`)
* Sorting index (`sort_index()`)
* Ranking (`rank()`)

#### 10. **Grouping and Aggregating**

* `groupby()` usage
* Aggregations (`sum()`, `mean()`, `agg()`, `transform()`)
* Multi-level groupby

#### 11. **Merging, Joining, and Concatenation**

* `concat()`, `merge()`, `join()`
* Inner, outer, left, right joins
* Combining data from multiple sources

#### 12. **Date and Time Handling**

* Parsing dates (`pd.to_datetime()`)
* Extracting date parts (year, month, etc.)
* Time series index and resampling

---

### 🔴 **Advanced Level**

#### 13. **MultiIndex and Hierarchical Data**

* Creating and manipulating MultiIndex
* Indexing/slicing MultiIndex DataFrames

#### 14. **Pivot Tables and Crosstab**

* Creating pivot tables (`pivot_table()`)
* Frequency tables using `crosstab()`

#### 15. **Window Functions**

* Rolling statistics (`rolling()`, `expanding()`)
* Moving averages, cumulative stats

#### 16. **Advanced Aggregation with `agg()` and `apply()`**

* Row/column-wise operations
* Lambda functions with `apply()`
* Custom aggregation functions

#### 17. **Categorical Data**

* Creating/using `Categorical` data
* Reducing memory usage

#### 18. **Performance Optimization**

* Working with large datasets
* Memory usage: `memory_usage()`
* Chunking large files
* Vectorized operations vs. looping

#### 19. **Visualization with Pandas**

* Built-in plotting: `.plot()`
* Histograms, boxplots, line, bar, scatter
* Integration with matplotlib/seaborn

#### 20. **Integration with Other Libraries**

* Using Pandas with NumPy, Matplotlib, Seaborn, Scikit-learn
* Exporting data for ML use