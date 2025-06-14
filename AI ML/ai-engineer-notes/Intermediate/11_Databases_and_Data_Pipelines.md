
## 🗃️ 11. Databases & Data Pipelines – Notes

### 📌 Overview:

Databases and pipelines ensure that **data flows efficiently and reliably** through collection, transformation, storage, and retrieval—essential for production-grade AI systems.

---

### 🧾 11.1 SQL (PostgreSQL / MySQL)

#### ✅ Why SQL?

* Structured data storage
* Fast querying with indexes
* Widely supported in analytics and ML pipelines

#### ✅ Common SQL Operations:

```sql
SELECT * FROM patients WHERE age > 60;
INSERT INTO users (name, age) VALUES ('Hasib', 22);
UPDATE sales SET amount = 500 WHERE id = 1;
```

#### ✅ Libraries:

* `psycopg2` / `sqlalchemy` for PostgreSQL
* `mysql-connector-python` for MySQL

---

### 📦 11.2 NoSQL (MongoDB)

#### ✅ Why NoSQL?

* Flexible schemas (JSON-like)
* Ideal for unstructured/semi-structured data
* Good for logs, documents, user tracking

#### ✅ MongoDB Example (PyMongo):

```python
from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017")
db = client["ml_app"]
db.users.insert_one({"name": "Hasib", "age": 22})
```

---

### 🔁 11.3 Airflow / Prefect Basics

#### ✅ Purpose:

* Automate, schedule, and monitor ML workflows

#### ✅ Airflow:

* DAG (Directed Acyclic Graph) defines tasks & dependencies

```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
```

#### ✅ Prefect:

* Python-native, simpler for beginners

```python
from prefect import flow, task

@task
def clean_data(): ...
@flow
def main_flow(): clean_data()
```

---

### 🔄 11.4 ETL Pipelines

#### ✅ ETL = Extract → Transform → Load

| Step      | Example Tool  | Description                      |
| --------- | ------------- | -------------------------------- |
| Extract   | Python, APIs  | Pull raw data from source        |
| Transform | Pandas, Spark | Clean/aggregate/feature-engineer |
| Load      | SQL, MongoDB  | Store into DB or data warehouse  |

#### ✅ Key Concepts:

* **Batch vs Stream** processing
* **Orchestration**: Ensures the right order and scheduling
* **Monitoring**: Track data freshness, failures, etc.

