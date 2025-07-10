### 🟢 Beginner Level – Topic 2: **Database Basics**

---

#### ✅ PostgreSQL Users and Roles

PostgreSQL uses a **role-based** system:

* A **role** can be a **user** or a **group**.
* Each role can own objects and have privileges.

**🧪 Create a role (user) with login:**

```sql
CREATE ROLE hasib WITH LOGIN PASSWORD 'securepassword';
```

**🧪 Grant privileges to a role:**

```sql
ALTER ROLE hasib CREATEDB;
```

**🧪 List all roles:**

```sql
\du
```

---

#### ✅ Creating and Managing Databases

**🧪 Create a database:**

```sql
CREATE DATABASE shop;
```

**🧪 Connect to a database:**

```bash
psql -d shop
```

or inside `psql`:

```sql
\c shop
```

**🧪 List all databases:**

```sql
\l
```

**🧪 Delete a database:**

```sql
DROP DATABASE shop;
```

---

#### ✅ Basic SQL Operations

##### 1. **Create a table**

```sql
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    price NUMERIC(10,2),
    in_stock BOOLEAN DEFAULT TRUE
);
```

##### 2. **Insert data**

```sql
INSERT INTO products (name, price) VALUES ('Laptop', 999.99);
```

##### 3. **Select data**

```sql
SELECT * FROM products;
```

##### 4. **Update data**

```sql
UPDATE products SET price = 899.99 WHERE name = 'Laptop';
```

##### 5. **Delete data**

```sql
DELETE FROM products WHERE name = 'Laptop';
```

---

#### ✅ Data Types Overview

| Type        | Description                     | Example                 |
| ----------- | ------------------------------- | ----------------------- |
| `INTEGER`   | Whole numbers                   | `42`                    |
| `TEXT`      | Variable-length strings         | `'hello'`               |
| `BOOLEAN`   | True or False                   | `TRUE`, `FALSE`         |
| `DATE`      | Calendar date                   | `'2025-06-27'`          |
| `TIMESTAMP` | Date and time                   | `'2025-06-27 14:00:00'` |
| `NUMERIC`   | Exact decimal                   | `123.45`                |
| `SERIAL`    | Auto-increment integer (pseudo) | —                       |

---

#### ✅ Helpful `psql` Commands

| Command        | Description                   |
| -------------- | ----------------------------- |
| `\q`           | Quit the `psql` shell         |
| `\l`           | List databases                |
| `\c dbname`    | Connect to a database         |
| `\dt`          | List tables in the current DB |
| `\d tablename` | Describe a table              |
| `\du`          | List roles (users)            |

---

### ✅ Summary

* You now know how to create users, databases, and perform basic SQL operations.
* PostgreSQL has rich data types and a powerful CLI (`psql`) for quick admin tasks.
* You should be comfortable inserting, querying, and managing basic records.

---

Would you like to continue to **Topic 3: Basic Schema Design** next?
