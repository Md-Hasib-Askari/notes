## 🟢 **Beginner Level – Fundamentals**

### 1. **Introduction to PostgreSQL**

* What is PostgreSQL and its use cases
* Advantages over MySQL, SQLite
* Installation on Linux/Windows/Mac
* Basic configuration (`postgresql.conf`, `pg_hba.conf`)

### 2. **Database Basics**

* Creating databases and users
* `psql` CLI tool
* Basic SQL commands: `SELECT`, `INSERT`, `UPDATE`, `DELETE`
* Data types: `INTEGER`, `TEXT`, `BOOLEAN`, `DATE`, `TIMESTAMP`

### 3. **Basic Schema Design**

* Creating tables, constraints (`PRIMARY KEY`, `UNIQUE`, `NOT NULL`)
* Foreign keys and basic normalization
* Simple ER diagram understanding

### 4. **Simple Queries**

* `WHERE`, `ORDER BY`, `LIMIT`, `OFFSET`
* `LIKE`, `ILIKE`, `IN`, `BETWEEN`
* Aggregate functions: `COUNT()`, `SUM()`, `AVG()`

### 5. **Tools**

* `psql`, pgAdmin
* DBeaver, TablePlus (GUI tools)

---

## 🟡 **Intermediate Level – Performance, Joins, Functions**

### 6. **Advanced Queries**

* `JOIN`: `INNER`, `LEFT`, `RIGHT`, `FULL`
* `GROUP BY`, `HAVING`
* Subqueries, CTEs (`WITH` clause)

### 7. **Data Modeling & Constraints**

* Indexing: `B-TREE`, `GIN`, `GiST`
* Composite keys
* Default values, `CHECK` constraints
* ENUM types and custom types

### 8. **Views and Materialized Views**

* Creating and using views
* Refreshing materialized views
* Pros and cons

### 9. **Functions and Stored Procedures**

* PL/pgSQL basics
* Writing custom functions
* `RETURNS`, `LANGUAGE plpgsql`
* Using parameters

### 10. **Transactions and Locks**

* `BEGIN`, `COMMIT`, `ROLLBACK`
* Isolation levels
* Deadlocks, optimistic vs pessimistic locking

---

## 🔵 **Advanced Level – Indexing, Optimization, Security**

### 11. **Query Optimization**

* `EXPLAIN`, `EXPLAIN ANALYZE`
* Vacuuming: `VACUUM`, `ANALYZE`, autovacuum
* Stats: `pg_stat_user_tables`, `pg_stat_activity`

### 12. **Advanced Indexing**

* Partial indexes, expression indexes
* Covering indexes
* Index usage patterns

### 13. **Security and Access Control**

* Roles and privileges
* Row-level security
* SSL and encrypted connections
* Secure password storage (`SCRAM`, MD5)

### 14. **Data Import/Export**

* `COPY`, `\copy`
* CSV, JSON, binary formats
* Backups: `pg_dump`, `pg_restore`

---

## 🔴 **Expert Level – Scaling, Replication, Extensions**

### 15. **Extensions & Full-text Search**

* `pg_trgm`, `fuzzystrmatch`, `uuid-ossp`
* `PostGIS` for GIS data
* Full-text search: `tsvector`, `tsquery`

### 16. **Replication and High Availability**

* Streaming replication
* Logical replication
* WAL archiving
* Failover setups (Patroni, repmgr)

### 17. **Partitioning**

* Declarative partitioning (range, list, hash)
* Performance tuning with partitions

### 18. **Connection Pooling & Parallelism**

* Using `PgBouncer`, `pgpool-II`
* Parallel queries and tuning for performance

### 19. **Monitoring and Logging**

* `pg_stat_statements`, `pgBadger`
* Logging slow queries
* Alerting & monitoring tools: Prometheus + Grafana

### 20. **PostgreSQL in Production**

* Schema migrations (Alembic, Flyway, Liquibase)
* CI/CD integration
* Handling large datasets (TB-scale)

---

## 🧪 Bonus: Practice and Projects

* Build a blog backend with PostgreSQL + Python/Django or Node/Express
* Optimize a slow query-heavy analytics dashboard
* Set up master-slave replication with failover
* Explore temporal tables and time-series modeling
