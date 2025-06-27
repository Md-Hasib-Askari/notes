### 🔵 Advanced Level – Topic 11: **Query Optimization**

## ✅ 1. **Understanding the Query Planner**

PostgreSQL has a powerful query planner that decides:

* Which indexes to use
* Whether to use sequential scans or index scans
* Join strategies (nested loop, hash join, merge join)

---

### 🔹 EXPLAIN and EXPLAIN ANALYZE

**EXPLAIN** shows the plan without running the query:

```sql
EXPLAIN SELECT * FROM customers WHERE email = 'a@example.com';
```

**EXPLAIN ANALYZE** actually runs the query and shows timing:

```sql
EXPLAIN ANALYZE SELECT * FROM orders WHERE customer_id = 123;
```

🧠 **Key output terms:**

| Term             | Meaning                           |
| ---------------- | --------------------------------- |
| `Seq Scan`       | Full table scan                   |
| `Index Scan`     | Uses an index                     |
| `Rows`           | Estimated vs actual rows returned |
| `Buffers`        | Pages read from memory or disk    |
| `Cost`           | Estimation of execution cost      |
| `Planning Time`  | Time to create query plan         |
| `Execution Time` | Actual time to run the query      |

---

## ✅ 2. **Common Optimization Techniques**

### 🔹 a. Add Indexes (When Needed)

* Index columns in `WHERE`, `JOIN`, `ORDER BY`, `GROUP BY`

```sql
CREATE INDEX idx_orders_customer_id ON orders(customer_id);
```

### 🔹 b. Use SELECT Specific Columns

❌ Bad:

```sql
SELECT * FROM orders;
```

✅ Good:

```sql
SELECT id, total FROM orders;
```

### 🔹 c. Avoid Repeated Subqueries – Use CTEs or JOINs

---

## ✅ 3. **Vacuuming & Table Maintenance**

PostgreSQL uses **MVCC** (Multi-Version Concurrency Control), which creates old row versions.

### 🔹 VACUUM

Cleans up dead rows:

```sql
VACUUM;
```

### 🔹 VACUUM FULL

Reclaims space and rebuilds the table (locks it!):

```sql
VACUUM FULL;
```

### 🔹 ANALYZE

Updates statistics used by the query planner:

```sql
ANALYZE;
```

🧠 Tip: PostgreSQL has **autovacuum**, but it can be tuned via `postgresql.conf`

---

## ✅ 4. **pg\_stat and Monitoring Views**

PostgreSQL has internal views to inspect performance:

| View                  | Purpose                |
| --------------------- | ---------------------- |
| `pg_stat_activity`    | Shows active sessions  |
| `pg_stat_user_tables` | Per-table stats        |
| `pg_stat_statements`  | Historical query stats |
| `pg_indexes`          | List of all indexes    |

### 🔹 Enable `pg_stat_statements`

```sql
-- In postgresql.conf:
shared_preload_libraries = 'pg_stat_statements'
```

```sql
CREATE EXTENSION pg_stat_statements;
SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 5;
```

---

## ✅ Summary

* Use `EXPLAIN ANALYZE` to understand query behavior and performance bottlenecks.
* Keep tables clean with `VACUUM` and `ANALYZE`.
* Optimize with indexes and smarter query structures.
* Monitor long-running or costly queries with `pg_stat_statements`.

---

# Query Optimization

## Understanding Query Performance

Query optimization is crucial for maintaining good database performance as data grows and query complexity increases. PostgreSQL provides powerful tools to analyze and optimize query execution.

## EXPLAIN and Query Plans

### Basic EXPLAIN
```sql
-- Basic execution plan
EXPLAIN SELECT * FROM customers WHERE city = 'New York';

-- Execution plan with costs
EXPLAIN (COSTS ON) 
SELECT c.name, COUNT(o.order_id) as order_count
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name;

-- Actual execution statistics
EXPLAIN ANALYZE 
SELECT * FROM products 
WHERE price BETWEEN 100 AND 500
ORDER BY price DESC;
```

### Advanced EXPLAIN Options
```sql
-- Detailed analysis with all options
EXPLAIN (
    ANALYZE TRUE,
    VERBOSE TRUE,
    COSTS TRUE,
    BUFFERS TRUE,
    TIMING TRUE,
    SUMMARY TRUE,
    FORMAT JSON
)
SELECT 
    p.name,
    p.price,
    c.name as category_name
FROM products p
JOIN categories c ON p.category_id = c.category_id
WHERE p.price > 100;

-- Text format with buffers (shows I/O)
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT * FROM large_table WHERE indexed_column = 'value';
```

### Reading Query Plans
```sql
-- Understanding plan nodes
EXPLAIN ANALYZE
SELECT 
    c.name,
    AVG(o.total_amount) as avg_order_value
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE c.registration_date >= '2023-01-01'
GROUP BY c.customer_id, c.name
HAVING AVG(o.total_amount) > 100
ORDER BY avg_order_value DESC
LIMIT 10;

/*
Common plan nodes:
- Seq Scan: Full table scan
- Index Scan: Using index to find rows
- Index Only Scan: All data from index
- Bitmap Heap Scan: Index + heap lookup
- Nested Loop: Join algorithm
- Hash Join: Join algorithm
- Merge Join: Join algorithm
- Sort: Sorting operation
- Aggregate: Grouping/aggregation
*/
```

## Index Optimization

### Index Types and Usage
```sql
-- B-tree index (default, most common)
CREATE INDEX idx_customers_email ON customers (email);
CREATE INDEX idx_orders_date ON orders (order_date);

-- Composite index (multiple columns)
CREATE INDEX idx_orders_customer_date ON orders (customer_id, order_date);

-- Partial index (with WHERE clause)
CREATE INDEX idx_active_customers ON customers (customer_id) 
WHERE status = 'active';

-- Expression index
CREATE INDEX idx_customers_lower_email ON customers (LOWER(email));

-- GIN index for array/JSON data
CREATE INDEX idx_products_tags ON products USING gin (tags);

-- Hash index (equality only)
CREATE INDEX idx_products_sku ON products USING hash (sku);
```

### Index Usage Analysis
```sql
-- Check if indexes are being used
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- Find unused indexes
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan
FROM pg_stat_user_indexes
WHERE idx_scan = 0
AND indexname NOT LIKE '%_pkey'; -- Exclude primary keys

-- Index size and usage
SELECT 
    t.tablename,
    indexname,
    c.reltuples AS num_rows,
    pg_size_pretty(pg_relation_size(c.oid)) AS table_size,
    pg_size_pretty(pg_relation_size(i.indexrelid)) AS index_size,
    ROUND(s.idx_scan::numeric / GREATEST(s.seq_scan + s.idx_scan, 1) * 100, 2) AS index_usage_pct
FROM pg_tables t
LEFT JOIN pg_class c ON t.tablename = c.relname
LEFT JOIN pg_index ix ON c.oid = ix.indrelid
LEFT JOIN pg_class i ON i.oid = ix.indexrelid
LEFT JOIN pg_stat_user_indexes s ON i.oid = s.indexrelid
WHERE t.schemaname = 'public'
ORDER BY pg_relation_size(i.indexrelid) DESC;
```

### Covering Indexes
```sql
-- Include additional columns in index
CREATE INDEX idx_orders_customer_include 
ON orders (customer_id) 
INCLUDE (order_date, total_amount);

-- This enables index-only scans for queries like:
EXPLAIN ANALYZE
SELECT order_date, total_amount 
FROM orders 
WHERE customer_id = 123;
```

## Join Optimization

### Join Algorithms
```sql
-- Force different join algorithms for testing
SET enable_nestloop = off;
SET enable_hashjoin = off;
SET enable_mergejoin = on;

-- Test query with different join algorithms
EXPLAIN ANALYZE
SELECT c.name, o.total_amount
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id;

-- Reset to defaults
RESET enable_nestloop;
RESET enable_hashjoin;
RESET enable_mergejoin;
```

### Join Conditions and Statistics
```sql
-- Ensure statistics are up to date
ANALYZE customers;
ANALYZE orders;

-- Good join condition (uses index)
EXPLAIN ANALYZE
SELECT c.name, COUNT(*)
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name;

-- Poor join condition (might not use index efficiently)
EXPLAIN ANALYZE
SELECT c.name, o.total_amount
FROM customers c
JOIN orders o ON CAST(c.customer_id AS TEXT) = CAST(o.customer_id AS TEXT);
```

### Join Order Optimization
```sql
-- PostgreSQL automatically optimizes join order, but you can influence it
-- with table order and join conditions

-- Good: Start with most selective table
EXPLAIN ANALYZE
SELECT *
FROM products p
JOIN categories c ON p.category_id = c.category_id
JOIN suppliers s ON p.supplier_id = s.supplier_id
WHERE p.price > 1000; -- Highly selective condition

-- Consider creating composite indexes for multi-table joins
CREATE INDEX idx_products_category_supplier ON products (category_id, supplier_id);
```

## WHERE Clause Optimization

### Sargable Predicates
```sql
-- Good: Sargable (can use index)
EXPLAIN ANALYZE SELECT * FROM orders WHERE order_date >= '2023-01-01';
EXPLAIN ANALYZE SELECT * FROM customers WHERE customer_id = 123;

-- Bad: Non-sargable (cannot use index efficiently)
EXPLAIN ANALYZE SELECT * FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2023;
EXPLAIN ANALYZE SELECT * FROM customers WHERE customer_id + 1 = 124;

-- Better alternatives
EXPLAIN ANALYZE 
SELECT * FROM orders 
WHERE order_date >= '2023-01-01' AND order_date < '2024-01-01';

EXPLAIN ANALYZE SELECT * FROM customers WHERE customer_id = 123;
```

### LIKE Optimization
```sql
-- Good: Can use index
EXPLAIN ANALYZE SELECT * FROM customers WHERE name LIKE 'John%';

-- Bad: Cannot use regular index
EXPLAIN ANALYZE SELECT * FROM customers WHERE name LIKE '%John%';

-- Better for partial matches: use full-text search or trigram indexes
CREATE EXTENSION pg_trgm;
CREATE INDEX idx_customers_name_trgm ON customers USING gin (name gin_trgm_ops);

-- Now this can use the trigram index
EXPLAIN ANALYZE SELECT * FROM customers WHERE name ILIKE '%john%';
```

### IN vs EXISTS
```sql
-- IN with small list (good)
EXPLAIN ANALYZE
SELECT * FROM products 
WHERE category_id IN (1, 2, 3);

-- IN with subquery (may be inefficient)
EXPLAIN ANALYZE
SELECT * FROM products 
WHERE category_id IN (
    SELECT category_id FROM categories WHERE name LIKE 'Electronics%'
);

-- EXISTS alternative (often better for subqueries)
EXPLAIN ANALYZE
SELECT * FROM products p
WHERE EXISTS (
    SELECT 1 FROM categories c 
    WHERE c.category_id = p.category_id 
    AND c.name LIKE 'Electronics%'
);
```

## Subquery Optimization

### Correlated vs Non-correlated Subqueries
```sql
-- Non-correlated subquery (executed once)
EXPLAIN ANALYZE
SELECT * FROM customers 
WHERE customer_id IN (
    SELECT DISTINCT customer_id FROM orders WHERE order_date >= '2023-01-01'
);

-- Correlated subquery (executed for each row)
EXPLAIN ANALYZE
SELECT * FROM customers c
WHERE EXISTS (
    SELECT 1 FROM orders o 
    WHERE o.customer_id = c.customer_id 
    AND o.order_date >= '2023-01-01'
);

-- Often JOIN is more efficient
EXPLAIN ANALYZE
SELECT DISTINCT c.*
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_date >= '2023-01-01';
```

### Window Functions vs Subqueries
```sql
-- Subquery approach (less efficient)
EXPLAIN ANALYZE
SELECT 
    customer_id,
    order_date,
    total_amount,
    (SELECT AVG(total_amount) FROM orders o2 WHERE o2.customer_id = o1.customer_id) as avg_amount
FROM orders o1;

-- Window function approach (more efficient)
EXPLAIN ANALYZE
SELECT 
    customer_id,
    order_date,
    total_amount,
    AVG(total_amount) OVER (PARTITION BY customer_id) as avg_amount
FROM orders;
```

## Aggregate Optimization

### GROUP BY Optimization
```sql
-- Ensure GROUP BY columns are indexed
CREATE INDEX idx_orders_customer_id ON orders (customer_id);

-- Good: Uses index for grouping
EXPLAIN ANALYZE
SELECT customer_id, COUNT(*), SUM(total_amount)
FROM orders
GROUP BY customer_id;

-- Consider partial indexes for filtered aggregates
CREATE INDEX idx_orders_recent_customer ON orders (customer_id)
WHERE order_date >= '2023-01-01';

EXPLAIN ANALYZE
SELECT customer_id, COUNT(*), SUM(total_amount)
FROM orders
WHERE order_date >= '2023-01-01'
GROUP BY customer_id;
```

### HAVING vs WHERE
```sql
-- Good: Filter before grouping with WHERE
EXPLAIN ANALYZE
SELECT customer_id, COUNT(*) as order_count
FROM orders
WHERE order_date >= '2023-01-01'  -- Filter before grouping
GROUP BY customer_id
HAVING COUNT(*) > 5;  -- Filter after grouping

-- Less efficient: Do all grouping then filter
EXPLAIN ANALYZE
SELECT customer_id, COUNT(*) as order_count
FROM orders
GROUP BY customer_id
HAVING COUNT(*) > 5 AND MIN(order_date) >= '2023-01-01';
```

## Sorting and Limiting

### ORDER BY Optimization
```sql
-- Index supports ordering
CREATE INDEX idx_orders_date_desc ON orders (order_date DESC);

-- Good: Can use index for ordering
EXPLAIN ANALYZE
SELECT * FROM orders
ORDER BY order_date DESC
LIMIT 10;

-- Composite index for ORDER BY with WHERE
CREATE INDEX idx_orders_customer_date ON orders (customer_id, order_date DESC);

EXPLAIN ANALYZE
SELECT * FROM orders
WHERE customer_id = 123
ORDER BY order_date DESC
LIMIT 5;
```

### LIMIT Optimization
```sql
-- LIMIT with ORDER BY (uses index efficiently)
EXPLAIN ANALYZE
SELECT * FROM products
ORDER BY price DESC
LIMIT 10;

-- LIMIT with OFFSET (less efficient for large offsets)
EXPLAIN ANALYZE
SELECT * FROM products
ORDER BY price DESC
LIMIT 10 OFFSET 1000;

-- Better pagination approach
EXPLAIN ANALYZE
SELECT * FROM products
WHERE price < (SELECT price FROM products ORDER BY price DESC LIMIT 1 OFFSET 1000)
ORDER BY price DESC
LIMIT 10;
```

## Statistics and Planner Configuration

### Statistics Collection
```sql
-- Manual statistics update
ANALYZE customers;
ANALYZE orders;

-- Auto-analyze settings
SHOW autovacuum_analyze_threshold;
SHOW autovacuum_analyze_scale_factor;

-- Statistics details
SELECT 
    attname,
    n_distinct,
    correlation,
    most_common_vals,
    most_common_freqs
FROM pg_stats 
WHERE tablename = 'customers' AND schemaname = 'public';
```

### Planner Configuration
```sql
-- View current planner settings
SELECT name, setting, unit, short_desc
FROM pg_settings
WHERE name LIKE '%cost%' OR name LIKE '%enable%'
ORDER BY name;

-- Adjust costs for your hardware
SET seq_page_cost = 1.0;    -- SSD storage
SET random_page_cost = 1.1; -- SSD storage (close to sequential)
SET cpu_tuple_cost = 0.01;
SET cpu_index_tuple_cost = 0.005;
SET cpu_operator_cost = 0.0025;

-- Work memory for sorts and hashes
SET work_mem = '256MB';

-- Effective cache size (total system memory for caching)
SET effective_cache_size = '4GB';
```

## Query Rewriting Techniques

### Eliminating Functions in WHERE
```sql
-- Bad: Function prevents index usage
SELECT * FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2023;

-- Good: Range condition can use index
SELECT * FROM orders 
WHERE order_date >= '2023-01-01' 
AND order_date < '2024-01-01';

-- Bad: UPPER() prevents index usage
SELECT * FROM customers WHERE UPPER(name) = 'JOHN DOE';

-- Good: Use expression index or case-insensitive collation
CREATE INDEX idx_customers_name_ci ON customers (UPPER(name));
-- or
SELECT * FROM customers WHERE name ILIKE 'john doe';
```

### EXISTS vs IN vs JOIN
```sql
-- Test different approaches for the same logical query
-- Approach 1: EXISTS
EXPLAIN ANALYZE
SELECT * FROM customers c
WHERE EXISTS (
    SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id
);

-- Approach 2: IN
EXPLAIN ANALYZE
SELECT * FROM customers
WHERE customer_id IN (SELECT DISTINCT customer_id FROM orders);

-- Approach 3: JOIN
EXPLAIN ANALYZE
SELECT DISTINCT c.*
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id;

-- Choose based on actual performance for your data
```

## Common Performance Anti-patterns

### SELECT * Problems
```sql
-- Bad: Retrieves unnecessary data
SELECT * FROM large_table WHERE id = 123;

-- Good: Select only needed columns
SELECT name, email, created_at FROM large_table WHERE id = 123;

-- Especially important for JOINs
-- Bad:
SELECT * FROM customers c JOIN orders o ON c.customer_id = o.customer_id;

-- Good:
SELECT c.name, c.email, o.order_date, o.total_amount
FROM customers c JOIN orders o ON c.customer_id = o.customer_id;
```

### N+1 Query Problem
```sql
-- Bad: N+1 queries (1 query + N queries for each result)
-- In application code:
-- customers = SELECT * FROM customers LIMIT 10;
-- for each customer:
--   orders = SELECT * FROM orders WHERE customer_id = ?;

-- Good: Single query with JOIN
SELECT 
    c.customer_id,
    c.name,
    o.order_id,
    o.order_date,
    o.total_amount
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE c.status = 'active'
ORDER BY c.customer_id, o.order_date;
```

### Unnecessary DISTINCT
```sql
-- Bad: DISTINCT when not needed (expensive operation)
SELECT DISTINCT customer_id FROM orders; -- If customer_id is always unique per order

-- Good: Remove DISTINCT if uniqueness is guaranteed
SELECT customer_id FROM orders;

-- Or use EXISTS instead of DISTINCT
-- Bad:
SELECT DISTINCT c.customer_id FROM customers c JOIN orders o ON c.customer_id = o.customer_id;

-- Good:
SELECT c.customer_id FROM customers c WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id);
```

## Performance Monitoring Queries

### Top Slow Queries
```sql
-- Requires pg_stat_statements extension
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows,
    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;
```

### Table and Index Usage
```sql
-- Table scan statistics
SELECT 
    schemaname,
    tablename,
    seq_scan,
    seq_tup_read,
    idx_scan,
    idx_tup_fetch,
    n_tup_ins,
    n_tup_upd,
    n_tup_del
FROM pg_stat_user_tables
ORDER BY seq_scan DESC;

-- Index efficiency
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE idx_scan > 0
ORDER BY idx_scan DESC;
```
