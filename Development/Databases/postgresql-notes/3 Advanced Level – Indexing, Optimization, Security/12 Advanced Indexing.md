### 🔵 Advanced Level – Topic 12: **Advanced Indexing**

## ✅ 1. **Partial Indexes**

Index **only a subset** of data using a `WHERE` clause.

### 🔹 Use Case:

Indexing only **active users** or **non-null fields**

```sql
CREATE INDEX idx_active_customers ON customers(email)
WHERE is_active = TRUE;
```

> ✅ Smaller size
> ✅ Faster lookups (on WHERE filter)
> ❌ Won’t be used for other values (inactive users)

---

## ✅ 2. **Expression Indexes**

Indexes the **result of an expression**, not just a column.

### 🔹 Use Case:

Case-insensitive email search

```sql
CREATE INDEX idx_lower_email ON customers(LOWER(email));
```

Then use:

```sql
SELECT * FROM customers WHERE LOWER(email) = 'a@example.com';
```

---

## ✅ 3. **Covering Indexes (INCLUDE Clause)**

PostgreSQL 11+ allows you to **include non-key columns** in the index to **avoid table lookups**.

### 🔹 Use Case:

Index used for filtering and selecting both `status` and `created_at`.

```sql
CREATE INDEX idx_orders_status ON orders(status) INCLUDE (created_at);
```

---

## ✅ 4. **Multi-Column Indexes**

When queries filter by **more than one column**, order matters.

```sql
CREATE INDEX idx_orders_customer_date ON orders(customer_id, order_date);
```

* Used for:

  * `WHERE customer_id = ?`
  * `WHERE customer_id = ? AND order_date = ?`
* **Not** used for just `order_date = ?` (unless first column matches)

---

## ✅ 5. **Index-only Scans**

Happen when:

* All selected columns are **in the index**
* Table is **"visible"** (i.e., no recent changes)

Use `EXPLAIN ANALYZE` to see if:

```
Index Only Scan using ...
```

---

## ✅ 6. **Unique Indexes vs Constraints**

```sql
-- Constraint (easier to manage, more expressive)
ALTER TABLE users ADD CONSTRAINT unique_email UNIQUE(email);

-- Index (can use partials, expressions)
CREATE UNIQUE INDEX unique_lower_email ON users(LOWER(email));
```

---

## ✅ 7. **Monitoring Index Usage**

```sql
SELECT relname AS table,
       indexrelname AS index,
       idx_scan AS times_used
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

> 🔍 Drop unused indexes if they slow down INSERT/UPDATE/DELETE operations.

---

## ✅ Summary

| Index Type       | Best For                                     |
| ---------------- | -------------------------------------------- |
| Partial Index    | Subset of rows                               |
| Expression Index | Searching computed expressions               |
| INCLUDE Index    | Covering columns without changing sort order |
| Multi-column     | WHERE filters with multiple columns          |
| Unique Index     | Enforcing uniqueness                         |

---

# Advanced Indexing

## Index Types Deep Dive

### B-tree Indexes (Default)
B-tree indexes are the most common and versatile index type in PostgreSQL, suitable for equality and range queries.

```sql
-- Basic B-tree index
CREATE INDEX idx_customers_email ON customers (email);

-- Composite B-tree index (column order matters!)
CREATE INDEX idx_orders_customer_date ON orders (customer_id, order_date);

-- B-tree with different sort orders
CREATE INDEX idx_products_price_desc ON products (price DESC);
CREATE INDEX idx_orders_mixed_sort ON orders (customer_id ASC, order_date DESC);

-- B-tree supports various operators
EXPLAIN ANALYZE SELECT * FROM customers WHERE email = 'john@example.com';     -- =
EXPLAIN ANALYZE SELECT * FROM customers WHERE email > 'john@example.com';     -- >
EXPLAIN ANALYZE SELECT * FROM customers WHERE email BETWEEN 'a' AND 'n';     -- BETWEEN
EXPLAIN ANALYZE SELECT * FROM customers WHERE email IS NULL;                  -- IS NULL
```

### Hash Indexes
Hash indexes are optimized for equality comparisons only.

```sql
-- Hash index (equality only)
CREATE INDEX idx_products_sku_hash ON products USING hash (sku);

-- Good for exact matches
EXPLAIN ANALYZE SELECT * FROM products WHERE sku = 'ABC123';

-- Cannot be used for range queries
EXPLAIN ANALYZE SELECT * FROM products WHERE sku > 'ABC123'; -- Won't use hash index

-- Hash indexes are smaller and faster for equality, but less versatile
SELECT pg_size_pretty(pg_relation_size('idx_products_sku_hash'));
```

### GIN (Generalized Inverted Index)
GIN indexes are designed for composite values like arrays, JSON, and full-text search.

```sql
-- GIN index for arrays
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    name TEXT,
    tags TEXT[]
);

CREATE INDEX idx_products_tags_gin ON products USING gin (tags);

-- Array operations that use GIN
EXPLAIN ANALYZE SELECT * FROM products WHERE tags @> ARRAY['electronics']; -- contains
EXPLAIN ANALYZE SELECT * FROM products WHERE tags && ARRAY['sale', 'new']; -- overlaps
EXPLAIN ANALYZE SELECT * FROM products WHERE ARRAY['mobile'] <@ tags;      -- contained by

-- GIN index for JSONB
CREATE TABLE user_profiles (
    user_id SERIAL PRIMARY KEY,
    profile_data JSONB
);

CREATE INDEX idx_user_profiles_gin ON user_profiles USING gin (profile_data);

-- JSONB operations using GIN
EXPLAIN ANALYZE SELECT * FROM user_profiles WHERE profile_data @> '{"city": "New York"}';
EXPLAIN ANALYZE SELECT * FROM user_profiles WHERE profile_data ? 'phone';
EXPLAIN ANALYZE SELECT * FROM user_profiles WHERE profile_data ?| array['email', 'phone'];
```

### GiST (Generalized Search Tree)
GiST indexes support various data types and are extensible.

```sql
-- GiST for geometric data
CREATE TABLE locations (
    location_id SERIAL PRIMARY KEY,
    name TEXT,
    coordinates POINT
);

CREATE INDEX idx_locations_gist ON locations USING gist (coordinates);

-- Geometric queries
EXPLAIN ANALYZE SELECT * FROM locations WHERE coordinates <-> point(0,0) < 10;

-- GiST for text similarity (with pg_trgm extension)
CREATE EXTENSION pg_trgm;
CREATE INDEX idx_customers_name_gist ON customers USING gist (name gist_trgm_ops);

-- Similarity searches
EXPLAIN ANALYZE SELECT * FROM customers WHERE name % 'John Smith'; -- similar to
```

### SP-GiST (Space-Partitioned GiST)
SP-GiST is optimized for data that can be partitioned by space.

```sql
-- SP-GiST for ranges
CREATE TABLE price_ranges (
    range_id SERIAL PRIMARY KEY,
    price_range int4range
);

CREATE INDEX idx_price_ranges_spgist ON price_ranges USING spgist (price_range);

-- Range queries
EXPLAIN ANALYZE SELECT * FROM price_ranges WHERE price_range @> 150;
EXPLAIN ANALYZE SELECT * FROM price_ranges WHERE price_range && int4range(100, 200);
```

### BRIN (Block Range Index)
BRIN indexes are very compact and suitable for large tables with natural ordering.

```sql
-- BRIN index for time-series data
CREATE TABLE sensor_data (
    sensor_id INTEGER,
    timestamp TIMESTAMP,
    value DECIMAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert data in chronological order for best BRIN performance
CREATE INDEX idx_sensor_data_timestamp_brin ON sensor_data USING brin (timestamp);

-- BRIN is very space-efficient
SELECT pg_size_pretty(pg_relation_size('idx_sensor_data_timestamp_brin'));

-- Good for range queries on naturally ordered data
EXPLAIN ANALYZE 
SELECT * FROM sensor_data 
WHERE timestamp BETWEEN '2023-01-01' AND '2023-01-31';

-- BRIN maintenance
SELECT brin_summarize_new_values('idx_sensor_data_timestamp_brin');
```

## Partial Indexes

### Conditional Indexes
```sql
-- Index only active records
CREATE INDEX idx_customers_active ON customers (customer_id) 
WHERE status = 'active';

-- Index only recent orders
CREATE INDEX idx_orders_recent ON orders (customer_id, order_date)
WHERE order_date >= '2023-01-01';

-- Index only non-null values
CREATE INDEX idx_customers_phone ON customers (phone)
WHERE phone IS NOT NULL;

-- Complex conditions
CREATE INDEX idx_orders_large_pending ON orders (order_date)
WHERE status = 'pending' AND total_amount > 1000;

-- Partial index usage
EXPLAIN ANALYZE 
SELECT * FROM customers 
WHERE status = 'active' AND customer_id = 123; -- Uses partial index

EXPLAIN ANALYZE 
SELECT * FROM customers 
WHERE customer_id = 123; -- May not use partial index
```

### Partial Index Benefits
```sql
-- Compare sizes: full vs partial index
CREATE INDEX idx_orders_full ON orders (customer_id);
CREATE INDEX idx_orders_partial ON orders (customer_id) WHERE status = 'active';

SELECT 
    'Full index' as type,
    pg_size_pretty(pg_relation_size('idx_orders_full')) as size
UNION ALL
SELECT 
    'Partial index' as type,
    pg_size_pretty(pg_relation_size('idx_orders_partial')) as size;
```

## Expression Indexes

### Function-based Indexes
```sql
-- Index on function result
CREATE INDEX idx_customers_lower_email ON customers (LOWER(email));

-- Case-insensitive email searches
EXPLAIN ANALYZE 
SELECT * FROM customers 
WHERE LOWER(email) = 'john@example.com';

-- Index on expression
CREATE INDEX idx_orders_year ON orders (EXTRACT(YEAR FROM order_date));

-- Year-based queries
EXPLAIN ANALYZE 
SELECT * FROM orders 
WHERE EXTRACT(YEAR FROM order_date) = 2023;

-- Complex expressions
CREATE INDEX idx_products_discounted_price ON products ((price * 0.9))
WHERE discount_eligible = true;

-- Mathematical expressions
CREATE INDEX idx_customers_full_name ON customers ((first_name || ' ' || last_name));
```

### Text Processing Indexes
```sql
-- Trigram indexes for fuzzy matching
CREATE EXTENSION pg_trgm;
CREATE INDEX idx_products_name_trgm ON products USING gin (name gin_trgm_ops);

-- Fuzzy searches
EXPLAIN ANALYZE SELECT * FROM products WHERE name % 'laptp'; -- Similar to 'laptop'
EXPLAIN ANALYZE SELECT * FROM products WHERE name ILIKE '%phone%';

-- Full-text search indexes
CREATE INDEX idx_products_fts ON products USING gin (to_tsvector('english', name || ' ' || description));

-- Full-text queries
EXPLAIN ANALYZE 
SELECT * FROM products 
WHERE to_tsvector('english', name || ' ' || description) @@ to_tsquery('smartphone & wireless');
```

## Multicolumn Indexes

### Column Order Optimization
```sql
-- Column order matters for multicolumn indexes
-- High selectivity columns should come first

-- Good: status is highly selective, date provides ordering
CREATE INDEX idx_orders_status_date ON orders (status, order_date);

-- Less optimal: date first, then status
CREATE INDEX idx_orders_date_status ON orders (order_date, status);

-- Test different column orders
EXPLAIN ANALYZE 
SELECT * FROM orders 
WHERE status = 'pending' 
ORDER BY order_date DESC;

-- Prefix matching works with multicolumn indexes
-- This can use idx_orders_status_date
EXPLAIN ANALYZE SELECT * FROM orders WHERE status = 'pending';

-- This cannot efficiently use idx_orders_status_date
EXPLAIN ANALYZE SELECT * FROM orders WHERE order_date > '2023-01-01';
```

### Covering Indexes (INCLUDE clause)
```sql
-- Include non-key columns for index-only scans
CREATE INDEX idx_orders_customer_include ON orders (customer_id) 
INCLUDE (order_date, total_amount, status);

-- This query can be answered entirely from the index
EXPLAIN ANALYZE 
SELECT order_date, total_amount, status 
FROM orders 
WHERE customer_id = 123;

-- Compare with and without INCLUDE
CREATE INDEX idx_orders_customer_basic ON orders (customer_id);

-- Size comparison
SELECT 
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) as size
FROM pg_indexes 
WHERE tablename = 'orders' 
AND indexname IN ('idx_orders_customer_include', 'idx_orders_customer_basic');
```

## Index Maintenance and Monitoring

### Index Usage Statistics
```sql
-- Check index usage
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
    pg_size_pretty(pg_relation_size(i.indexrelid)) as size
FROM pg_stat_user_indexes s
JOIN pg_index i ON s.indexrelid = i.indexrelid
WHERE s.idx_scan = 0
AND NOT i.indisunique -- Keep unique indexes
AND NOT i.indisprimary -- Keep primary key indexes
ORDER BY pg_relation_size(i.indexrelid) DESC;

-- Index hit ratio
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    round(100.0 * idx_scan / GREATEST(idx_scan + seq_scan, 1), 2) as index_usage_pct
FROM pg_stat_user_indexes s
JOIN pg_stat_user_tables t ON s.relid = t.relid
ORDER BY index_usage_pct DESC;
```

### Index Bloat Detection
```sql
-- Estimate index bloat
WITH index_bloat AS (
    SELECT 
        schemaname,
        tablename,
        indexname,
        pg_relation_size(i.indexrelid) as index_size,
        CASE 
            WHEN pg_relation_size(i.indexrelid) = 0 THEN 0
            ELSE round(100.0 * (pg_relation_size(i.indexrelid) - 
                 pg_stat_get_live_tuples(i.indexrelid) * 
                 pg_catalog.current_setting('block_size')::int / 8) / 
                 pg_relation_size(i.indexrelid), 2)
        END as bloat_pct
    FROM pg_stat_user_indexes s
    JOIN pg_index i ON s.indexrelid = i.indexrelid
    WHERE pg_relation_size(i.indexrelid) > 0
)
SELECT *
FROM index_bloat
WHERE bloat_pct > 20
ORDER BY index_size DESC;

-- Rebuild bloated indexes
REINDEX INDEX idx_bloated_index;
-- Or for entire table
REINDEX TABLE table_with_bloated_indexes;
```

### Index Maintenance Commands
```sql
-- Rebuild specific index
REINDEX INDEX idx_customers_email;

-- Rebuild all indexes on a table
REINDEX TABLE customers;

-- Rebuild all indexes in schema
REINDEX SCHEMA public;

-- Concurrent reindex (PostgreSQL 12+)
REINDEX INDEX CONCURRENTLY idx_customers_email;

-- Analyze after reindex
ANALYZE customers;
```

## Advanced Index Strategies

### Conditional Unique Indexes
```sql
-- Allow multiple NULL values, but enforce uniqueness for non-NULL
CREATE UNIQUE INDEX idx_users_email_unique ON users (email)
WHERE email IS NOT NULL;

-- Partial unique constraint
CREATE UNIQUE INDEX idx_active_user_email ON users (email)
WHERE status = 'active';

-- This allows the same email for inactive users
INSERT INTO users (email, status) VALUES ('test@example.com', 'active');   -- OK
INSERT INTO users (email, status) VALUES ('test@example.com', 'inactive'); -- OK  
INSERT INTO users (email, status) VALUES ('test@example.com', 'active');   -- ERROR
```

### Indexes for JSON Queries
```sql
-- JSON data structure
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    name TEXT,
    attributes JSONB
);

-- GIN index for general JSON queries
CREATE INDEX idx_products_attributes_gin ON products USING gin (attributes);

-- Specific JSON path indexes
CREATE INDEX idx_products_brand ON products ((attributes->>'brand'));
CREATE INDEX idx_products_price ON products (((attributes->>'price')::numeric));

-- Query patterns that use these indexes
EXPLAIN ANALYZE SELECT * FROM products WHERE attributes @> '{"brand": "Apple"}';
EXPLAIN ANALYZE SELECT * FROM products WHERE attributes->>'brand' = 'Apple';
EXPLAIN ANALYZE SELECT * FROM products WHERE (attributes->>'price')::numeric > 500;

-- JSON path indexes for nested data
CREATE INDEX idx_products_specs_cpu ON products ((attributes#>>'{specs,cpu}'));
```

### Indexes for Time-Series Data
```sql
-- Time-series table
CREATE TABLE metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    metric_name TEXT NOT NULL,
    value DOUBLE PRECISION,
    tags JSONB
);

-- Compound index for time-series queries
CREATE INDEX idx_metrics_name_time ON metrics (metric_name, timestamp DESC);

-- BRIN index for timestamp (if data is naturally ordered)
CREATE INDEX idx_metrics_timestamp_brin ON metrics USING brin (timestamp);

-- GIN index for tag queries
CREATE INDEX idx_metrics_tags_gin ON metrics USING gin (tags);

-- Partial indexes for active metrics
CREATE INDEX idx_metrics_recent ON metrics (metric_name, timestamp)
WHERE timestamp >= NOW() - INTERVAL '7 days';

-- Efficient time-series queries
EXPLAIN ANALYZE 
SELECT * FROM metrics 
WHERE metric_name = 'cpu_usage' 
AND timestamp >= NOW() - INTERVAL '1 hour'
ORDER BY timestamp DESC;
```

### Multi-dimensional Indexes
```sql
-- Geographic data
CREATE TABLE stores (
    store_id SERIAL PRIMARY KEY,
    name TEXT,
    location POINT
);

-- GiST index for spatial queries
CREATE INDEX idx_stores_location_gist ON stores USING gist (location);

-- Range data
CREATE TABLE reservations (
    reservation_id SERIAL PRIMARY KEY,
    room_id INTEGER,
    time_range TSTZRANGE
);

-- GiST index for range queries
CREATE INDEX idx_reservations_range_gist ON reservations USING gist (time_range);

-- Exclude overlapping reservations
ALTER TABLE reservations 
ADD CONSTRAINT no_overlapping_reservations 
EXCLUDE USING gist (room_id WITH =, time_range WITH &&);
```

## Index Performance Tuning

### Index Scan vs Index Only Scan
```sql
-- Regular index scan (requires heap lookup)
CREATE INDEX idx_orders_customer ON orders (customer_id);

EXPLAIN ANALYZE 
SELECT customer_id, order_date, total_amount 
FROM orders 
WHERE customer_id = 123;

-- Index-only scan (all data in index)
CREATE INDEX idx_orders_customer_covering ON orders (customer_id) 
INCLUDE (order_date, total_amount);

EXPLAIN ANALYZE 
SELECT customer_id, order_date, total_amount 
FROM orders 
WHERE customer_id = 123;

-- Visibility map impact on index-only scans
VACUUM orders; -- Updates visibility map

-- Check visibility map
SELECT 
    pg_size_pretty(pg_relation_size('orders')) as table_size,
    pg_size_pretty(pg_relation_size('orders', 'vm')) as visibility_map_size;
```

### Bitmap Index Scans
```sql
-- Multiple indexes combined with bitmap scan
CREATE INDEX idx_products_category ON products (category_id);
CREATE INDEX idx_products_price ON products (price);

-- PostgreSQL may use bitmap scan to combine indexes
EXPLAIN ANALYZE 
SELECT * FROM products 
WHERE category_id = 1 AND price BETWEEN 100 AND 500;

-- Control bitmap scan behavior
SET enable_bitmapscan = off; -- Disable for testing
EXPLAIN ANALYZE 
SELECT * FROM products 
WHERE category_id = 1 AND price BETWEEN 100 AND 500;
SET enable_bitmapscan = on; -- Re-enable
```

### Index Statistics and Correlation
```sql
-- Check index statistics
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation,
    most_common_vals[1:3] as top_values
FROM pg_stats 
WHERE tablename = 'orders' 
AND schemaname = 'public';

-- Physical ordering correlation affects index efficiency
-- High correlation (close to 1.0 or -1.0) means data is physically ordered
-- Low correlation means random distribution

-- Update statistics
ANALYZE orders;

-- Adjust statistics target for better estimates
ALTER TABLE orders ALTER COLUMN customer_id SET STATISTICS 1000;
ANALYZE orders;
```

## Best Practices and Anti-patterns

### Index Design Guidelines
```sql
-- DO: Create indexes for frequently queried columns
CREATE INDEX idx_orders_status ON orders (status) WHERE status IN ('pending', 'processing');

-- DO: Use composite indexes for multi-column queries
CREATE INDEX idx_orders_customer_date ON orders (customer_id, order_date);

-- DON'T: Create too many indexes on frequently updated tables
-- Each index adds overhead to INSERT/UPDATE/DELETE operations

-- DO: Use partial indexes to reduce size
CREATE INDEX idx_customers_premium ON customers (customer_id) 
WHERE subscription_type = 'premium';

-- DON'T: Create redundant indexes
-- If you have (a, b, c), you don't need (a) or (a, b)
-- But you might need (b) or (c) separately

-- DO: Consider covering indexes for frequently accessed columns
CREATE INDEX idx_orders_lookup ON orders (order_id) 
INCLUDE (customer_id, order_date, total_amount);
```

### Index Monitoring Query
```sql
-- Comprehensive index analysis
WITH index_stats AS (
    SELECT 
        s.schemaname,
        s.tablename,
        s.indexname,
        s.idx_scan,
        s.idx_tup_read,
        s.idx_tup_fetch,
        pg_relation_size(s.indexrelid) as index_size,
        pg_relation_size(t.oid) as table_size,
        i.indisunique,
        i.indisprimary,
        array_to_string(array_agg(a.attname ORDER BY a.attnum), ', ') as columns
    FROM pg_stat_user_indexes s
    JOIN pg_index i ON s.indexrelid = i.indexrelid
    JOIN pg_class t ON s.relid = t.oid
    JOIN pg_attribute a ON i.indrelid = a.attrelid AND a.attnum = ANY(i.indkey)
    GROUP BY s.schemaname, s.tablename, s.indexname, s.idx_scan, s.idx_tup_read, 
             s.idx_tup_fetch, s.indexrelid, t.oid, i.indisunique, i.indisprimary
)
SELECT 
    schemaname,
    tablename,
    indexname,
    columns,
    idx_scan,
    pg_size_pretty(index_size) as index_size,
    round(100.0 * index_size / NULLIF(table_size, 0), 2) as size_ratio_pct,
    CASE 
        WHEN idx_scan = 0 AND NOT indisunique AND NOT indisprimary THEN 'Consider dropping'
        WHEN idx_scan < 10 THEN 'Low usage'
        WHEN idx_scan > 1000 THEN 'High usage'
        ELSE 'Normal usage'
    END as recommendation
FROM index_stats
ORDER BY index_size DESC;
```
