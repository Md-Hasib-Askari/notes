### 🔴 Expert Level – Topic 17: **Partitioning**

## ✅ 1. **What is Partitioning?**

Partitioning splits a large table into **child tables** based on certain rules. Queries target only relevant partitions, reducing scan time (called **partition pruning**).

PostgreSQL 10+ supports **native declarative partitioning** — no triggers or inheritance needed.

---

## ✅ 2. **Types of Declarative Partitioning**

---

### 🔹 a. Range Partitioning

Useful for time-based data.

```sql
CREATE TABLE logs (
    log_id SERIAL,
    log_date DATE NOT NULL,
    message TEXT
) PARTITION BY RANGE (log_date);

CREATE TABLE logs_2024 PARTITION OF logs
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

CREATE TABLE logs_2025 PARTITION OF logs
FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');
```

---

### 🔹 b. List Partitioning

Useful for categorical data.

```sql
CREATE TABLE orders (
    id SERIAL,
    region TEXT
) PARTITION BY LIST (region);

CREATE TABLE orders_asia PARTITION OF orders FOR VALUES IN ('asia');
CREATE TABLE orders_europe PARTITION OF orders FOR VALUES IN ('europe');
```

---

### 🔹 c. Hash Partitioning

Evenly distributes data (e.g., sharding).

```sql
CREATE TABLE metrics (
    id INT,
    data TEXT
) PARTITION BY HASH (id);

CREATE TABLE metrics_p0 PARTITION OF metrics FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE metrics_p1 PARTITION OF metrics FOR VALUES WITH (MODULUS 4, REMAINDER 1);
CREATE TABLE metrics_p2 PARTITION OF metrics FOR VALUES WITH (MODULUS 4, REMAINDER 2);
CREATE TABLE metrics_p3 PARTITION OF metrics FOR VALUES WITH (MODULUS 4, REMAINDER 3);
```

---

## ✅ 3. **Default Partition**

Catches rows that don’t match any other partition.

```sql
CREATE TABLE logs_default PARTITION OF logs DEFAULT;
```

---

## ✅ 4. **Performance Benefits**

* **Partition Pruning**: PostgreSQL skips scanning irrelevant partitions.
* **Index per Partition**: Each partition can have its own index.
* **Parallelism**: Queries over partitions can be parallelized.
* **Maintenance**: Easier `DROP`, `VACUUM`, `ANALYZE` on specific partitions.

---

## ✅ 5. **Indexing Strategies**

Each partition can be indexed separately:

```sql
CREATE INDEX idx_logs_2025_date ON logs_2025(log_date);
```

You cannot (yet) create a single global index across all partitions.

---

## ✅ 6. **Partitioning Limitations (and Fixes)**

| Limitation                                         | Status                                              |
| -------------------------------------------------- | --------------------------------------------------- |
| No global indexes                                  | ✅ Known, workaround = partial indexes per partition |
| Can’t reference child tables directly              | ✅ By design; query parent table                     |
| Insert performance degrades if too many partitions | ✅ Tune `constraint_exclusion`, limit count          |

---

## ✅ 7. **Best Practices**

* Use **range** partitioning for time-series data.
* Use **list** when data fits into known categories.
* Use **hash** when uniform distribution is needed (e.g., sharding).
* Always create a **default partition** to catch unexpected data.
* Monitor `EXPLAIN` to ensure **partition pruning** is working.

---

## ✅ Summary

| Type  | Best For             | Example Use Case                |
| ----- | -------------------- | ------------------------------- |
| Range | Time-based data      | Logs, events                    |
| List  | Categories           | Regions, product types          |
| Hash  | Uniform distribution | Load-balanced inserts, sharding |

Partitioning is **essential for large-scale databases** — enabling better performance, cheaper maintenance, and cleaner architecture.

---

# 17. Partitioning

## Overview
PostgreSQL partitioning allows you to split large tables into smaller, more manageable pieces while maintaining the appearance of a single table. This improves query performance and makes maintenance operations more efficient.

## Table of Contents
- [What is Partitioning?](#what-is-partitioning)
- [Types of Partitioning](#types-of-partitioning)
- [Range Partitioning](#range-partitioning)
- [List Partitioning](#list-partitioning)
- [Hash Partitioning](#hash-partitioning)
- [Partition Management](#partition-management)
- [Query Planning with Partitions](#query-planning-with-partitions)
- [Best Practices](#best-practices)

## What is Partitioning?

Partitioning is a database design technique where a large table is divided into smaller, more manageable pieces called partitions. Each partition is a separate table that stores a subset of the data.

### Benefits of Partitioning
- **Improved Query Performance**: Queries can access only relevant partitions
- **Parallel Processing**: Operations can run on multiple partitions simultaneously
- **Easier Maintenance**: Operations like VACUUM, ANALYZE can be performed on individual partitions
- **Bulk Operations**: Loading and deleting data becomes more efficient

## Types of Partitioning

PostgreSQL supports three types of partitioning:

### 1. Range Partitioning
Data is distributed based on a range of values.

### 2. List Partitioning
Data is distributed based on a list of discrete values.

### 3. Hash Partitioning
Data is distributed based on a hash function.

## Range Partitioning

Range partitioning is most commonly used for time-based data.

### Creating a Range Partitioned Table

```sql
-- Create the parent table
CREATE TABLE sales (
    id SERIAL,
    sale_date DATE NOT NULL,
    amount DECIMAL(10,2),
    customer_id INTEGER
) PARTITION BY RANGE (sale_date);

-- Create partitions for different date ranges
CREATE TABLE sales_2023 PARTITION OF sales
    FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');

CREATE TABLE sales_2024 PARTITION OF sales
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

CREATE TABLE sales_2025 PARTITION OF sales
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');
```

### Creating Indexes on Partitioned Tables

```sql
-- Create indexes on the parent table (will be created on all partitions)
CREATE INDEX idx_sales_customer ON sales (customer_id);
CREATE INDEX idx_sales_amount ON sales (amount);

-- Create partition-specific indexes if needed
CREATE INDEX idx_sales_2024_amount ON sales_2024 (amount, sale_date);
```

## List Partitioning

List partitioning is useful for categorical data.

```sql
-- Create parent table partitioned by region
CREATE TABLE customer_data (
    id SERIAL,
    name VARCHAR(100),
    region VARCHAR(50),
    registration_date DATE
) PARTITION BY LIST (region);

-- Create partitions for different regions
CREATE TABLE customer_data_north PARTITION OF customer_data
    FOR VALUES IN ('north', 'northeast', 'northwest');

CREATE TABLE customer_data_south PARTITION OF customer_data
    FOR VALUES IN ('south', 'southeast', 'southwest');

CREATE TABLE customer_data_other PARTITION OF customer_data
    DEFAULT;  -- Catch-all for values not in other partitions
```

## Hash Partitioning

Hash partitioning distributes data evenly across partitions.

```sql
-- Create parent table with hash partitioning
CREATE TABLE user_activity (
    id SERIAL,
    user_id INTEGER,
    activity_type VARCHAR(50),
    timestamp TIMESTAMP
) PARTITION BY HASH (user_id);

-- Create hash partitions
CREATE TABLE user_activity_0 PARTITION OF user_activity
    FOR VALUES WITH (MODULUS 4, REMAINDER 0);

CREATE TABLE user_activity_1 PARTITION OF user_activity
    FOR VALUES WITH (MODULUS 4, REMAINDER 1);

CREATE TABLE user_activity_2 PARTITION OF user_activity
    FOR VALUES WITH (MODULUS 4, REMAINDER 2);

CREATE TABLE user_activity_3 PARTITION OF user_activity
    FOR VALUES WITH (MODULUS 4, REMAINDER 3);
```

## Partition Management

### Adding New Partitions

```sql
-- Add a new partition for 2026
CREATE TABLE sales_2026 PARTITION OF sales
    FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');
```

### Dropping Partitions

```sql
-- Drop a partition (and its data)
DROP TABLE sales_2023;

-- Detach a partition (keep data but remove from partitioned table)
ALTER TABLE sales DETACH PARTITION sales_2023;
```

### Moving Data Between Partitions

```sql
-- Create a new partition with different range
CREATE TABLE sales_q1_2024 PARTITION OF sales
    FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');

-- Move data from existing partition
WITH moved_rows AS (
    DELETE FROM sales_2024 
    WHERE sale_date >= '2024-01-01' AND sale_date < '2024-04-01'
    RETURNING *
)
INSERT INTO sales_q1_2024 SELECT * FROM moved_rows;
```

## Query Planning with Partitions

### Partition Pruning

PostgreSQL automatically excludes irrelevant partitions from query execution:

```sql
-- This query will only scan sales_2024 partition
SELECT * FROM sales 
WHERE sale_date BETWEEN '2024-06-01' AND '2024-06-30';

-- Check execution plan
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM sales 
WHERE sale_date = '2024-06-15';
```

### Constraint Exclusion

Enable constraint exclusion for better partition pruning:

```sql
-- Set constraint exclusion
SET constraint_exclusion = partition;

-- Add check constraints to partitions for better pruning
ALTER TABLE sales_2024 ADD CONSTRAINT sales_2024_check 
    CHECK (sale_date >= '2024-01-01' AND sale_date < '2025-01-01');
```

## Advanced Partitioning Techniques

### Sub-partitioning

```sql
-- Create a table partitioned by range, then by list
CREATE TABLE events (
    id SERIAL,
    event_date DATE,
    event_type VARCHAR(50),
    data JSONB
) PARTITION BY RANGE (event_date);

-- Create yearly partitions
CREATE TABLE events_2024 PARTITION OF events
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01')
    PARTITION BY LIST (event_type);

-- Create sub-partitions by event type
CREATE TABLE events_2024_user PARTITION OF events_2024
    FOR VALUES IN ('login', 'logout', 'signup');

CREATE TABLE events_2024_system PARTITION OF events_2024
    FOR VALUES IN ('backup', 'maintenance', 'alert');
```

### Partition-wise Joins

```sql
-- Enable partition-wise joins for better performance
SET enable_partitionwise_join = on;
SET enable_partitionwise_aggregate = on;

-- Query that benefits from partition-wise operations
SELECT s.sale_date, s.amount, c.region
FROM sales s
JOIN customer_data c ON s.customer_id = c.id
WHERE s.sale_date BETWEEN '2024-01-01' AND '2024-12-31';
```

## Monitoring Partitions

### Check Partition Information

```sql
-- View partition information
SELECT 
    schemaname,
    tablename,
    partitionkeydef,
    partitionname,
    partitionrangestart,
    partitionrangeend
FROM pg_partitions 
WHERE tablename = 'sales';

-- Check partition sizes
SELECT 
    t.schemaname,
    t.tablename,
    pg_size_pretty(pg_total_relation_size(t.schemaname||'.'||t.tablename)) as size
FROM pg_tables t
WHERE t.tablename LIKE 'sales_%'
ORDER BY pg_total_relation_size(t.schemaname||'.'||t.tablename) DESC;
```

### Partition Statistics

```sql
-- Check row counts per partition
SELECT 
    schemaname,
    tablename,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes
FROM pg_stat_user_tables 
WHERE tablename LIKE 'sales_%';
```

## Maintenance Operations

### Automated Partition Creation

```sql
-- Function to create monthly partitions
CREATE OR REPLACE FUNCTION create_monthly_partition(
    table_name TEXT,
    start_date DATE
) RETURNS VOID AS $$
DECLARE
    partition_name TEXT;
    end_date DATE;
BEGIN
    partition_name := table_name || '_' || to_char(start_date, 'YYYY_MM');
    end_date := start_date + INTERVAL '1 month';
    
    EXECUTE format('CREATE TABLE %I PARTITION OF %I
                    FOR VALUES FROM (%L) TO (%L)',
                   partition_name, table_name, start_date, end_date);
END;
$$ LANGUAGE plpgsql;

-- Create partitions for the next 12 months
SELECT create_monthly_partition('sales', date_trunc('month', CURRENT_DATE + interval '1 month' * generate_series(0, 11)));
```

### Partition Maintenance Script

```sql
-- Procedure to maintain partitions
CREATE OR REPLACE FUNCTION maintain_partitions()
RETURNS VOID AS $$
DECLARE
    old_partition_date DATE;
    old_partition_name TEXT;
BEGIN
    -- Drop partitions older than 2 years
    old_partition_date := CURRENT_DATE - INTERVAL '2 years';
    old_partition_name := 'sales_' || to_char(old_partition_date, 'YYYY');
    
    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = old_partition_name) THEN
        EXECUTE 'DROP TABLE ' || old_partition_name;
        RAISE NOTICE 'Dropped partition: %', old_partition_name;
    END IF;
    
    -- Create next month's partition if it doesn't exist
    PERFORM create_monthly_partition('sales', 
                                   date_trunc('month', CURRENT_DATE + INTERVAL '2 months'));
END;
$$ LANGUAGE plpgsql;
```

## Best Practices

### Design Considerations

1. **Choose the Right Partition Key**
   - Use columns frequently used in WHERE clauses
   - Ensure even data distribution
   - Consider query patterns

2. **Partition Size**
   - Aim for partitions with 1-10 million rows
   - Balance between too many small partitions and too few large ones

3. **Indexing Strategy**
   - Create indexes on the parent table when possible
   - Consider partition-specific indexes for unique access patterns

### Performance Tips

```sql
-- Use partition pruning effectively
-- GOOD: Uses partition key in WHERE clause
SELECT * FROM sales WHERE sale_date = '2024-06-15';

-- BAD: Doesn't use partition key
SELECT * FROM sales WHERE customer_id = 123;

-- Enable runtime partition pruning
SET enable_runtime_pruning = on;
```

### Common Pitfalls

1. **Unique Constraints**: Must include partition key
2. **Foreign Keys**: Complex with partitioned tables
3. **Cross-partition Queries**: Can be expensive
4. **Too Many Partitions**: Can hurt performance

## Troubleshooting

### Check Partition Pruning

```sql
-- Verify partition pruning is working
EXPLAIN (ANALYZE, BUFFERS, TIMING OFF) 
SELECT * FROM sales 
WHERE sale_date BETWEEN '2024-01-01' AND '2024-01-31';
```

### Performance Issues

```sql
-- Check if queries are scanning too many partitions
SELECT 
    query,
    calls,
    total_time,
    mean_time
FROM pg_stat_statements 
WHERE query LIKE '%sales%'
ORDER BY total_time DESC;
```

## Summary

Partitioning is a powerful technique for managing large tables in PostgreSQL. Key takeaways:

- Choose the appropriate partitioning strategy based on your data and query patterns
- Implement proper maintenance procedures for partition lifecycle management
- Monitor partition performance and adjust strategies as needed
- Consider the trade-offs between partition granularity and management overhead

Proper partitioning can dramatically improve query performance and make database maintenance more manageable for large datasets.
