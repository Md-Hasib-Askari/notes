# 18. Connection Pooling & Parallelism

## Overview
Connection pooling and parallelism are crucial for scaling PostgreSQL applications. Connection pooling manages database connections efficiently, while parallelism allows PostgreSQL to use multiple CPU cores for query execution.

## Table of Contents
- [Connection Pooling](#connection-pooling)
- [PgBouncer](#pgbouncer)
- [Connection Pool Configuration](#connection-pool-configuration)
- [Parallel Query Execution](#parallel-query-execution)
- [Parallel Workers](#parallel-workers)
- [Parallel Operations](#parallel-operations)
- [Monitoring and Tuning](#monitoring-and-tuning)
- [Best Practices](#best-practices)

## Connection Pooling

Connection pooling reduces the overhead of establishing and tearing down database connections by maintaining a pool of reusable connections.

### Why Connection Pooling?

- **Reduced Connection Overhead**: Avoid the cost of creating new connections
- **Better Resource Management**: Control the number of concurrent connections
- **Improved Performance**: Faster response times for application requests
- **Connection Limits**: PostgreSQL has a maximum connection limit

### Connection Pool Types

1. **Session Pooling**: Each client gets a dedicated connection for the session
2. **Transaction Pooling**: Connections are shared between transactions
3. **Statement Pooling**: Connections are shared between individual statements

## PgBouncer

PgBouncer is the most popular connection pooler for PostgreSQL.

### Installation

```bash
# Ubuntu/Debian
sudo apt-get install pgbouncer

# CentOS/RHEL
sudo yum install pgbouncer

# Using Docker
docker run -d --name pgbouncer \
  -p 6432:6432 \
  -v /path/to/pgbouncer.ini:/etc/pgbouncer/pgbouncer.ini \
  pgbouncer/pgbouncer
```

### Basic Configuration

Create `/etc/pgbouncer/pgbouncer.ini`:

```ini
[databases]
mydb = host=localhost port=5432 dbname=mydb
testdb = host=192.168.1.100 port=5432 dbname=testdb

[pgbouncer]
listen_port = 6432
listen_addr = *
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
logfile = /var/log/pgbouncer/pgbouncer.log
pidfile = /var/run/pgbouncer/pgbouncer.pid
admin_users = admin
stats_users = stats, admin

# Pool settings
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 20
min_pool_size = 5
reserve_pool_size = 5
max_db_connections = 100

# Connection limits
server_lifetime = 3600
server_idle_timeout = 600
```

### User Authentication

Create `/etc/pgbouncer/userlist.txt`:

```
"username1" "md5hash_of_password"
"username2" "md5hash_of_password"
"admin" "admin_password_hash"
```

Generate MD5 hash:

```bash
# Generate MD5 hash for password
echo -n "passwordusername" | md5sum
```

### Pool Modes

```ini
# Session pooling - one connection per client session
pool_mode = session

# Transaction pooling - connection released after each transaction
pool_mode = transaction

# Statement pooling - connection released after each statement
pool_mode = statement
```

## Connection Pool Configuration

### Sizing Connection Pools

```ini
# Application connections
max_client_conn = 1000        # Maximum client connections
default_pool_size = 20        # Connections per database
min_pool_size = 5            # Minimum connections to maintain
reserve_pool_size = 5        # Reserved connections for admin

# Database connections
max_db_connections = 100     # Total database connections
```

### Advanced Configuration

```ini
[pgbouncer]
# Performance tuning
server_check_delay = 30
query_timeout = 0
query_wait_timeout = 120
client_idle_timeout = 0
server_idle_timeout = 600
server_lifetime = 3600

# Security
ignore_startup_parameters = extra_float_digits

# Logging
log_connections = 1
log_disconnections = 1
log_pooler_errors = 1
```

## Parallel Query Execution

PostgreSQL can use multiple CPU cores to execute queries in parallel.

### Enabling Parallel Queries

```sql
-- Check current parallel settings
SHOW max_parallel_workers;
SHOW max_parallel_workers_per_gather;
SHOW parallel_tuple_cost;
SHOW parallel_setup_cost;

-- Configure parallel workers
SET max_parallel_workers = 8;
SET max_parallel_workers_per_gather = 4;
SET max_parallel_maintenance_workers = 4;
```

### Parallel Query Configuration

```sql
-- Cost parameters for parallel execution
SET parallel_tuple_cost = 0.1;
SET parallel_setup_cost = 1000.0;

-- Minimum table size for parallel scans
SET min_parallel_table_scan_size = '8MB';
SET min_parallel_index_scan_size = '512kB';

-- Enable/disable parallel operations
SET enable_parallel_append = on;
SET enable_parallel_hash = on;
SET force_parallel_mode = off;
```

## Parallel Workers

### Worker Configuration in postgresql.conf

```ini
# Maximum number of parallel workers for the system
max_parallel_workers = 8

# Maximum parallel workers per Gather node
max_parallel_workers_per_gather = 4

# Maximum parallel workers for maintenance operations
max_parallel_maintenance_workers = 4

# Background worker processes
max_worker_processes = 16
```

### Dynamic Worker Configuration

```sql
-- Set for current session
SET max_parallel_workers_per_gather = 2;

-- Set for specific query
/*+ Parallel(table_name 4) */
SELECT COUNT(*) FROM large_table;
```

## Parallel Operations

### Parallel Sequential Scan

```sql
-- Force parallel scan for demonstration
SET parallel_setup_cost = 0;
SET parallel_tuple_cost = 0;
SET min_parallel_table_scan_size = 0;

-- Check execution plan
EXPLAIN (ANALYZE, BUFFERS) 
SELECT COUNT(*) FROM large_table;
```

### Parallel Aggregation

```sql
-- Parallel aggregation example
EXPLAIN (ANALYZE, BUFFERS)
SELECT 
    category,
    COUNT(*),
    AVG(price),
    SUM(quantity)
FROM products 
GROUP BY category;
```

### Parallel Joins

```sql
-- Parallel hash join
EXPLAIN (ANALYZE, BUFFERS)
SELECT o.order_id, c.customer_name, SUM(oi.amount)
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY o.order_id, c.customer_name;
```

### Parallel Index Operations

```sql
-- Parallel index creation
CREATE INDEX CONCURRENTLY idx_large_table_parallel 
ON large_table (column1, column2);

-- Parallel index scan
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM large_table 
WHERE indexed_column BETWEEN 1000 AND 2000;
```

### Parallel Maintenance Operations

```sql
-- Parallel VACUUM
VACUUM (PARALLEL 4) large_table;

-- Parallel CREATE INDEX
SET maintenance_work_mem = '1GB';
SET max_parallel_maintenance_workers = 4;
CREATE INDEX idx_parallel ON large_table (column1);
```

## Advanced Parallelism

### Parallel Append

```sql
-- Query across partitioned tables
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM partitioned_sales 
WHERE sale_date BETWEEN '2024-01-01' AND '2024-12-31';
```

### Custom Parallel Functions

```sql
-- Create a parallel-safe function
CREATE OR REPLACE FUNCTION expensive_calculation(input_value INTEGER)
RETURNS INTEGER AS $$
BEGIN
    -- Simulate expensive computation
    PERFORM pg_sleep(0.001);
    RETURN input_value * 2;
END;
$$ LANGUAGE plpgsql PARALLEL SAFE;

-- Use in parallel query
SELECT id, expensive_calculation(value)
FROM large_table
WHERE id < 10000;
```

## Monitoring and Tuning

### Connection Pool Monitoring

```sql
-- PgBouncer admin commands
SHOW POOLS;
SHOW CLIENTS;
SHOW SERVERS;
SHOW DATABASES;
SHOW STATS;
SHOW LISTS;
```

### Connection Pool Statistics

```bash
# Connect to PgBouncer admin
psql -h localhost -p 6432 -U admin pgbouncer

# Check pool statistics
SHOW STATS;
SHOW POOLS;

# Reset statistics
RESET STATS;
```

### Parallel Query Monitoring

```sql
-- Check parallel worker usage
SELECT 
    pid,
    application_name,
    state,
    query
FROM pg_stat_activity 
WHERE backend_type = 'parallel worker';

-- Monitor parallel execution
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    parallel_workers_launched
FROM pg_stat_statements
WHERE parallel_workers_launched > 0
ORDER BY total_time DESC;
```

### Performance Metrics

```sql
-- Check if queries are using parallelism
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT COUNT(*) FROM large_table;

-- Monitor wait events for parallel queries
SELECT 
    wait_event_type,
    wait_event,
    COUNT(*)
FROM pg_stat_activity 
WHERE backend_type = 'parallel worker'
GROUP BY wait_event_type, wait_event;
```

## Connection Pool Management

### PgBouncer Commands

```sql
-- Administrative commands
PAUSE database_name;    -- Pause connections to database
RESUME database_name;   -- Resume connections
RELOAD;                 -- Reload configuration
SHUTDOWN;              -- Graceful shutdown

-- Kill connections
KILL database_name;     -- Kill all connections to database
```

### Health Checks

```bash
#!/bin/bash
# PgBouncer health check script

PGBOUNCER_HOST="localhost"
PGBOUNCER_PORT="6432"
ADMIN_USER="admin"

# Check if PgBouncer is responding
if psql -h $PGBOUNCER_HOST -p $PGBOUNCER_PORT -U $ADMIN_USER -d pgbouncer -c "SHOW POOLS;" > /dev/null 2>&1; then
    echo "PgBouncer is healthy"
    exit 0
else
    echo "PgBouncer is not responding"
    exit 1
fi
```

## Load Balancing

### Multiple PgBouncer Instances

```ini
# PgBouncer instance 1
[databases]
mydb = host=db1.example.com port=5432 dbname=mydb

# PgBouncer instance 2  
[databases]
mydb = host=db2.example.com port=5432 dbname=mydb
```

### HAProxy Configuration

```
global
    daemon

defaults
    mode tcp
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms

listen postgresql
    bind *:5432
    option tcp-check
    tcp-check connect
    tcp-check send-binary 00000000
    server pgbouncer1 pgbouncer1:6432 check
    server pgbouncer2 pgbouncer2:6432 check backup
```

## Best Practices

### Connection Pooling

1. **Choose the Right Pool Mode**
   - Transaction pooling for most web applications
   - Session pooling for applications requiring session state
   - Statement pooling for high-throughput, simple queries

2. **Size Pools Appropriately**
   - Monitor connection usage patterns
   - Start conservative and adjust based on load
   - Consider database connection limits

3. **Monitor Pool Health**
   - Set up alerts for pool exhaustion
   - Monitor connection wait times
   - Track error rates

### Parallelism

1. **Tune Parallel Settings**
   - Match `max_parallel_workers` to CPU cores
   - Adjust cost parameters based on hardware
   - Monitor parallel efficiency

2. **Design for Parallelism**
   - Use large tables for parallel operations
   - Avoid functions that aren't parallel-safe
   - Consider partitioning for parallel append

### Configuration Examples

```sql
-- Recommended parallel settings for 8-core system
SET max_parallel_workers = 8;
SET max_parallel_workers_per_gather = 4;
SET max_parallel_maintenance_workers = 4;
SET parallel_tuple_cost = 0.1;
SET parallel_setup_cost = 1000.0;
```

### Troubleshooting

```sql
-- Check why parallel execution isn't being used
EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
SELECT COUNT(*) FROM large_table;

-- Common issues:
-- 1. Table too small (min_parallel_table_scan_size)
-- 2. Cost too high (parallel_setup_cost)
-- 3. No parallel workers available
-- 4. Functions not parallel-safe
```

## Summary

Connection pooling and parallelism are essential for scaling PostgreSQL:

**Connection Pooling:**
- Reduces connection overhead and manages resources efficiently
- PgBouncer is the standard solution with flexible pool modes
- Proper sizing and monitoring are crucial for optimal performance

**Parallelism:**
- Enables PostgreSQL to use multiple CPU cores effectively
- Requires proper configuration and parallel-safe operations
- Most beneficial for large tables and complex aggregations

Both techniques require careful tuning based on your specific workload and hardware configuration.
