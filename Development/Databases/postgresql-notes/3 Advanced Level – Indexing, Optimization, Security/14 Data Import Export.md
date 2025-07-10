### 🔵 Advanced Level – Topic 14: **Data Import and Export**

## ✅ 1. **`COPY` and `\copy`**

---

### 🔹 `COPY` (Runs inside PostgreSQL server)

```sql
COPY customers TO '/var/lib/postgresql/customers.csv' DELIMITER ',' CSV HEADER;
COPY customers FROM '/var/lib/postgresql/customers.csv' DELIMITER ',' CSV HEADER;
```

> Requires superuser privileges and file must be on the **server**.

---

### 🔹 `\copy` (psql client command – reads from **your local machine**)

```bash
\copy customers FROM 'customers.csv' DELIMITER ',' CSV HEADER;
\copy customers TO 'customers.csv' DELIMITER ',' CSV HEADER;
```

✅ Safer for shared environments
✅ Works without superuser access

---

## ✅ 2. **Supported Formats**

| Format | Usage                               | Example           |
| ------ | ----------------------------------- | ----------------- |
| CSV    | Common, readable                    | `CSV HEADER`      |
| TEXT   | Raw text format                     | `DELIMITER E'\t'` |
| BINARY | Fastest, not human-readable         | `COPY ... BINARY` |
| JSON   | Used with `json` or `jsonb` columns | See below         |

---

### 🔹 Import JSON Data (into JSONB column)

```sql
CREATE TABLE logs (
    id SERIAL,
    data JSONB
);

\copy logs(data) FROM 'logs.json';
```

Each line in `logs.json` must be a valid JSON object:

```json
{"event":"login","user":"hasib"}
{"event":"logout","user":"admin"}
```

---

## ✅ 3. **pg\_dump – Export/Backup**

---

### 🔹 Backup a single database

```bash
pg_dump -U postgres dbname > backup.sql
```

### 🔹 Backup as compressed

```bash
pg_dump -U postgres -Fc dbname > db.dump
```

---

### 🔹 Backup a specific table

```bash
pg_dump -U postgres -t customers dbname > customers.sql
```

---

## ✅ 4. **pg\_restore – Restore from pg\_dump**

Used to restore `.dump` files.

```bash
pg_restore -U postgres -d newdb db.dump
```

### 🔹 Restore only a table:

```bash
pg_restore -U postgres -d dbname -t customers db.dump
```

---

## ✅ 5. **Other Useful Flags**

| Flag              | Description                 |
| ----------------- | --------------------------- |
| `--schema=public` | Only dump specific schema   |
| `--data-only`     | Exclude structure           |
| `--schema-only`   | Exclude data                |
| `--clean`         | Drop objects before restore |

---

## ✅ Summary

| Tool         | Best For                 | Requires Superuser |
| ------------ | ------------------------ | ------------------ |
| `\copy`      | Local file import/export | ❌                  |
| `COPY`       | Server-side file access  | ✅                  |
| `pg_dump`    | Full/partial backups     | ❌ (mostly)         |
| `pg_restore` | Restore compressed dumps | ❌                  |

---

# Data Import/Export

## Overview of Data Import/Export

PostgreSQL provides multiple methods for importing and exporting data, each suited for different use cases, data volumes, and performance requirements.

## COPY Command

### Basic COPY Syntax
```sql
-- Export table to CSV
COPY customers TO '/path/to/customers.csv' DELIMITER ',' CSV HEADER;

-- Import from CSV
COPY customers FROM '/path/to/customers.csv' DELIMITER ',' CSV HEADER;

-- Export specific columns
COPY customers (customer_id, name, email) TO '/path/to/customers_subset.csv' CSV HEADER;

-- Export with conditions (using query)
COPY (SELECT * FROM customers WHERE status = 'active') TO '/path/to/active_customers.csv' CSV HEADER;
```

### Advanced COPY Options
```sql
-- Custom delimiter and quote character
COPY products TO '/path/to/products.tsv' 
DELIMITER E'\t' 
QUOTE '"' 
ESCAPE '\' 
CSV HEADER;

-- Handle NULL values
COPY orders TO '/path/to/orders.csv' 
DELIMITER ',' 
NULL 'NULL' 
CSV HEADER;

-- Specify encoding
COPY customers TO '/path/to/customers_utf8.csv' 
DELIMITER ',' 
CSV HEADER 
ENCODING 'UTF8';

-- Force quotes on all fields
COPY products TO '/path/to/products_quoted.csv' 
DELIMITER ',' 
FORCE_QUOTE * 
CSV HEADER;

-- Custom date format (for text format)
COPY orders TO '/path/to/orders.txt' 
DELIMITER '|' 
NULL '\N';
```

### COPY with Error Handling
```sql
-- Import with error logging (PostgreSQL 14+)
COPY customers FROM '/path/to/customers.csv' 
DELIMITER ',' 
CSV HEADER
LOG_VERBOSITY verbose;

-- Skip header row
COPY customers FROM '/path/to/customers.csv' 
DELIMITER ',' 
CSV 
HEADER false;

-- Import specific columns
COPY customers (name, email, status) FROM '/path/to/customers_partial.csv' 
DELIMITER ',' 
CSV HEADER;
```

## \copy Command (psql)

### Client-side COPY
```sql
-- \copy runs on client side, useful for remote connections
\copy customers TO 'local_customers.csv' CSV HEADER
\copy customers FROM 'local_customers.csv' CSV HEADER

-- With query
\copy (SELECT * FROM orders WHERE order_date >= '2023-01-01') TO 'recent_orders.csv' CSV HEADER

-- Different formats
\copy products TO 'products.tsv' DELIMITER E'\t' CSV HEADER
\copy users FROM 'users.txt' DELIMITER '|' NULL 'NULL'

-- Compressed output (with shell pipes)
\copy customers TO PROGRAM 'gzip > customers.csv.gz' CSV HEADER
\copy customers FROM PROGRAM 'gunzip -c customers.csv.gz' CSV HEADER
```

## pg_dump and pg_restore

### Basic Database Backup
```bash
# Complete database backup
pg_dump mydb > mydb_backup.sql

# Backup with specific user
pg_dump -U username -h hostname mydb > mydb_backup.sql

# Compressed backup
pg_dump -Fc mydb > mydb_backup.dump

# Backup specific schema
pg_dump -n schema_name mydb > schema_backup.sql

# Backup specific tables
pg_dump -t customers -t orders mydb > tables_backup.sql
```

### Advanced pg_dump Options
```bash
# Schema only (no data)
pg_dump -s mydb > schema_only.sql

# Data only (no schema)
pg_dump -a mydb > data_only.sql

# Exclude specific tables
pg_dump --exclude-table=temp_* mydb > backup_no_temp.sql

# Include only specific tables with pattern
pg_dump -t 'customer*' mydb > customer_tables.sql

# Backup with custom format and compression
pg_dump -Fc -Z 9 mydb > mydb_backup_compressed.dump

# Parallel backup (faster for large databases)
pg_dump -Fd -j 4 mydb -f mydb_backup_dir

# Backup with verbose output
pg_dump -v mydb > mydb_backup.sql
```

### pg_restore Usage
```bash
# Restore from custom format
pg_restore -d newdb mydb_backup.dump

# Restore with verbose output
pg_restore -v -d newdb mydb_backup.dump

# Restore specific tables only
pg_restore -t customers -t orders -d newdb mydb_backup.dump

# Restore schema only
pg_restore -s -d newdb mydb_backup.dump

# Restore data only
pg_restore -a -d newdb mydb_backup.dump

# Parallel restore
pg_restore -j 4 -d newdb mydb_backup.dump

# Restore with different database name
pg_restore -d different_db_name mydb_backup.dump

# List contents of backup file
pg_restore -l mydb_backup.dump

# Create custom restore script
pg_restore -l mydb_backup.dump > restore_list.txt
# Edit restore_list.txt to customize
pg_restore -L restore_list.txt -d newdb mydb_backup.dump
```

## Foreign Data Wrappers (FDW)

### file_fdw for CSV Files
```sql
-- Enable file_fdw extension
CREATE EXTENSION file_fdw;

-- Create foreign server
CREATE SERVER file_server FOREIGN DATA WRAPPER file_fdw;

-- Create foreign table for CSV file
CREATE FOREIGN TABLE external_customers (
    customer_id INTEGER,
    name TEXT,
    email TEXT,
    created_at DATE
) SERVER file_server
OPTIONS (filename '/path/to/customers.csv', format 'csv', header 'true');

-- Query external CSV as if it's a regular table
SELECT * FROM external_customers WHERE name LIKE 'John%';

-- Join with local tables
SELECT 
    ec.name,
    ec.email,
    COUNT(o.order_id) as order_count
FROM external_customers ec
LEFT JOIN orders o ON ec.customer_id = o.customer_id
GROUP BY ec.customer_id, ec.name, ec.email;
```

### postgres_fdw for Remote Databases
```sql
-- Enable postgres_fdw extension
CREATE EXTENSION postgres_fdw;

-- Create foreign server
CREATE SERVER remote_server 
FOREIGN DATA WRAPPER postgres_fdw
OPTIONS (host 'remote-host.com', port '5432', dbname 'remotedb');

-- Create user mapping
CREATE USER MAPPING FOR current_user
SERVER remote_server
OPTIONS (user 'remote_user', password 'remote_password');

-- Import specific tables
IMPORT FOREIGN SCHEMA public 
LIMIT TO (customers, orders)
FROM SERVER remote_server 
INTO foreign_schema;

-- Or create foreign table manually
CREATE FOREIGN TABLE remote_customers (
    customer_id INTEGER,
    name TEXT,
    email TEXT,
    status TEXT
) SERVER remote_server
OPTIONS (schema_name 'public', table_name 'customers');

-- Query remote data
SELECT * FROM remote_customers WHERE status = 'active';
```

## Bulk Data Loading Techniques

### Optimized COPY for Large Datasets
```sql
-- Prepare for bulk loading
BEGIN;

-- Disable autocommit and use single transaction
SET autocommit = off;

-- Increase work_mem for better performance
SET work_mem = '256MB';

-- Disable synchronous commit for faster loading (less safe)
SET synchronous_commit = off;

-- Drop indexes temporarily for faster loading
DROP INDEX IF EXISTS idx_customers_email;
DROP INDEX IF EXISTS idx_customers_status;

-- Disable triggers temporarily
ALTER TABLE customers DISABLE TRIGGER ALL;

-- Perform bulk copy
COPY customers FROM '/path/to/large_customer_file.csv' CSV HEADER;

-- Re-enable triggers
ALTER TABLE customers ENABLE TRIGGER ALL;

-- Recreate indexes
CREATE INDEX idx_customers_email ON customers (email);
CREATE INDEX idx_customers_status ON customers (status);

-- Update statistics
ANALYZE customers;

COMMIT;
```

### Using UNLOGGED Tables for Staging
```sql
-- Create unlogged table for fast loading
CREATE UNLOGGED TABLE customers_staging (LIKE customers INCLUDING ALL);

-- Load data into staging table (faster)
COPY customers_staging FROM '/path/to/customers.csv' CSV HEADER;

-- Validate and clean data
DELETE FROM customers_staging WHERE email IS NULL OR email = '';
UPDATE customers_staging SET status = 'active' WHERE status IS NULL;

-- Move validated data to main table
INSERT INTO customers SELECT * FROM customers_staging 
ON CONFLICT (email) DO UPDATE SET
    name = EXCLUDED.name,
    status = EXCLUDED.status,
    updated_at = CURRENT_TIMESTAMP;

-- Clean up staging table
DROP TABLE customers_staging;
```

### Parallel Data Loading
```sql
-- Split large CSV files and load in parallel
-- File 1: customers_part1.csv
-- File 2: customers_part2.csv
-- File 3: customers_part3.csv

-- Session 1:
COPY customers FROM '/path/to/customers_part1.csv' CSV HEADER;

-- Session 2 (parallel):
COPY customers FROM '/path/to/customers_part2.csv' CSV HEADER;

-- Session 3 (parallel):
COPY customers FROM '/path/to/customers_part3.csv' CSV HEADER;
```

## Data Transformation During Import

### Using INSERT with SELECT
```sql
-- Transform data during import using staging table
CREATE TEMP TABLE raw_data (
    full_name TEXT,
    email_address TEXT,
    phone_number TEXT,
    join_date TEXT,
    account_status TEXT
);

-- Load raw data
COPY raw_data FROM '/path/to/raw_customers.csv' CSV HEADER;

-- Transform and insert into target table
INSERT INTO customers (first_name, last_name, email, phone, created_at, status)
SELECT 
    split_part(full_name, ' ', 1) as first_name,
    split_part(full_name, ' ', 2) as last_name,
    lower(trim(email_address)) as email,
    regexp_replace(phone_number, '[^0-9]', '', 'g') as phone,
    to_date(join_date, 'MM/DD/YYYY') as created_at,
    CASE 
        WHEN lower(account_status) = 'active' THEN 'active'
        WHEN lower(account_status) = 'inactive' THEN 'inactive'
        ELSE 'pending'
    END as status
FROM raw_data
WHERE email_address IS NOT NULL 
AND email_address LIKE '%@%';
```

### Data Cleaning Functions
```sql
-- Create functions for data cleaning
CREATE OR REPLACE FUNCTION clean_phone(phone_text TEXT)
RETURNS TEXT AS $$
BEGIN
    -- Remove all non-numeric characters
    RETURN regexp_replace(phone_text, '[^0-9]', '', 'g');
END;
$$ LANGUAGE plpgsql IMMUTABLE;

CREATE OR REPLACE FUNCTION standardize_email(email_text TEXT)
RETURNS TEXT AS $$
BEGIN
    -- Convert to lowercase and trim whitespace
    RETURN lower(trim(email_text));
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Use in data transformation
INSERT INTO customers (name, email, phone)
SELECT 
    trim(name),
    standardize_email(email_raw),
    clean_phone(phone_raw)
FROM raw_customer_data;
```

## Advanced Export Techniques

### Dynamic CSV Generation
```sql
-- Generate CSV with dynamic columns based on JSON data
COPY (
    SELECT 
        customer_id,
        name,
        email,
        profile_data->>'city' as city,
        profile_data->>'age' as age,
        profile_data->>'interests' as interests
    FROM customers
    WHERE profile_data IS NOT NULL
) TO '/path/to/customers_with_profile.csv' CSV HEADER;

-- Export aggregated data
COPY (
    SELECT 
        DATE_TRUNC('month', order_date) as month,
        COUNT(*) as order_count,
        SUM(total_amount) as total_revenue,
        AVG(total_amount) as avg_order_value
    FROM orders
    WHERE order_date >= '2023-01-01'
    GROUP BY DATE_TRUNC('month', order_date)
    ORDER BY month
) TO '/path/to/monthly_sales_summary.csv' CSV HEADER;
```

### Conditional Exports
```sql
-- Export different data based on conditions
DO $$
DECLARE
    export_date DATE := CURRENT_DATE;
    filename TEXT;
BEGIN
    -- Generate filename with current date
    filename := '/path/to/daily_export_' || to_char(export_date, 'YYYY_MM_DD') || '.csv';
    
    -- Export today's orders
    EXECUTE format('COPY (SELECT * FROM orders WHERE DATE(order_date) = %L) TO %L CSV HEADER', 
                   export_date, filename);
END $$;
```

## JSON and XML Import/Export

### JSON Data Handling
```sql
-- Export data as JSON
COPY (
    SELECT row_to_json(customers_json) 
    FROM (
        SELECT customer_id, name, email, 
               json_build_object(
                   'orders', (
                       SELECT json_agg(
                           json_build_object(
                               'order_id', order_id,
                               'order_date', order_date,
                               'total_amount', total_amount
                           )
                       )
                       FROM orders 
                       WHERE orders.customer_id = customers.customer_id
                   )
               ) as order_data
        FROM customers
    ) customers_json
) TO '/path/to/customers_with_orders.json';

-- Import JSON data
CREATE TEMP TABLE json_import (data JSONB);

COPY json_import FROM '/path/to/customer_data.json';

INSERT INTO customers (name, email, phone)
SELECT 
    data->>'name' as name,
    data->>'email' as email,
    data->>'phone' as phone
FROM json_import;
```

### XML Data Export
```sql
-- Export as XML (requires xml2 extension)
COPY (
    SELECT xmlelement(
        name customer,
        xmlattributes(customer_id as id),
        xmlelement(name name, name),
        xmlelement(name email, email),
        xmlelement(name status, status)
    )
    FROM customers
) TO '/path/to/customers.xml';
```

## Error Handling and Validation

### Data Validation Before Import
```sql
-- Create validation function
CREATE OR REPLACE FUNCTION validate_customer_data()
RETURNS TABLE(
    row_number BIGINT,
    error_message TEXT,
    raw_data TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH numbered_data AS (
        SELECT 
            row_number() OVER () as rn,
            *
        FROM customers_staging
    )
    SELECT 
        nd.rn,
        CASE 
            WHEN nd.email IS NULL OR nd.email = '' THEN 'Missing email'
            WHEN nd.email NOT LIKE '%@%' THEN 'Invalid email format'
            WHEN nd.name IS NULL OR nd.name = '' THEN 'Missing name'
            WHEN length(nd.name) < 2 THEN 'Name too short'
            ELSE NULL
        END as error_msg,
        nd::TEXT
    FROM numbered_data nd
    WHERE nd.email IS NULL OR nd.email = '' 
       OR nd.email NOT LIKE '%@%'
       OR nd.name IS NULL OR nd.name = ''
       OR length(nd.name) < 2;
END;
$$ LANGUAGE plpgsql;

-- Run validation
SELECT * FROM validate_customer_data();
```

### Import with Error Logging
```sql
-- Create error log table
CREATE TABLE import_errors (
    error_id SERIAL PRIMARY KEY,
    import_batch TEXT,
    error_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    table_name TEXT,
    error_message TEXT,
    raw_data TEXT
);

-- Import with error handling
DO $$
DECLARE
    batch_id TEXT := 'customer_import_' || to_char(CURRENT_TIMESTAMP, 'YYYY_MM_DD_HH24_MI_SS');
    rec RECORD;
BEGIN
    FOR rec IN SELECT * FROM customers_staging LOOP
        BEGIN
            INSERT INTO customers (name, email, phone, status)
            VALUES (rec.name, rec.email, rec.phone, COALESCE(rec.status, 'active'));
        EXCEPTION 
            WHEN OTHERS THEN
                INSERT INTO import_errors (import_batch, table_name, error_message, raw_data)
                VALUES (batch_id, 'customers', SQLERRM, rec::TEXT);
        END;
    END LOOP;
END $$;
```

## Performance Optimization

### Best Practices for Large Imports
```sql
-- 1. Use transactions wisely
BEGIN;
    -- Disable autocommit for bulk operations
    SET autocommit = OFF;
    
    -- Import large dataset
    COPY large_table FROM '/path/to/huge_file.csv' CSV;
    
    -- Validate critical constraints
    SELECT COUNT(*) FROM large_table WHERE critical_field IS NULL;
    
COMMIT;

-- 2. Optimize PostgreSQL settings for import
-- In postgresql.conf or SET for session:
SET maintenance_work_mem = '1GB';        -- For index creation
SET checkpoint_segments = 32;            -- Reduce checkpoint frequency
SET wal_buffers = '16MB';               -- Larger WAL buffers
SET synchronous_commit = off;           -- Faster but less safe

-- 3. Use UNLOGGED tables for temporary data
CREATE UNLOGGED TABLE temp_import (LIKE target_table);
-- Load data into temp_import first, then move to main table

-- 4. Partition large imports
-- Split files and import in chunks
\copy target_table FROM 'chunk_001.csv' CSV;
\copy target_table FROM 'chunk_002.csv' CSV;
-- ... continue with other chunks
```

### Monitoring Import Progress
```sql
-- Monitor COPY progress (PostgreSQL 14+)
SELECT 
    pid,
    datname,
    usename,
    state,
    query_start,
    query
FROM pg_stat_activity
WHERE query LIKE 'COPY%';

-- Monitor table size growth during import
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
    n_tup_ins,
    n_tup_upd,
    n_tup_del
FROM pg_stat_user_tables
WHERE tablename = 'your_import_table';

-- Check import progress with custom logging
CREATE TABLE import_progress (
    batch_id TEXT,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    records_processed BIGINT,
    status TEXT
);
```
