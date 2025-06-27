### 🔴 Expert Level – Topic 15: **Extensions & Full-Text Search**

## ✅ 1. **Using Extensions in PostgreSQL**

To use any extension:

```sql
CREATE EXTENSION IF NOT EXISTS extension_name;
```

Check what’s available:

```sql
SELECT * FROM pg_available_extensions;
```

---

## ✅ 2. **Key Extensions**

---

### 🔹 `uuid-ossp` – UUID Generator

Adds functions for generating universally unique identifiers.

```sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

**Generate UUIDs:**

```sql
SELECT uuid_generate_v4();  -- random
SELECT uuid_generate_v1();  -- timestamp + MAC address
```

Use in table:

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT
);
```

---

### 🔹 `pg_trgm` – Trigram Similarity

Used for fast **fuzzy string search**.

```sql
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

**Example usage:**

```sql
SELECT 'hello' % 'helo';  -- true if similar enough
SELECT similarity('postgres', 'postgras');  -- ~0.88
```

**Index for speed:**

```sql
CREATE INDEX trgm_idx ON users USING gin (name gin_trgm_ops);
```

---

### 🔹 `fuzzystrmatch` – Phonetic Matching

```sql
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
```

Functions:

* `soundex('word')` – compares pronunciation
* `difference('hello', 'hollow')` – returns similarity score (0–4)

---

### 🔹 `PostGIS` – Geospatial SQL

A full **GIS engine** inside PostgreSQL.

```sql
CREATE EXTENSION IF NOT EXISTS postgis;
```

Adds new types:

* `GEOGRAPHY`, `GEOMETRY`, `POINT`, `POLYGON`

**Example:**

```sql
SELECT ST_Distance(
  ST_GeomFromText('POINT(0 0)'),
  ST_GeomFromText('POINT(3 4)')
);
-- → 5
```

Used in location-based apps, mapping, etc.

---

## ✅ 3. **Full-Text Search in PostgreSQL**

---

PostgreSQL supports built-in **Google-like text search** using:

* `tsvector` – indexable document representation
* `tsquery` – structured search query

---

### 🔹 Create a Searchable Column

```sql
ALTER TABLE articles ADD COLUMN tsv tsvector;
UPDATE articles SET tsv = to_tsvector('english', title || ' ' || body);
```

**Add trigger to auto-update:**

```sql
CREATE TRIGGER tsv_update BEFORE INSERT OR UPDATE
ON articles FOR EACH ROW EXECUTE FUNCTION
tsvector_update_trigger(tsv, 'pg_catalog.english', title, body);
```

---

### 🔹 Search with `@@`

```sql
SELECT * FROM articles
WHERE tsv @@ to_tsquery('database & optimization');
```

**Rank and sort results:**

```sql
SELECT title, ts_rank(tsv, to_tsquery('postgres')) AS rank
FROM articles
WHERE tsv @@ to_tsquery('postgres')
ORDER BY rank DESC;
```

---

### 🔹 Index for Fast Search

```sql
CREATE INDEX idx_articles_tsv ON articles USING GIN(tsv);
```

---

## ✅ Summary

| Extension            | Purpose                           |
| -------------------- | --------------------------------- |
| `uuid-ossp`          | Unique ID generation              |
| `pg_trgm`            | Fuzzy matching (great for search) |
| `fuzzystrmatch`      | Soundex/phonetic comparison       |
| `PostGIS`            | Geospatial queries                |
| `tsvector`/`tsquery` | Full-text search tools            |

PostgreSQL becomes a **search engine**, a **GIS system**, or a **fuzzy matcher** with just a few lines of SQL. That’s serious power.

---

# Extensions & Full-text Search

## PostgreSQL Extensions Overview

PostgreSQL's extension system allows for modular functionality, enabling you to add features without modifying the core database system.

### Managing Extensions
```sql
-- List available extensions
SELECT name, default_version, comment 
FROM pg_available_extensions 
ORDER BY name;

-- List installed extensions
SELECT extname, extversion, nspname as schema
FROM pg_extension e
JOIN pg_namespace n ON e.extnamespace = n.oid;

-- Install extension
CREATE EXTENSION pg_trgm;
CREATE EXTENSION pg_stat_statements;

-- Install extension in specific schema
CREATE EXTENSION hstore SCHEMA extensions;

-- Update extension
ALTER EXTENSION pg_trgm UPDATE TO '1.6';

-- Drop extension
DROP EXTENSION pg_trgm CASCADE;
```

## Full-text Search (FTS)

### Basic Text Search
```sql
-- Simple text search
SELECT * FROM articles 
WHERE to_tsvector('english', title || ' ' || content) @@ to_tsquery('postgresql & search');

-- Using default configuration
SELECT * FROM articles 
WHERE to_tsvector(title || ' ' || content) @@ to_tsquery('database | sql');

-- Search with ranking
SELECT 
    title,
    ts_rank(to_tsvector('english', title || ' ' || content), query) as rank
FROM articles, to_tsquery('english', 'postgresql & performance') query
WHERE to_tsvector('english', title || ' ' || content) @@ query
ORDER BY rank DESC;
```

### Advanced Text Search Configuration
```sql
-- Create custom text search configuration
CREATE TEXT SEARCH CONFIGURATION my_config (COPY = english);

-- Modify configuration for specific token types
ALTER TEXT SEARCH CONFIGURATION my_config
ALTER MAPPING FOR hword, hword_part, word WITH simple;

-- Create text search dictionary
CREATE TEXT SEARCH DICTIONARY my_dict (
    TEMPLATE = simple,
    STOPWORDS = my_stopwords,
    ACCEPT = false
);

-- Use custom configuration
SELECT * FROM articles 
WHERE to_tsvector('my_config', content) @@ to_tsquery('my_config', 'search terms');
```

### Optimizing Full-text Search
```sql
-- Add tsvector column for better performance
ALTER TABLE articles ADD COLUMN content_tsvector tsvector;

-- Update tsvector column
UPDATE articles SET content_tsvector = to_tsvector('english', title || ' ' || content);

-- Create GIN index for fast text search
CREATE INDEX idx_articles_fts ON articles USING gin(content_tsvector);

-- Maintain tsvector with triggers
CREATE OR REPLACE FUNCTION update_content_tsvector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.content_tsvector := to_tsvector('english', NEW.title || ' ' || NEW.content);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tr_update_content_tsvector
    BEFORE INSERT OR UPDATE ON articles
    FOR EACH ROW EXECUTE FUNCTION update_content_tsvector();
```

### Advanced Query Types
```sql
-- Phrase search
SELECT * FROM articles 
WHERE content_tsvector @@ phraseto_tsquery('english', 'postgresql database');

-- Proximity search
SELECT * FROM articles 
WHERE content_tsvector @@ to_tsquery('postgresql <-> database');

-- Wildcard search
SELECT * FROM articles 
WHERE content_tsvector @@ to_tsquery('postgre:*');

-- Boolean queries
SELECT * FROM articles 
WHERE content_tsvector @@ to_tsquery('(postgresql | mysql) & !oracle');

-- Weighted search
SELECT 
    title,
    ts_rank_cd(
        setweight(to_tsvector('english', title), 'A') ||
        setweight(to_tsvector('english', content), 'B'),
        query
    ) as rank
FROM articles, to_tsquery('english', 'postgresql') query
WHERE (setweight(to_tsvector('english', title), 'A') ||
       setweight(to_tsvector('english', content), 'B')) @@ query
ORDER BY rank DESC;
```

### Highlighting Search Results
```sql
-- Highlight search terms in results
SELECT 
    title,
    ts_headline(
        'english',
        content,
        to_tsquery('english', 'postgresql & performance'),
        'StartSel=<mark>, StopSel=</mark>, MaxWords=50'
    ) as highlighted_content
FROM articles 
WHERE to_tsvector('english', content) @@ to_tsquery('english', 'postgresql & performance');

-- Custom headline options
SELECT 
    title,
    ts_headline(
        content,
        to_tsquery('search & optimization'),
        'MaxFragments=3, FragmentDelimiter=" ... ", MaxWords=30'
    ) as snippet
FROM articles 
WHERE content_tsvector @@ to_tsquery('search & optimization');
```

## pg_trgm Extension (Trigram Matching)

### Similarity Search
```sql
-- Install trigram extension
CREATE EXTENSION pg_trgm;

-- Basic similarity search
SELECT name, similarity(name, 'PostgreSQL') as sim
FROM products
WHERE name % 'PostgreSQL'  -- % operator for similarity
ORDER BY sim DESC;

-- Set similarity threshold
SET pg_trgm.similarity_threshold = 0.3;

-- Find similar strings
SELECT * FROM customers 
WHERE name % 'John Smith';

-- Distance-based search
SELECT name, word_similarity('postgres', name) as dist
FROM products
ORDER BY 'postgres' <-> name  -- Distance operator
LIMIT 10;
```

### Fuzzy Text Indexing
```sql
-- Create trigram index for fuzzy matching
CREATE INDEX idx_customers_name_trgm ON customers USING gin (name gin_trgm_ops);
CREATE INDEX idx_products_description_trgm ON products USING gist (description gist_trgm_ops);

-- Fuzzy search with index support
EXPLAIN ANALYZE 
SELECT * FROM customers 
WHERE name % 'Jon Smth';  -- Matches "John Smith"

-- Regex search with trigram index
SELECT * FROM products 
WHERE description ~ 'data.*base' 
AND description % 'database';  -- Use % to enable index usage
```

### Advanced Trigram Operations
```sql
-- Show trigrams for a string
SELECT show_trgm('PostgreSQL database');

-- Compare multiple similarity methods
SELECT 
    name,
    similarity(name, 'postgres') as similarity,
    word_similarity('postgres', name) as word_sim,
    strict_word_similarity('postgres', name) as strict_word_sim
FROM products
WHERE name % 'postgres'
ORDER BY similarity DESC;
```

## JSON Extensions

### Basic JSONB Operations
```sql
-- Query JSONB data
SELECT * FROM products 
WHERE attributes @> '{"category": "electronics"}';

-- Extract JSON values
SELECT 
    name,
    attributes->>'brand' as brand,
    (attributes->>'price')::numeric as price
FROM products;

-- JSON path queries
SELECT * FROM products 
WHERE attributes #> '{specs,weight}' IS NOT NULL;

-- JSON aggregation
SELECT 
    attributes->>'category' as category,
    json_agg(
        json_build_object(
            'name', name,
            'price', attributes->>'price'
        )
    ) as products
FROM products
GROUP BY attributes->>'category';
```

### Advanced JSONB Indexing
```sql
-- GIN index for general JSONB queries
CREATE INDEX idx_products_attributes ON products USING gin (attributes);

-- Specific path index
CREATE INDEX idx_products_brand ON products USING btree ((attributes->>'brand'));

-- Expression index for JSON operations
CREATE INDEX idx_products_price ON products 
USING btree (((attributes->>'price')::numeric));

-- Partial index for specific JSON conditions
CREATE INDEX idx_electronics ON products 
USING gin (attributes)
WHERE attributes @> '{"category": "electronics"}';
```

### JSON Schema Validation
```sql
-- Create JSON schema validation function
CREATE OR REPLACE FUNCTION validate_product_json(data JSONB)
RETURNS BOOLEAN AS $$
BEGIN
    -- Check required fields
    IF NOT (data ? 'name' AND data ? 'price' AND data ? 'category') THEN
        RETURN FALSE;
    END IF;
    
    -- Validate data types
    IF NOT (jsonb_typeof(data->'price') = 'number') THEN
        RETURN FALSE;
    END IF;
    
    -- Validate ranges
    IF (data->>'price')::numeric <= 0 THEN
        RETURN FALSE;
    END IF;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Add constraint using validation function
ALTER TABLE products 
ADD CONSTRAINT valid_product_json 
CHECK (validate_product_json(attributes));
```

## PostGIS Extension (Geospatial)

### Basic PostGIS Setup
```sql
-- Install PostGIS extension
CREATE EXTENSION postgis;
CREATE EXTENSION postgis_topology;

-- Check PostGIS version
SELECT PostGIS_Version();

-- Create table with geometry column
CREATE TABLE stores (
    store_id SERIAL PRIMARY KEY,
    name TEXT,
    location GEOMETRY(POINT, 4326)  -- WGS84 coordinate system
);

-- Insert spatial data
INSERT INTO stores (name, location) VALUES 
('Store A', ST_GeomFromText('POINT(-74.006 40.7128)', 4326)),  -- NYC
('Store B', ST_GeomFromText('POINT(-118.2437 34.0522)', 4326)); -- LA
```

### Spatial Queries
```sql
-- Distance calculations
SELECT 
    s1.name as store1,
    s2.name as store2,
    ST_Distance(s1.location, s2.location) as distance_degrees,
    ST_Distance_Sphere(s1.location, s2.location) as distance_meters
FROM stores s1
CROSS JOIN stores s2
WHERE s1.store_id < s2.store_id;

-- Find nearby stores
SELECT name, ST_AsText(location)
FROM stores
WHERE ST_DWithin(
    location,
    ST_GeomFromText('POINT(-74.0 40.7)', 4326),
    0.1  -- degrees
);

-- Spatial relationships
SELECT * FROM stores 
WHERE ST_Within(
    location,
    ST_GeomFromText('POLYGON((-75 40, -73 40, -73 41, -75 41, -75 40))', 4326)
);
```

### Spatial Indexing
```sql
-- Create spatial index
CREATE INDEX idx_stores_location ON stores USING gist (location);

-- Analyze spatial performance
EXPLAIN ANALYZE 
SELECT * FROM stores 
WHERE ST_DWithin(location, ST_GeomFromText('POINT(-74 40.7)', 4326), 0.1);

-- Spatial clustering
CLUSTER stores USING idx_stores_location;
```

## pg_stat_statements Extension

### Query Performance Monitoring
```sql
-- Install and configure pg_stat_statements
CREATE EXTENSION pg_stat_statements;

-- View top queries by total time
SELECT 
    query,
    calls,
    total_exec_time,
    mean_exec_time,
    stddev_exec_time,
    rows,
    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) as hit_percent
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 10;

-- Find slow queries
SELECT 
    query,
    calls,
    mean_exec_time,
    max_exec_time
FROM pg_stat_statements
WHERE mean_exec_time > 1000  -- Slower than 1 second
ORDER BY mean_exec_time DESC;

-- Reset statistics
SELECT pg_stat_statements_reset();
```

### Query Analysis Functions
```sql
-- Create query analysis function
CREATE OR REPLACE FUNCTION analyze_query_performance()
RETURNS TABLE(
    query_type TEXT,
    avg_time NUMERIC,
    total_calls BIGINT,
    total_time NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        CASE 
            WHEN query LIKE 'SELECT%' THEN 'SELECT'
            WHEN query LIKE 'INSERT%' THEN 'INSERT'
            WHEN query LIKE 'UPDATE%' THEN 'UPDATE'
            WHEN query LIKE 'DELETE%' THEN 'DELETE'
            ELSE 'OTHER'
        END as query_type,
        AVG(mean_exec_time) as avg_time,
        SUM(calls) as total_calls,
        SUM(total_exec_time) as total_time
    FROM pg_stat_statements
    GROUP BY 1
    ORDER BY total_time DESC;
END;
$$ LANGUAGE plpgsql;

-- Use analysis function
SELECT * FROM analyze_query_performance();
```

## hstore Extension

### Key-Value Storage
```sql
-- Install hstore extension
CREATE EXTENSION hstore;

-- Create table with hstore column
CREATE TABLE user_preferences (
    user_id SERIAL PRIMARY KEY,
    username TEXT,
    settings hstore
);

-- Insert hstore data
INSERT INTO user_preferences (username, settings) VALUES 
('john', 'theme=>dark, notifications=>true, language=>en'),
('jane', 'theme=>light, notifications=>false, language=>es, timezone=>PST');

-- Query hstore data
SELECT username, settings->'theme' as theme
FROM user_preferences
WHERE settings ? 'theme';

-- Check for specific key-value pairs
SELECT * FROM user_preferences 
WHERE settings @> 'theme=>dark';

-- Update hstore values
UPDATE user_preferences 
SET settings = settings || 'theme=>blue'::hstore
WHERE username = 'john';
```

### hstore Indexing and Operations
```sql
-- Create GIN index for hstore
CREATE INDEX idx_user_settings ON user_preferences USING gin (settings);

-- hstore operations
SELECT 
    username,
    akeys(settings) as all_keys,
    skeys(settings) as key_set,
    avals(settings) as all_values
FROM user_preferences;

-- Convert hstore to JSON
SELECT 
    username,
    hstore_to_json(settings) as settings_json
FROM user_preferences;

-- Merge hstore values
SELECT settings || 'new_key=>new_value'::hstore as merged_settings
FROM user_preferences
WHERE username = 'john';
```

## UUID Extension

### UUID Generation and Usage
```sql
-- Install UUID extension
CREATE EXTENSION "uuid-ossp";

-- Generate different types of UUIDs
SELECT 
    uuid_generate_v1() as uuid_v1,      -- MAC address + timestamp
    uuid_generate_v1mc() as uuid_v1mc,  -- Random MAC + timestamp
    uuid_generate_v4() as uuid_v4;      -- Random UUID

-- Create table with UUID primary key
CREATE TABLE orders (
    order_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id INTEGER,
    total_amount DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Use UUID in queries
INSERT INTO orders (customer_id, total_amount) 
VALUES (123, 99.99);

SELECT * FROM orders 
WHERE order_id = 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11';
```

## pg_cron Extension

### Job Scheduling
```sql
-- Install pg_cron (requires superuser)
CREATE EXTENSION pg_cron;

-- Schedule daily maintenance at 2 AM
SELECT cron.schedule('daily-maintenance', '0 2 * * *', 'VACUUM ANALYZE;');

-- Schedule hourly cleanup
SELECT cron.schedule('hourly-cleanup', '0 * * * *', 
    'DELETE FROM temp_logs WHERE created_at < NOW() - INTERVAL ''1 day'';');

-- Schedule monthly report generation
SELECT cron.schedule('monthly-reports', '0 0 1 * *',
    'SELECT generate_monthly_report();');

-- View scheduled jobs
SELECT jobid, schedule, command, nodename, nodeport, database, username, active
FROM cron.job;

-- Unschedule job
SELECT cron.unschedule('daily-maintenance');

-- View job run history
SELECT jobid, runid, job_pid, database, username, command, status, return_message, start_time, end_time
FROM cron.job_run_details
ORDER BY start_time DESC;
```

## Custom Extensions

### Creating Simple Extension
```sql
-- Create extension files
-- my_extension.control file:
-- comment = 'My custom extension'
-- default_version = '1.0'
-- relocatable = true
-- requires = ''

-- my_extension--1.0.sql file:
-- CREATE FUNCTION my_custom_function()
-- RETURNS TEXT AS $$
-- BEGIN
--     RETURN 'Hello from my extension!';
-- END;
-- $$ LANGUAGE plpgsql;

-- Install custom extension
-- CREATE EXTENSION my_extension;
```

### Extension Versioning
```sql
-- my_extension--1.0--1.1.sql (upgrade script):
-- ALTER FUNCTION my_custom_function() ...
-- CREATE FUNCTION new_function() ...

-- Upgrade extension
-- ALTER EXTENSION my_extension UPDATE TO '1.1';
```

## Extension Best Practices

### Performance Considerations
```sql
-- Monitor extension impact
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE schemaname = 'extension_schema'
ORDER BY idx_scan DESC;

-- Check extension sizes
SELECT 
    e.extname,
    n.nspname as schema,
    pg_size_pretty(
        COALESCE(
            SUM(pg_total_relation_size(c.oid)), 0
        )
    ) as size
FROM pg_extension e
JOIN pg_namespace n ON e.extnamespace = n.oid
LEFT JOIN pg_depend d ON e.oid = d.refobjid
LEFT JOIN pg_class c ON d.objid = c.oid AND c.relkind IN ('r', 'i')
GROUP BY e.extname, n.nspname
ORDER BY SUM(pg_total_relation_size(c.oid)) DESC NULLS LAST;
```

### Security Considerations
```sql
-- Check extension privileges
SELECT 
    e.extname,
    e.extowner::regrole as owner,
    n.nspname as schema,
    has_schema_privilege(current_user, n.nspname, 'USAGE') as can_use
FROM pg_extension e
JOIN pg_namespace n ON e.extnamespace = n.oid;

-- Limit extension access
REVOKE ALL ON SCHEMA postgis FROM public;
GRANT USAGE ON SCHEMA postgis TO gis_users;
```

### Maintenance and Monitoring
```sql
-- Check for extension updates
SELECT 
    extname,
    extversion as current_version,
    default_version as available_version,
    CASE 
        WHEN extversion = default_version THEN 'Current'
        ELSE 'Update Available'
    END as status
FROM pg_extension e
JOIN pg_available_extensions a ON e.extname = a.name;

-- Extension dependency check
WITH RECURSIVE ext_deps AS (
    -- Base case: direct dependencies
    SELECT 
        e.extname,
        d.refobjid::regclass as dependent_object,
        1 as level
    FROM pg_extension e
    JOIN pg_depend d ON e.oid = d.refobjid
    WHERE d.deptype = 'e'
    
    UNION ALL
    
    -- Recursive case: indirect dependencies
    SELECT 
        ed.extname,
        d.refobjid::regclass,
        ed.level + 1
    FROM ext_deps ed
    JOIN pg_depend d ON ed.dependent_object = d.objid
    WHERE ed.level < 5  -- Prevent infinite recursion
)
SELECT extname, array_agg(DISTINCT dependent_object) as dependencies
FROM ext_deps
GROUP BY extname;
```
