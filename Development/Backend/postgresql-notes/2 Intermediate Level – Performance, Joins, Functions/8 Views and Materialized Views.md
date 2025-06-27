### 🟡 Intermediate Level – Topic 8: **Views and Materialized Views**

## ✅ 1. **What is a View?**

A **view** is a saved SQL query treated like a table.
It doesn't store data — just logic.

### 🔹 Create a View

```sql
CREATE VIEW active_customers AS
SELECT id, name, email
FROM customers
WHERE is_active = TRUE;
```

### 🔹 Use the View

```sql
SELECT * FROM active_customers;
```

### 🔹 Update the View Definition

```sql
CREATE OR REPLACE VIEW active_customers AS
SELECT id, name
FROM customers
WHERE is_active = TRUE AND age > 21;
```

### 🔹 Drop a View

```sql
DROP VIEW active_customers;
```

---

## ✅ 2. **Read-only vs Updatable Views**

* A view **is updatable** only if it maps directly to one base table and doesn't include:

  * Aggregates (`SUM()`, `COUNT()`)
  * `DISTINCT`, `GROUP BY`, `LIMIT`, `JOIN`, etc.

### 🔹 Updatable View Example

```sql
CREATE VIEW adult_customers AS
SELECT id, name, age FROM customers WHERE age >= 18;
```

You can do:

```sql
UPDATE adult_customers SET age = 30 WHERE id = 1;
```

---

## ✅ 3. **What is a Materialized View?**

A **materialized view** stores the result of a query like a table.

* Faster for large aggregations or joins
* Needs to be manually refreshed when source data changes

---

### 🔹 Create a Materialized View

```sql
CREATE MATERIALIZED VIEW customer_order_summary AS
SELECT c.id, c.name, COUNT(o.id) AS order_count
FROM customers c
JOIN orders o ON c.id = o.customer_id
GROUP BY c.id;
```

### 🔹 Query It

```sql
SELECT * FROM customer_order_summary;
```

---

### 🔹 Refresh the Materialized View

```sql
REFRESH MATERIALIZED VIEW customer_order_summary;
```

> Use `WITH DATA` (default) or `WITH NO DATA` for faster creation without populating initially.

---

### 🔹 Drop a Materialized View

```sql
DROP MATERIALIZED VIEW customer_order_summary;
```

---

## ✅ 4. **Performance Tip**

* Use **views** for frequently-used complex logic to **simplify code**
* Use **materialized views** to **cache expensive queries**

---

## ✅ Summary

| Feature               | View             | Materialized View      |
| --------------------- | ---------------- | ---------------------- |
| Stores Data           | ❌ No             | ✅ Yes                  |
| Automatically Updates | ✅ Yes            | ❌ No (manual refresh)  |
| Use Case              | Simplify queries | Speed up heavy queries |

---

# Views and Materialized Views

## Introduction to Views

Views are virtual tables that display data from one or more underlying tables. They provide a layer of abstraction and can simplify complex queries, enhance security, and present data in different formats.

## Basic Views

### Creating Simple Views
```sql
-- Basic view from single table
CREATE VIEW active_customers AS
SELECT customer_id, name, email, created_at
FROM customers
WHERE status = 'active';

-- View with calculated columns
CREATE VIEW customer_order_summary AS
SELECT 
    c.customer_id,
    c.name,
    COUNT(o.order_id) as total_orders,
    COALESCE(SUM(o.total_amount), 0) as total_spent,
    COALESCE(AVG(o.total_amount), 0) as avg_order_value
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name;
```

### Using Views
```sql
-- Query views like regular tables
SELECT * FROM active_customers WHERE name LIKE 'John%';

SELECT customer_id, total_orders
FROM customer_order_summary
WHERE total_spent > 1000
ORDER BY total_spent DESC;
```

### View Management
```sql
-- Modify view definition
CREATE OR REPLACE VIEW active_customers AS
SELECT customer_id, name, email, created_at, last_login
FROM customers
WHERE status = 'active' AND last_login >= CURRENT_DATE - INTERVAL '90 days';

-- Drop view
DROP VIEW IF EXISTS customer_order_summary;

-- Rename view
ALTER VIEW active_customers RENAME TO current_customers;
```

## Advanced Views

### Complex Multi-table Views
```sql
-- Comprehensive order view
CREATE VIEW order_details AS
SELECT 
    o.order_id,
    o.order_date,
    c.name as customer_name,
    c.email as customer_email,
    STRING_AGG(p.name, ', ' ORDER BY oi.item_id) as products,
    COUNT(oi.item_id) as item_count,
    SUM(oi.quantity * oi.unit_price) as subtotal,
    o.tax_amount,
    o.total_amount,
    CASE 
        WHEN o.shipped_date IS NOT NULL THEN 'Shipped'
        WHEN o.processed_date IS NOT NULL THEN 'Processing'
        ELSE 'Pending'
    END as status
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
GROUP BY o.order_id, o.order_date, c.name, c.email, o.tax_amount, o.total_amount, o.shipped_date, o.processed_date;
```

### Views with Window Functions
```sql
-- Customer ranking view
CREATE VIEW customer_rankings AS
SELECT 
    customer_id,
    name,
    total_spent,
    RANK() OVER (ORDER BY total_spent DESC) as spending_rank,
    NTILE(10) OVER (ORDER BY total_spent DESC) as spending_decile,
    CASE 
        WHEN total_spent >= PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY total_spent) OVER() THEN 'VIP'
        WHEN total_spent >= PERCENTILE_CONT(0.7) WITHIN GROUP (ORDER BY total_spent) OVER() THEN 'Premium'
        ELSE 'Standard'
    END as customer_tier
FROM (
    SELECT 
        c.customer_id,
        c.name,
        COALESCE(SUM(o.total_amount), 0) as total_spent
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_id, c.name
) customer_totals;
```

### Recursive Views
```sql
-- Organizational hierarchy view
CREATE VIEW org_hierarchy AS
WITH RECURSIVE hierarchy AS (
    -- Base case: top-level managers
    SELECT 
        employee_id,
        name,
        manager_id,
        name as manager_name,
        1 as level,
        ARRAY[employee_id] as path
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    -- Recursive case
    SELECT 
        e.employee_id,
        e.name,
        e.manager_id,
        h.name as manager_name,
        h.level + 1,
        h.path || e.employee_id
    FROM employees e
    JOIN hierarchy h ON e.manager_id = h.employee_id
    WHERE NOT e.employee_id = ANY(h.path) -- Prevent cycles
)
SELECT 
    employee_id,
    name,
    manager_id,
    manager_name,
    level,
    REPEAT('  ', level - 1) || name as indented_name
FROM hierarchy;
```

## Updatable Views

### Simple Updatable Views
```sql
-- View that can be updated
CREATE VIEW active_products AS
SELECT product_id, name, price, description
FROM products
WHERE status = 'active';

-- Updates, inserts, and deletes work automatically
UPDATE active_products SET price = 29.99 WHERE product_id = 1;
INSERT INTO active_products (name, price, description) VALUES ('New Product', 19.99, 'Description');
DELETE FROM active_products WHERE product_id = 2;
```

### Views with Rules for Updates
```sql
-- Create view
CREATE VIEW expensive_products AS
SELECT product_id, name, price, category_id
FROM products
WHERE price > 100;

-- Add rule for inserts to ensure price > 100
CREATE OR REPLACE RULE expensive_products_insert AS
ON INSERT TO expensive_products
DO INSTEAD
INSERT INTO products (name, price, category_id)
VALUES (NEW.name, GREATEST(NEW.price, 100.01), NEW.category_id);

-- Add rule for updates
CREATE OR REPLACE RULE expensive_products_update AS
ON UPDATE TO expensive_products
WHERE NEW.price <= 100
DO INSTEAD NOTHING;
```

### Instead-of Triggers for Complex Views
```sql
-- Complex view that cannot be automatically updated
CREATE VIEW customer_order_info AS
SELECT 
    c.customer_id,
    c.name as customer_name,
    o.order_id,
    o.order_date,
    o.total_amount
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id;

-- Create instead-of trigger function
CREATE OR REPLACE FUNCTION update_customer_order_info()
RETURNS TRIGGER AS $$
BEGIN
    -- Update customer name
    IF OLD.customer_name != NEW.customer_name THEN
        UPDATE customers 
        SET name = NEW.customer_name 
        WHERE customer_id = NEW.customer_id;
    END IF;
    
    -- Update order information
    IF OLD.order_date != NEW.order_date OR OLD.total_amount != NEW.total_amount THEN
        UPDATE orders 
        SET order_date = NEW.order_date, total_amount = NEW.total_amount
        WHERE order_id = NEW.order_id;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create instead-of trigger
CREATE TRIGGER tr_update_customer_order_info
INSTEAD OF UPDATE ON customer_order_info
FOR EACH ROW EXECUTE FUNCTION update_customer_order_info();
```

## Materialized Views

### Basic Materialized Views
```sql
-- Create materialized view
CREATE MATERIALIZED VIEW monthly_sales_summary AS
SELECT 
    DATE_TRUNC('month', order_date) as month,
    COUNT(*) as order_count,
    SUM(total_amount) as total_sales,
    AVG(total_amount) as avg_order_value,
    COUNT(DISTINCT customer_id) as unique_customers
FROM orders
WHERE order_date >= '2020-01-01'
GROUP BY DATE_TRUNC('month', order_date)
ORDER BY month;

-- Create unique index (recommended for refresh performance)
CREATE UNIQUE INDEX idx_monthly_sales_month ON monthly_sales_summary (month);
```

### Refreshing Materialized Views
```sql
-- Full refresh (recreates all data)
REFRESH MATERIALIZED VIEW monthly_sales_summary;

-- Concurrent refresh (requires unique index)
REFRESH MATERIALIZED VIEW CONCURRENTLY monthly_sales_summary;

-- Drop and recreate
DROP MATERIALIZED VIEW monthly_sales_summary;
-- Recreate with new definition
```

### Materialized Views with Dependencies
```sql
-- Base materialized view
CREATE MATERIALIZED VIEW product_sales_base AS
SELECT 
    p.product_id,
    p.name,
    p.category_id,
    SUM(oi.quantity) as total_quantity_sold,
    SUM(oi.quantity * oi.unit_price) as total_revenue
FROM products p
LEFT JOIN order_items oi ON p.product_id = oi.product_id
LEFT JOIN orders o ON oi.order_id = o.order_id
WHERE o.order_date >= CURRENT_DATE - INTERVAL '1 year'
GROUP BY p.product_id, p.name, p.category_id;

-- Dependent materialized view
CREATE MATERIALIZED VIEW category_performance AS
SELECT 
    c.category_id,
    c.name as category_name,
    COUNT(psb.product_id) as product_count,
    SUM(psb.total_quantity_sold) as category_quantity_sold,
    SUM(psb.total_revenue) as category_revenue,
    AVG(psb.total_revenue) as avg_product_revenue
FROM categories c
LEFT JOIN product_sales_base psb ON c.category_id = psb.category_id
GROUP BY c.category_id, c.name;

-- Refresh in dependency order
REFRESH MATERIALIZED VIEW product_sales_base;
REFRESH MATERIALIZED VIEW category_performance;
```

## Performance Optimization

### Indexing Views and Materialized Views
```sql
-- Regular views use indexes on underlying tables
-- Ensure base tables have appropriate indexes
CREATE INDEX idx_customers_status ON customers (status);
CREATE INDEX idx_orders_customer_date ON orders (customer_id, order_date);

-- Materialized views can have their own indexes
CREATE MATERIALIZED VIEW customer_analytics AS
SELECT 
    customer_id,
    registration_date,
    last_order_date,
    total_orders,
    total_spent,
    EXTRACT(DAYS FROM CURRENT_DATE - last_order_date) as days_since_last_order
FROM (
    SELECT 
        c.customer_id,
        c.registration_date,
        MAX(o.order_date) as last_order_date,
        COUNT(o.order_id) as total_orders,
        COALESCE(SUM(o.total_amount), 0) as total_spent
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_id, c.registration_date
) customer_data;

-- Index materialized view for different query patterns
CREATE INDEX idx_customer_analytics_spent ON customer_analytics (total_spent DESC);
CREATE INDEX idx_customer_analytics_last_order ON customer_analytics (last_order_date);
CREATE INDEX idx_customer_analytics_days_since ON customer_analytics (days_since_last_order);
```

### Partitioned Materialized Views
```sql
-- Create partitioned materialized view for large datasets
CREATE MATERIALIZED VIEW daily_metrics (
    metric_date,
    total_orders,
    total_revenue,
    new_customers
) PARTITION BY RANGE (metric_date);

-- Create partitions
CREATE MATERIALIZED VIEW daily_metrics_2023 PARTITION OF daily_metrics
FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');

CREATE MATERIALIZED VIEW daily_metrics_2024 PARTITION OF daily_metrics
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

-- Populate partitioned materialized view
INSERT INTO daily_metrics
SELECT 
    order_date as metric_date,
    COUNT(*) as total_orders,
    SUM(total_amount) as total_revenue,
    COUNT(DISTINCT CASE WHEN c.registration_date = o.order_date THEN c.customer_id END) as new_customers
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
GROUP BY order_date;
```

## View Security and Access Control

### Security Views
```sql
-- Create view that filters sensitive data
CREATE VIEW public_customer_info AS
SELECT 
    customer_id,
    name,
    city,
    state,
    registration_date
FROM customers
-- Exclude sensitive columns like email, phone, SSN
WHERE status = 'active';

-- Grant access to view instead of base table
GRANT SELECT ON public_customer_info TO public_users;
REVOKE ALL ON customers FROM public_users;
```

### Row-Level Security with Views
```sql
-- Create view with built-in row-level security
CREATE VIEW user_orders AS
SELECT 
    order_id,
    order_date,
    total_amount,
    status
FROM orders
WHERE customer_id = current_setting('app.current_customer_id')::INTEGER;

-- Application sets the customer context
-- SET app.current_customer_id = '123';
```

## Automated View Maintenance

### Triggers for Near Real-time Updates
```sql
-- Create log table for tracking changes
CREATE TABLE order_changes (
    change_id SERIAL PRIMARY KEY,
    order_id INTEGER,
    change_type VARCHAR(10), -- INSERT, UPDATE, DELETE
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trigger function to log changes
CREATE OR REPLACE FUNCTION log_order_changes()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO order_changes (order_id, change_type) VALUES (NEW.order_id, 'INSERT');
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO order_changes (order_id, change_type) VALUES (NEW.order_id, 'UPDATE');
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO order_changes (order_id, change_type) VALUES (OLD.order_id, 'DELETE');
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger
CREATE TRIGGER tr_log_order_changes
AFTER INSERT OR UPDATE OR DELETE ON orders
FOR EACH ROW EXECUTE FUNCTION log_order_changes();

-- Procedure to refresh materialized views based on changes
CREATE OR REPLACE FUNCTION refresh_order_summary_views()
RETURNS VOID AS $$
BEGIN
    -- Check if there are any changes since last refresh
    IF EXISTS (
        SELECT 1 FROM order_changes 
        WHERE changed_at > (
            SELECT COALESCE(MAX(last_refresh), '1900-01-01'::timestamp)
            FROM view_refresh_log
            WHERE view_name = 'monthly_sales_summary'
        )
    ) THEN
        REFRESH MATERIALIZED VIEW CONCURRENTLY monthly_sales_summary;
        
        -- Log the refresh
        INSERT INTO view_refresh_log (view_name, last_refresh)
        VALUES ('monthly_sales_summary', CURRENT_TIMESTAMP)
        ON CONFLICT (view_name) DO UPDATE SET last_refresh = CURRENT_TIMESTAMP;
    END IF;
END;
$$ LANGUAGE plpgsql;
```

### Scheduled Refresh Jobs
```sql
-- Using pg_cron extension for scheduled refreshes
-- Enable pg_cron extension first: CREATE EXTENSION pg_cron;

-- Schedule daily refresh at 2 AM
SELECT cron.schedule('refresh-daily-views', '0 2 * * *', 'REFRESH MATERIALIZED VIEW daily_sales_summary;');

-- Schedule hourly refresh for frequently updated views
SELECT cron.schedule('refresh-hourly-views', '0 * * * *', 'REFRESH MATERIALIZED VIEW CONCURRENTLY product_inventory;');
```

## Best Practices

### View Design Guidelines

1. **Keep Views Simple**: Complex views can be hard to maintain and optimize
2. **Use Meaningful Names**: View names should clearly indicate their purpose
3. **Document Views**: Add comments explaining the business logic
4. **Consider Performance**: Views should not significantly slow down queries
5. **Security**: Use views to limit access to sensitive data

### Materialized View Guidelines

1. **Refresh Strategy**: Choose between manual, scheduled, or trigger-based refresh
2. **Size Considerations**: Large materialized views may need partitioning
3. **Index Strategy**: Index materialized views based on query patterns
4. **Dependencies**: Manage refresh order for dependent views
5. **Storage**: Consider storage costs vs. query performance benefits

### Common Anti-patterns

```sql
-- DON'T: Overly complex views
CREATE VIEW bad_complex_view AS
SELECT ... -- 50+ columns from 10+ tables with complex logic

-- DO: Break into simpler, focused views
CREATE VIEW customer_basics AS SELECT ...;
CREATE VIEW order_summaries AS SELECT ...;

-- DON'T: Views without proper indexing on base tables
CREATE VIEW slow_view AS
SELECT * FROM large_table WHERE unindexed_column = 'value';

-- DO: Ensure base tables have appropriate indexes
CREATE INDEX idx_large_table_column ON large_table (indexed_column);
CREATE VIEW fast_view AS
SELECT * FROM large_table WHERE indexed_column = 'value';
```

## Monitoring and Maintenance

### View Usage Statistics
```sql
-- Query to see view usage
SELECT 
    schemaname,
    viewname,
    n_tup_ins,
    n_tup_upd,
    n_tup_del
FROM pg_stat_user_tables
WHERE schemaname NOT IN ('information_schema', 'pg_catalog');

-- Materialized view sizes
SELECT 
    schemaname,
    matviewname,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||matviewname)) as size
FROM pg_matviews;
```

### View Dependencies
```sql
-- Find view dependencies
SELECT DISTINCT
    dependent_ns.nspname as dependent_schema,
    dependent_view.relname as dependent_view,
    source_ns.nspname as source_schema,
    source_table.relname as source_table
FROM pg_depend
JOIN pg_rewrite ON pg_depend.objid = pg_rewrite.oid
JOIN pg_class as dependent_view ON pg_rewrite.ev_class = dependent_view.oid
JOIN pg_class as source_table ON pg_depend.refobjid = source_table.oid
JOIN pg_namespace dependent_ns ON dependent_ns.oid = dependent_view.relnamespace
JOIN pg_namespace source_ns ON source_ns.oid = source_table.relnamespace
WHERE source_table.relkind IN ('r','v','m') -- tables, views, materialized views
ORDER BY dependent_schema, dependent_view;
```
