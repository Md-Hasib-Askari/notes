### 🟡 Intermediate Level – Topic 6: **Advanced Queries (Joins, GROUP BY, Subqueries)**

## Complex JOIN Operations

### INNER JOIN
```sql
-- Basic INNER JOIN
SELECT u.username, p.title, p.content
FROM users u
INNER JOIN posts p ON u.user_id = p.author_id;

-- Multiple table JOIN
SELECT 
    u.username,
    p.title,
    c.name as category_name
FROM users u
INNER JOIN posts p ON u.user_id = p.author_id
INNER JOIN categories c ON p.category_id = c.category_id;
```

### LEFT/RIGHT JOIN
```sql
-- LEFT JOIN (all users, with or without posts)
SELECT u.username, COUNT(p.post_id) as post_count
FROM users u
LEFT JOIN posts p ON u.user_id = p.author_id
GROUP BY u.user_id, u.username;

-- RIGHT JOIN (all posts, with user info)
SELECT u.username, p.title
FROM users u
RIGHT JOIN posts p ON u.user_id = p.author_id;
```

### FULL OUTER JOIN
```sql
-- FULL OUTER JOIN (all users and all posts)
SELECT u.username, p.title
FROM users u
FULL OUTER JOIN posts p ON u.user_id = p.author_id;
```

### CROSS JOIN
```sql
-- CROSS JOIN (Cartesian product)
SELECT s.size, c.color
FROM sizes s
CROSS JOIN colors c;
```

### Self JOIN
```sql
-- Self JOIN for hierarchical data
SELECT 
    e.name as employee_name,
    m.name as manager_name
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.employee_id;

-- Find employees in same department
SELECT 
    e1.name as employee1,
    e2.name as employee2,
    e1.department
FROM employees e1
JOIN employees e2 ON e1.department = e2.department
WHERE e1.employee_id < e2.employee_id;
```

## Advanced Subqueries

### Correlated Subqueries
```sql
-- Find employees earning more than department average
SELECT employee_id, name, salary, department
FROM employees e1
WHERE salary > (
    SELECT AVG(salary)
    FROM employees e2
    WHERE e2.department = e1.department
);

-- Find latest order for each customer
SELECT customer_id, order_date, total_amount
FROM orders o1
WHERE order_date = (
    SELECT MAX(order_date)
    FROM orders o2
    WHERE o2.customer_id = o1.customer_id
);
```

### EXISTS and NOT EXISTS
```sql
-- Customers who have placed orders
SELECT customer_id, name
FROM customers c
WHERE EXISTS (
    SELECT 1
    FROM orders o
    WHERE o.customer_id = c.customer_id
);

-- Products never ordered
SELECT product_id, name
FROM products p
WHERE NOT EXISTS (
    SELECT 1
    FROM order_items oi
    WHERE oi.product_id = p.product_id
);
```

### Subqueries with ANY/ALL
```sql
-- Products more expensive than ANY product in category 1
SELECT name, price
FROM products
WHERE price > ANY (
    SELECT price
    FROM products
    WHERE category_id = 1
);

-- Products more expensive than ALL products in category 1
SELECT name, price
FROM products
WHERE price > ALL (
    SELECT price
    FROM products
    WHERE category_id = 1
);
```

## Window Functions

### Basic Window Functions
```sql
-- Row number
SELECT 
    employee_id,
    name,
    salary,
    ROW_NUMBER() OVER (ORDER BY salary DESC) as rank
FROM employees;

-- Rank and dense rank
SELECT 
    employee_id,
    name,
    salary,
    RANK() OVER (ORDER BY salary DESC) as rank,
    DENSE_RANK() OVER (ORDER BY salary DESC) as dense_rank
FROM employees;
```

### Partition BY
```sql
-- Rank within each department
SELECT 
    employee_id,
    name,
    department,
    salary,
    RANK() OVER (PARTITION BY department ORDER BY salary DESC) as dept_rank
FROM employees;

-- Running totals by department
SELECT 
    employee_id,
    name,
    department,
    salary,
    SUM(salary) OVER (
        PARTITION BY department 
        ORDER BY employee_id 
        ROWS UNBOUNDED PRECEDING
    ) as running_total
FROM employees;
```

### Lead and Lag
```sql
-- Compare with previous and next values
SELECT 
    order_date,
    total_amount,
    LAG(total_amount) OVER (ORDER BY order_date) as prev_amount,
    LEAD(total_amount) OVER (ORDER BY order_date) as next_amount
FROM orders;

-- Calculate differences
SELECT 
    order_date,
    total_amount,
    total_amount - LAG(total_amount) OVER (ORDER BY order_date) as diff_from_prev
FROM orders;
```

### First and Last Value
```sql
-- First and last values in window
SELECT 
    employee_id,
    name,
    department,
    salary,
    FIRST_VALUE(salary) OVER (
        PARTITION BY department 
        ORDER BY salary DESC
    ) as highest_salary_in_dept,
    LAST_VALUE(salary) OVER (
        PARTITION BY department 
        ORDER BY salary DESC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) as lowest_salary_in_dept
FROM employees;
```

## Common Table Expressions (CTEs)

### Basic CTE
```sql
-- Simple CTE
WITH high_earners AS (
    SELECT employee_id, name, salary
    FROM employees
    WHERE salary > 50000
)
SELECT * FROM high_earners
ORDER BY salary DESC;
```

### Multiple CTEs
```sql
-- Multiple CTEs
WITH 
dept_averages AS (
    SELECT department, AVG(salary) as avg_salary
    FROM employees
    GROUP BY department
),
high_earners AS (
    SELECT e.employee_id, e.name, e.department, e.salary
    FROM employees e
    JOIN dept_averages da ON e.department = da.department
    WHERE e.salary > da.avg_salary
)
SELECT * FROM high_earners;
```

### Recursive CTEs
```sql
-- Recursive CTE for hierarchical data
WITH RECURSIVE org_chart AS (
    -- Base case: top-level managers
    SELECT employee_id, name, manager_id, 1 as level
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    -- Recursive case: employees reporting to current level
    SELECT e.employee_id, e.name, e.manager_id, oc.level + 1
    FROM employees e
    JOIN org_chart oc ON e.manager_id = oc.employee_id
)
SELECT employee_id, name, level
FROM org_chart
ORDER BY level, employee_id;
```

## Set Operations

### UNION and UNION ALL
```sql
-- UNION (removes duplicates)
SELECT customer_id, 'customer' as type
FROM customers
UNION
SELECT supplier_id, 'supplier' as type
FROM suppliers;

-- UNION ALL (keeps duplicates)
SELECT product_name FROM products WHERE price < 10
UNION ALL
SELECT product_name FROM products WHERE category_id = 1;
```

### INTERSECT and EXCEPT
```sql
-- INTERSECT (common values)
SELECT customer_id FROM orders WHERE order_date >= '2023-01-01'
INTERSECT
SELECT customer_id FROM orders WHERE total_amount > 100;

-- EXCEPT (difference)
SELECT customer_id FROM customers
EXCEPT
SELECT DISTINCT customer_id FROM orders;
```

## Advanced Aggregation

### GROUPING SETS
```sql
-- Multiple grouping levels
SELECT 
    department,
    job_title,
    COUNT(*) as employee_count,
    AVG(salary) as avg_salary
FROM employees
GROUP BY GROUPING SETS (
    (department),
    (job_title),
    (department, job_title),
    ()
);
```

### ROLLUP and CUBE
```sql
-- ROLLUP (hierarchical totals)
SELECT 
    department,
    job_title,
    COUNT(*) as employee_count
FROM employees
GROUP BY ROLLUP (department, job_title);

-- CUBE (all possible combinations)
SELECT 
    department,
    job_title,
    COUNT(*) as employee_count
FROM employees
GROUP BY CUBE (department, job_title);
```

### FILTER Clause
```sql
-- Conditional aggregation
SELECT 
    department,
    COUNT(*) as total_employees,
    COUNT(*) FILTER (WHERE salary > 50000) as high_earners,
    AVG(salary) FILTER (WHERE hire_date >= '2020-01-01') as avg_recent_hire_salary
FROM employees
GROUP BY department;
```

## Advanced Date/Time Operations

### Date Ranges and Intervals
```sql
-- Generate date series
SELECT generate_series(
    '2023-01-01'::date,
    '2023-12-31'::date,
    '1 day'::interval
) as date_series;

-- Date truncation
SELECT 
    DATE_TRUNC('month', order_date) as month,
    SUM(total_amount) as monthly_total
FROM orders
GROUP BY DATE_TRUNC('month', order_date)
ORDER BY month;
```

### Time Zone Handling
```sql
-- Convert time zones
SELECT 
    created_at,
    created_at AT TIME ZONE 'UTC' as utc_time,
    created_at AT TIME ZONE 'America/New_York' as est_time
FROM events;

-- Current time in different zones
SELECT 
    NOW() as local_time,
    NOW() AT TIME ZONE 'UTC' as utc_time,
    NOW() AT TIME ZONE 'Asia/Tokyo' as tokyo_time;
```

## JSON Operations

### Basic JSON Queries
```sql
-- Extract JSON fields
SELECT 
    id,
    data->>'name' as name,
    data->>'email' as email,
    (data->>'age')::int as age
FROM users_json;

-- JSON path queries
SELECT 
    id,
    data #> '{address,city}' as city,
    data #>> '{contact,phones,0}' as first_phone
FROM users_json;
```

### JSON Aggregation
```sql
-- Aggregate to JSON
SELECT 
    department,
    JSON_AGG(
        JSON_BUILD_OBJECT(
            'name', name,
            'salary', salary
        )
    ) as employees
FROM employees
GROUP BY department;

-- JSON object aggregation
SELECT 
    JSON_OBJECT_AGG(name, salary) as employee_salaries
FROM employees;
```

## Performance Optimization Techniques

### Using LATERAL JOINs
```sql
-- LATERAL JOIN for correlated queries
SELECT 
    c.customer_id,
    c.name,
    recent_orders.order_count
FROM customers c
LEFT JOIN LATERAL (
    SELECT COUNT(*) as order_count
    FROM orders o
    WHERE o.customer_id = c.customer_id
    AND o.order_date >= CURRENT_DATE - INTERVAL '30 days'
) recent_orders ON true;
```

### Materialized CTEs
```sql
-- Force materialization for performance
WITH expensive_calculation AS MATERIALIZED (
    SELECT 
        product_id,
        complex_calculation(price, cost) as profit_margin
    FROM products
)
SELECT * FROM expensive_calculation
WHERE profit_margin > 0.2;
```

### Query Optimization Hints
```sql
-- Force specific join order (if needed)
SELECT /*+ LEADING(c o) */ 
    c.name, 
    COUNT(o.order_id)
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name;
```

## Error Handling in Queries

### COALESCE and NULLIF
```sql
-- Handle NULL values
SELECT 
    name,
    COALESCE(phone, email, 'No contact') as contact_method,
    NULLIF(description, '') as clean_description
FROM contacts;
```

### TRY-CATCH Equivalent
```sql
-- Safe type conversion
SELECT 
    name,
    CASE 
        WHEN age_text ~ '^[0-9]+$' THEN age_text::int
        ELSE NULL
    END as age
FROM users;
```

## Query Patterns for Analytics

### Cohort Analysis
```sql
-- User cohort analysis
WITH user_cohorts AS (
    SELECT 
        user_id,
        DATE_TRUNC('month', registration_date) as cohort_month
    FROM users
),
monthly_activity AS (
    SELECT 
        user_id,
        DATE_TRUNC('month', activity_date) as activity_month
    FROM user_activities
)
SELECT 
    cohort_month,
    activity_month,
    COUNT(DISTINCT ma.user_id) as active_users
FROM user_cohorts uc
JOIN monthly_activity ma ON uc.user_id = ma.user_id
GROUP BY cohort_month, activity_month
ORDER BY cohort_month, activity_month;
```

### Running Calculations
```sql
-- Running averages and moving windows
SELECT 
    order_date,
    total_amount,
    AVG(total_amount) OVER (
        ORDER BY order_date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as seven_day_avg,
    SUM(total_amount) OVER (
        ORDER BY order_date 
        ROWS UNBOUNDED PRECEDING
    ) as cumulative_total
FROM daily_orders
ORDER BY order_date;
```
---

## ✅ 1. **JOINS – Combining Data from Multiple Tables**

---

### 🔹 INNER JOIN

Returns matching rows from both tables.

```sql
SELECT orders.id, customers.name
FROM orders
INNER JOIN customers ON orders.customer_id = customers.id;
```

---

### 🔹 LEFT JOIN

Returns all rows from the left table, and matched rows from the right table.

```sql
SELECT customers.name, orders.id
FROM customers
LEFT JOIN orders ON customers.id = orders.customer_id;
```

---

### 🔹 RIGHT JOIN

Returns all rows from the right table, and matched rows from the left.

```sql
SELECT customers.name, orders.id
FROM customers
RIGHT JOIN orders ON customers.id = orders.customer_id;
```

---

### 🔹 FULL OUTER JOIN

Returns all rows from both tables, with `NULL` where there’s no match.

```sql
SELECT customers.name, orders.id
FROM customers
FULL OUTER JOIN orders ON customers.id = orders.customer_id;
```

---

## ✅ 2. **GROUP BY and Aggregate Functions**

Used for summarizing data.

```sql
SELECT customer_id, COUNT(*) AS order_count
FROM orders
GROUP BY customer_id;
```

**Common Aggregate Functions:**

* `COUNT()`
* `SUM()`
* `AVG()`
* `MAX()`
* `MIN()`

---

### 🔹 GROUP BY with WHERE and HAVING

```sql
SELECT customer_id, COUNT(*) AS order_count
FROM orders
WHERE order_date >= '2025-01-01'
GROUP BY customer_id
HAVING COUNT(*) > 3;
```

* `WHERE` filters rows **before** grouping.
* `HAVING` filters **after** grouping.

---

## ✅ 3. **Subqueries**

---

### 🔹 Scalar Subquery (returns one value)

```sql
SELECT name,
    (SELECT COUNT(*) FROM orders WHERE orders.customer_id = customers.id) AS order_count
FROM customers;
```

---

### 🔹 Subquery in WHERE

```sql
SELECT name FROM customers
WHERE id IN (
    SELECT customer_id FROM orders WHERE total > 1000
);
```

---

### 🔹 WITH Clause (CTE – Common Table Expression)

Improves readability of complex queries.

```sql
WITH big_spenders AS (
    SELECT customer_id, SUM(total) AS total_spent
    FROM orders
    GROUP BY customer_id
    HAVING SUM(total) > 5000
)
SELECT customers.name, big_spenders.total_spent
FROM customers
JOIN big_spenders ON customers.id = big_spenders.customer_id;
```

---

## ✅ Summary

* Use **JOINs** to combine data from multiple tables based on relationships.
* Use **GROUP BY** for analytics and reporting.
* **Subqueries** and **CTEs** help structure complex logic in readable ways.
