### 🟢 Beginner Level – Topic 4: **Simple Queries**

## Introduction to SQL Queries

SQL (Structured Query Language) is the standard language for interacting with relational databases. This guide covers fundamental query operations.

## Basic SELECT Statement

### Simple Selection
```sql
-- Select all columns from a table
SELECT * FROM users;

-- Select specific columns
SELECT first_name, last_name, email FROM users;

-- Select with column aliases
SELECT first_name AS "First Name", 
       last_name AS "Last Name" 
FROM users;
```

### DISTINCT Keyword
```sql
-- Remove duplicate values
SELECT DISTINCT department FROM employees;

-- Multiple columns with DISTINCT
SELECT DISTINCT department, job_title FROM employees;
```

## WHERE Clause - Filtering Data

### Basic Conditions
```sql
-- Equality condition
SELECT * FROM products WHERE price = 29.99;

-- Inequality conditions
SELECT * FROM products WHERE price > 100;
SELECT * FROM products WHERE price <= 50;
SELECT * FROM products WHERE price != 0;

-- String matching
SELECT * FROM users WHERE first_name = 'John';
```

### Logical Operators
```sql
-- AND operator
SELECT * FROM products 
WHERE price > 50 AND category_id = 1;

-- OR operator
SELECT * FROM products 
WHERE category_id = 1 OR category_id = 2;

-- NOT operator
SELECT * FROM products 
WHERE NOT category_id = 3;
```

### IN and NOT IN
```sql
-- IN operator (multiple values)
SELECT * FROM products 
WHERE category_id IN (1, 2, 3);

-- NOT IN operator
SELECT * FROM products 
WHERE category_id NOT IN (4, 5);
```

### BETWEEN Operator
```sql
-- Range conditions
SELECT * FROM products 
WHERE price BETWEEN 10 AND 100;

-- Date ranges
SELECT * FROM orders 
WHERE order_date BETWEEN '2023-01-01' AND '2023-12-31';
```

### Pattern Matching with LIKE
```sql
-- Wildcards: % (any characters), _ (single character)
SELECT * FROM users 
WHERE first_name LIKE 'J%';  -- Starts with 'J'

SELECT * FROM users 
WHERE first_name LIKE '%son';  -- Ends with 'son'

SELECT * FROM users 
WHERE first_name LIKE '%ar%';  -- Contains 'ar'

SELECT * FROM users 
WHERE first_name LIKE 'J_hn';  -- J + any char + hn
```

### NULL Values
```sql
-- Check for NULL values
SELECT * FROM users WHERE phone IS NULL;

-- Check for non-NULL values
SELECT * FROM users WHERE phone IS NOT NULL;
```

## ORDER BY - Sorting Results

```sql
-- Ascending order (default)
SELECT * FROM products ORDER BY price;

-- Descending order
SELECT * FROM products ORDER BY price DESC;

-- Multiple columns
SELECT * FROM products 
ORDER BY category_id ASC, price DESC;

-- Order by column alias
SELECT product_name, price * 1.1 AS price_with_tax
FROM products
ORDER BY price_with_tax;
```

## LIMIT and OFFSET - Pagination

```sql
-- Limit number of results
SELECT * FROM products LIMIT 10;

-- Skip rows and limit (pagination)
SELECT * FROM products 
ORDER BY product_id 
LIMIT 10 OFFSET 20;  -- Skip 20, take next 10

-- Alternative syntax (PostgreSQL specific)
SELECT * FROM products 
ORDER BY product_id 
OFFSET 20 LIMIT 10;
```

## Aggregate Functions

### Basic Aggregates
```sql
-- Count rows
SELECT COUNT(*) FROM users;
SELECT COUNT(phone) FROM users;  -- Excludes NULLs

-- Sum values
SELECT SUM(price) FROM products;

-- Average values
SELECT AVG(price) FROM products;

-- Min and max values
SELECT MIN(price), MAX(price) FROM products;
```

### GROUP BY Clause
```sql
-- Group by single column
SELECT category_id, COUNT(*) as product_count
FROM products
GROUP BY category_id;

-- Group by multiple columns
SELECT category_id, supplier_id, AVG(price) as avg_price
FROM products
GROUP BY category_id, supplier_id;

-- Group with ORDER BY
SELECT category_id, COUNT(*) as product_count
FROM products
GROUP BY category_id
ORDER BY product_count DESC;
```

### HAVING Clause
```sql
-- Filter groups (use HAVING, not WHERE)
SELECT category_id, COUNT(*) as product_count
FROM products
GROUP BY category_id
HAVING COUNT(*) > 5;

-- Multiple conditions in HAVING
SELECT category_id, AVG(price) as avg_price
FROM products
GROUP BY category_id
HAVING AVG(price) > 50 AND COUNT(*) > 3;
```

## Basic String Functions

```sql
-- String concatenation
SELECT first_name || ' ' || last_name AS full_name
FROM users;

-- String length
SELECT product_name, LENGTH(product_name) as name_length
FROM products;

-- Case conversion
SELECT UPPER(first_name), LOWER(last_name)
FROM users;

-- Substring
SELECT SUBSTRING(product_name FROM 1 FOR 10) as short_name
FROM products;

-- Trim whitespace
SELECT TRIM(product_name) FROM products;
```

## Date and Time Functions

```sql
-- Current date and time
SELECT CURRENT_DATE, CURRENT_TIME, CURRENT_TIMESTAMP;

-- Extract parts of date
SELECT EXTRACT(YEAR FROM order_date) as order_year,
       EXTRACT(MONTH FROM order_date) as order_month
FROM orders;

-- Date arithmetic
SELECT order_date,
       order_date + INTERVAL '30 days' as due_date
FROM orders;

-- Age calculation
SELECT birth_date,
       AGE(birth_date) as age
FROM users;
```

## Mathematical Functions

```sql
-- Basic math operations
SELECT price,
       price * 1.1 as price_with_tax,
       ROUND(price * 1.1, 2) as rounded_price
FROM products;

-- Absolute value
SELECT ABS(-15.5);  -- Returns 15.5

-- Power and square root
SELECT POWER(2, 3);  -- Returns 8
SELECT SQRT(16);     -- Returns 4

-- Random numbers
SELECT RANDOM();  -- Returns value between 0 and 1
```

## CASE Statements

```sql
-- Simple CASE
SELECT product_name,
       price,
       CASE
           WHEN price < 10 THEN 'Cheap'
           WHEN price BETWEEN 10 AND 50 THEN 'Moderate'
           ELSE 'Expensive'
       END as price_category
FROM products;

-- CASE with aggregation
SELECT 
    SUM(CASE WHEN price < 10 THEN 1 ELSE 0 END) as cheap_products,
    SUM(CASE WHEN price >= 10 THEN 1 ELSE 0 END) as expensive_products
FROM products;
```

## COALESCE - Handling NULLs

```sql
-- Return first non-NULL value
SELECT first_name,
       COALESCE(phone, email, 'No contact info') as contact
FROM users;

-- Default values for calculations
SELECT product_name,
       price * COALESCE(discount_rate, 0) as discount_amount
FROM products;
```

## Subqueries (Basic)

```sql
-- Subquery in WHERE clause
SELECT * FROM products
WHERE price > (SELECT AVG(price) FROM products);

-- Subquery with IN
SELECT * FROM orders
WHERE customer_id IN (
    SELECT customer_id FROM customers 
    WHERE city = 'New York'
);
```

## Common Query Patterns

### Top N Records
```sql
-- Top 5 most expensive products
SELECT * FROM products
ORDER BY price DESC
LIMIT 5;
```

### Counting Different Categories
```sql
-- Count products by category
SELECT c.category_name, COUNT(p.product_id) as product_count
FROM categories c
LEFT JOIN products p ON c.category_id = p.category_id
GROUP BY c.category_id, c.category_name
ORDER BY product_count DESC;
```

### Finding Duplicates
```sql
-- Find duplicate email addresses
SELECT email, COUNT(*)
FROM users
GROUP BY email
HAVING COUNT(*) > 1;
```

### Date-based Queries
```sql
-- Orders from last 30 days
SELECT * FROM orders
WHERE order_date >= CURRENT_DATE - INTERVAL '30 days';

-- Monthly sales summary
SELECT 
    EXTRACT(YEAR FROM order_date) as year,
    EXTRACT(MONTH FROM order_date) as month,
    SUM(total_amount) as monthly_total
FROM orders
GROUP BY EXTRACT(YEAR FROM order_date), EXTRACT(MONTH FROM order_date)
ORDER BY year, month;
```

## Query Optimization Tips

1. **Use indexes on frequently queried columns**
2. **Limit result sets with LIMIT**
3. **Use specific column names instead of SELECT ***
4. **Filter early with WHERE clauses**
5. **Use appropriate data types**

## Common Mistakes to Avoid

1. **Forgetting WHERE clause in UPDATE/DELETE**
2. **Using SELECT * in production code**
3. **Not handling NULL values properly**
4. **Inefficient use of LIKE with leading wildcards**
5. **Forgetting to use LIMIT for large datasets**

---

### ✅ Basic SELECT Queries

```sql
SELECT * FROM customers;
```

* Returns **all rows** and **all columns**

```sql
SELECT name, email FROM customers;
```

* Returns only selected columns

---

### ✅ Filtering with WHERE

```sql
SELECT * FROM customers WHERE age > 25;
```

**Operators you can use:**

| Operator | Description      |
| -------- | ---------------- |
| `=`      | Equal            |
| `!=`     | Not equal        |
| `>`      | Greater than     |
| `<`      | Less than        |
| `>=`     | Greater or equal |
| `<=`     | Less or equal    |

---

### ✅ Sorting Results with ORDER BY

```sql
SELECT * FROM customers ORDER BY age ASC;
```

```sql
SELECT * FROM customers ORDER BY name DESC;
```

---

### ✅ LIMIT and OFFSET

```sql
SELECT * FROM orders LIMIT 5;
```

```sql
SELECT * FROM orders OFFSET 5 LIMIT 5;
```

* Useful for **pagination**

---

### ✅ Pattern Matching with LIKE & ILIKE

```sql
SELECT * FROM customers WHERE name LIKE 'A%';
```

* `LIKE` is case-sensitive
* `ILIKE` is case-insensitive

```sql
SELECT * FROM customers WHERE name ILIKE '%john%';
```

---

### ✅ IN, BETWEEN, and IS NULL

```sql
SELECT * FROM customers WHERE age IN (20, 25, 30);
```

```sql
SELECT * FROM orders WHERE order_date BETWEEN '2025-01-01' AND '2025-06-30';
```

```sql
SELECT * FROM customers WHERE email IS NULL;
```

---

### ✅ Renaming Columns (AS)

```sql
SELECT name AS customer_name FROM customers;
```

* Gives an **alias** to a column

---

### ✅ Distinct Values

```sql
SELECT DISTINCT age FROM customers;
```

* Returns only **unique** values

---

### ✅ Math and String Functions

```sql
SELECT name, age + 5 AS age_in_5_years FROM customers;
```

```sql
SELECT UPPER(name), LENGTH(email) FROM customers;
```

---

### ✅ Summary

* Master `SELECT` + `WHERE` + `ORDER BY` for most day-to-day querying.
* Use `LIMIT/OFFSET` for pagination and `LIKE/ILIKE` for search.
* Combine filters logically using `AND`, `OR`, `NOT`.
