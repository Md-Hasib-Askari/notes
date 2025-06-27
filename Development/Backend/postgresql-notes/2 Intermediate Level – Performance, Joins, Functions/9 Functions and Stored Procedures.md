### 🟡 Intermediate Level – Topic 9: **Functions and Stored Procedures**

## ✅ 1. **What is a Function?**

A **function** is a block of reusable SQL code that **returns a result**.

---

### 🔹 Basic Function Syntax

```sql
CREATE FUNCTION get_customer_name(cid INT)
RETURNS TEXT AS $$
BEGIN
    RETURN (SELECT name FROM customers WHERE id = cid);
END;
$$ LANGUAGE plpgsql;
```

### 🔹 Use It:

```sql
SELECT get_customer_name(1);
```

---

## ✅ 2. **Function Parameters and Return Types**

You can return:

* A **scalar value** (`TEXT`, `INT`, etc.)
* A **row** or **table**
* `VOID` (no return)

### 🔹 Return Table

```sql
CREATE FUNCTION get_top_customers(min_orders INT)
RETURNS TABLE(id INT, name TEXT) AS $$
BEGIN
    RETURN QUERY
    SELECT c.id, c.name
    FROM customers c
    JOIN orders o ON c.id = o.customer_id
    GROUP BY c.id
    HAVING COUNT(o.id) >= min_orders;
END;
$$ LANGUAGE plpgsql;
```

```sql
SELECT * FROM get_top_customers(5);
```

---

## ✅ 3. **Variables and IF Statements in Functions**

```sql
CREATE FUNCTION age_group(age INT)
RETURNS TEXT AS $$
DECLARE
    result TEXT;
BEGIN
    IF age < 18 THEN
        result := 'Minor';
    ELSIF age < 65 THEN
        result := 'Adult';
    ELSE
        result := 'Senior';
    END IF;
    RETURN result;
END;
$$ LANGUAGE plpgsql;
```

---

## ✅ 4. **Stored Procedures (PostgreSQL 11+)**

Unlike functions, **procedures don’t return values**, but can manage transactions.

### 🔹 Example Procedure

```sql
CREATE PROCEDURE delete_customer(cid INT)
LANGUAGE plpgsql
AS $$
BEGIN
    DELETE FROM orders WHERE customer_id = cid;
    DELETE FROM customers WHERE id = cid;
END;
$$;
```

### 🔹 Call It

```sql
CALL delete_customer(1);
```

---

## ✅ 5. **Function Management**

| Operation        | Command Example                         |
| ---------------- | --------------------------------------- |
| List functions   | `\df` (in `psql`)                       |
| Drop function    | `DROP FUNCTION get_customer_name(INT);` |
| Replace function | `CREATE OR REPLACE FUNCTION ...`        |

---

## ✅ 6. **Use Cases**

* Common logic like calculations or text parsing
* Encapsulate repeated queries
* Trigger functions (used in table events)
* Batch operations via procedures

---

## ✅ Summary

* **Functions** are reusable and return results (scalars or tables).
* **Procedures** perform actions and can control transactions (like batch deletes).
* Written in `plpgsql`, PostgreSQL's powerful procedural language.

---

# Functions and Stored Procedures

## Introduction to PostgreSQL Functions

PostgreSQL functions are reusable code blocks that can perform calculations, manipulate data, and implement business logic. They can be written in multiple languages including SQL, PL/pgSQL, Python, and others.

## SQL Functions

### Basic SQL Functions
```sql
-- Simple calculation function
CREATE OR REPLACE FUNCTION calculate_discount(price DECIMAL, discount_rate DECIMAL)
RETURNS DECIMAL AS $$
    SELECT price * discount_rate;
$$ LANGUAGE sql IMMUTABLE;

-- Function with conditional logic
CREATE OR REPLACE FUNCTION get_customer_status(total_spent DECIMAL)
RETURNS TEXT AS $$
    SELECT CASE
        WHEN total_spent >= 10000 THEN 'VIP'
        WHEN total_spent >= 5000 THEN 'Premium'
        WHEN total_spent >= 1000 THEN 'Regular'
        ELSE 'New'
    END;
$$ LANGUAGE sql IMMUTABLE;

-- Using functions in queries
SELECT 
    product_name,
    price,
    calculate_discount(price, 0.15) as discount_amount,
    price - calculate_discount(price, 0.15) as final_price
FROM products;
```

### Functions Returning Tables
```sql
-- Function returning a table
CREATE OR REPLACE FUNCTION get_customer_orders(customer_id_param INTEGER)
RETURNS TABLE(
    order_id INTEGER,
    order_date DATE,
    total_amount DECIMAL,
    status TEXT
) AS $$
    SELECT o.order_id, o.order_date, o.total_amount, o.status
    FROM orders o
    WHERE o.customer_id = customer_id_param
    ORDER BY o.order_date DESC;
$$ LANGUAGE sql STABLE;

-- Using table-returning function
SELECT * FROM get_customer_orders(123);

-- Join with table-returning function
SELECT c.name, co.order_date, co.total_amount
FROM customers c
CROSS JOIN LATERAL get_customer_orders(c.customer_id) co
WHERE c.status = 'active';
```

### Set-Returning Functions
```sql
-- Function returning multiple rows
CREATE OR REPLACE FUNCTION generate_monthly_dates(start_date DATE, end_date DATE)
RETURNS SETOF DATE AS $$
    SELECT generate_series(
        DATE_TRUNC('month', start_date),
        DATE_TRUNC('month', end_date),
        '1 month'::interval
    )::DATE;
$$ LANGUAGE sql IMMUTABLE;

-- Using set-returning function
SELECT * FROM generate_monthly_dates('2023-01-15', '2023-12-20');
```

## PL/pgSQL Functions

### Basic PL/pgSQL Structure
```sql
-- Basic PL/pgSQL function
CREATE OR REPLACE FUNCTION calculate_order_total(order_id_param INTEGER)
RETURNS DECIMAL AS $$
DECLARE
    total_amount DECIMAL := 0;
    item_count INTEGER;
BEGIN
    -- Calculate total from order items
    SELECT 
        COALESCE(SUM(quantity * unit_price), 0),
        COUNT(*)
    INTO total_amount, item_count
    FROM order_items
    WHERE order_id = order_id_param;
    
    -- Apply bulk discount if many items
    IF item_count > 10 THEN
        total_amount := total_amount * 0.95; -- 5% discount
    END IF;
    
    RETURN total_amount;
END;
$$ LANGUAGE plpgsql STABLE;
```

### Variables and Declarations
```sql
CREATE OR REPLACE FUNCTION process_customer_data(customer_id_param INTEGER)
RETURNS TEXT AS $$
DECLARE
    -- Variable declarations
    customer_name VARCHAR(100);
    order_count INTEGER := 0;
    total_spent DECIMAL := 0.0;
    last_order_date DATE;
    customer_tier TEXT;
    result_message TEXT;
    
    -- Record type variable
    customer_info RECORD;
    
    -- Custom type variable
    customer_summary customers%ROWTYPE;
BEGIN
    -- Get customer information
    SELECT name INTO customer_name
    FROM customers
    WHERE customer_id = customer_id_param;
    
    -- Check if customer exists
    IF NOT FOUND THEN
        RETURN 'Customer not found';
    END IF;
    
    -- Get order statistics
    SELECT 
        COUNT(*),
        COALESCE(SUM(total_amount), 0),
        MAX(order_date)
    INTO order_count, total_spent, last_order_date
    FROM orders
    WHERE customer_id = customer_id_param;
    
    -- Determine customer tier
    customer_tier := get_customer_status(total_spent);
    
    -- Build result message
    result_message := format(
        'Customer: %s, Orders: %s, Total Spent: $%s, Tier: %s',
        customer_name, order_count, total_spent, customer_tier
    );
    
    RETURN result_message;
END;
$$ LANGUAGE plpgsql STABLE;
```

### Control Structures

#### IF-THEN-ELSE
```sql
CREATE OR REPLACE FUNCTION apply_pricing_rules(
    base_price DECIMAL,
    customer_tier TEXT,
    quantity INTEGER
)
RETURNS DECIMAL AS $$
DECLARE
    final_price DECIMAL;
    discount_rate DECIMAL := 0;
BEGIN
    -- Set base discount by customer tier
    IF customer_tier = 'VIP' THEN
        discount_rate := 0.20;
    ELSIF customer_tier = 'Premium' THEN
        discount_rate := 0.15;
    ELSIF customer_tier = 'Regular' THEN
        discount_rate := 0.10;
    ELSE
        discount_rate := 0.05;
    END IF;
    
    -- Additional quantity discount
    IF quantity >= 100 THEN
        discount_rate := discount_rate + 0.05;
    ELSIF quantity >= 50 THEN
        discount_rate := discount_rate + 0.03;
    END IF;
    
    -- Calculate final price
    final_price := base_price * (1 - discount_rate);
    
    -- Ensure minimum price
    IF final_price < base_price * 0.5 THEN
        final_price := base_price * 0.5;
    END IF;
    
    RETURN final_price;
END;
$$ LANGUAGE plpgsql IMMUTABLE;
```

#### CASE Statement
```sql
CREATE OR REPLACE FUNCTION get_shipping_cost(
    weight DECIMAL,
    distance INTEGER,
    shipping_method TEXT
)
RETURNS DECIMAL AS $$
DECLARE
    base_cost DECIMAL;
    weight_multiplier DECIMAL;
    distance_multiplier DECIMAL;
BEGIN
    -- Base cost by shipping method
    base_cost := CASE shipping_method
        WHEN 'express' THEN 25.00
        WHEN 'standard' THEN 15.00
        WHEN 'economy' THEN 8.00
        ELSE 10.00
    END;
    
    -- Weight factor
    weight_multiplier := CASE
        WHEN weight <= 1 THEN 1.0
        WHEN weight <= 5 THEN 1.5
        WHEN weight <= 10 THEN 2.0
        ELSE 3.0
    END;
    
    -- Distance factor
    distance_multiplier := CASE
        WHEN distance <= 100 THEN 1.0
        WHEN distance <= 500 THEN 1.3
        WHEN distance <= 1000 THEN 1.6
        ELSE 2.0
    END;
    
    RETURN base_cost * weight_multiplier * distance_multiplier;
END;
$$ LANGUAGE plpgsql IMMUTABLE;
```

#### Loops
```sql
-- FOR loop with integer range
CREATE OR REPLACE FUNCTION calculate_compound_interest(
    principal DECIMAL,
    rate DECIMAL,
    years INTEGER
)
RETURNS DECIMAL AS $$
DECLARE
    amount DECIMAL := principal;
    year_num INTEGER;
BEGIN
    FOR year_num IN 1..years LOOP
        amount := amount * (1 + rate);
    END LOOP;
    
    RETURN amount;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- FOR loop with query results
CREATE OR REPLACE FUNCTION update_customer_tiers()
RETURNS INTEGER AS $$
DECLARE
    customer_record RECORD;
    updated_count INTEGER := 0;
    new_tier TEXT;
BEGIN
    FOR customer_record IN
        SELECT 
            c.customer_id,
            c.name,
            COALESCE(SUM(o.total_amount), 0) as total_spent
        FROM customers c
        LEFT JOIN orders o ON c.customer_id = o.customer_id
        GROUP BY c.customer_id, c.name
    LOOP
        -- Determine new tier
        new_tier := get_customer_status(customer_record.total_spent);
        
        -- Update customer tier
        UPDATE customers
        SET tier = new_tier
        WHERE customer_id = customer_record.customer_id;
        
        updated_count := updated_count + 1;
    END LOOP;
    
    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;

-- WHILE loop
CREATE OR REPLACE FUNCTION fibonacci(n INTEGER)
RETURNS INTEGER AS $$
DECLARE
    a INTEGER := 0;
    b INTEGER := 1;
    temp INTEGER;
    counter INTEGER := 2;
BEGIN
    IF n <= 1 THEN
        RETURN n;
    END IF;
    
    WHILE counter <= n LOOP
        temp := a + b;
        a := b;
        b := temp;
        counter := counter + 1;
    END LOOP;
    
    RETURN b;
END;
$$ LANGUAGE plpgsql IMMUTABLE;
```

## Error Handling

### Exception Handling
```sql
CREATE OR REPLACE FUNCTION safe_divide(numerator DECIMAL, denominator DECIMAL)
RETURNS DECIMAL AS $$
BEGIN
    IF denominator = 0 THEN
        RAISE EXCEPTION 'Division by zero is not allowed';
    END IF;
    
    RETURN numerator / denominator;
EXCEPTION
    WHEN division_by_zero THEN
        RAISE NOTICE 'Caught division by zero error';
        RETURN NULL;
    WHEN OTHERS THEN
        RAISE NOTICE 'An unexpected error occurred: %', SQLERRM;
        RETURN NULL;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function with custom exception handling
CREATE OR REPLACE FUNCTION create_order(
    customer_id_param INTEGER,
    product_ids INTEGER[],
    quantities INTEGER[]
)
RETURNS INTEGER AS $$
DECLARE
    new_order_id INTEGER;
    product_id INTEGER;
    quantity INTEGER;
    i INTEGER;
    available_stock INTEGER;
BEGIN
    -- Validate input arrays have same length
    IF array_length(product_ids, 1) != array_length(quantities, 1) THEN
        RAISE EXCEPTION 'Product IDs and quantities arrays must have same length';
    END IF;
    
    -- Create order
    INSERT INTO orders (customer_id, order_date, status)
    VALUES (customer_id_param, CURRENT_DATE, 'pending')
    RETURNING order_id INTO new_order_id;
    
    -- Add order items
    FOR i IN 1..array_length(product_ids, 1) LOOP
        product_id := product_ids[i];
        quantity := quantities[i];
        
        -- Check stock availability
        SELECT stock_quantity INTO available_stock
        FROM products
        WHERE product_id = product_ids[i];
        
        IF available_stock < quantity THEN
            RAISE EXCEPTION 'Insufficient stock for product %. Available: %, Requested: %',
                product_id, available_stock, quantity;
        END IF;
        
        -- Add order item
        INSERT INTO order_items (order_id, product_id, quantity, unit_price)
        SELECT new_order_id, product_id, quantity, price
        FROM products
        WHERE product_id = product_ids[i];
        
        -- Update stock
        UPDATE products
        SET stock_quantity = stock_quantity - quantity
        WHERE product_id = product_ids[i];
    END LOOP;
    
    RETURN new_order_id;
    
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Order creation failed: %', SQLERRM;
        -- The transaction will be automatically rolled back
        RETURN NULL;
END;
$$ LANGUAGE plpgsql;
```

## Advanced Function Features

### Function Overloading
```sql
-- Function with different parameter types
CREATE OR REPLACE FUNCTION format_currency(amount DECIMAL)
RETURNS TEXT AS $$
    SELECT '$' || to_char(amount, '999,999.99');
$$ LANGUAGE sql IMMUTABLE;

CREATE OR REPLACE FUNCTION format_currency(amount DECIMAL, currency_code TEXT)
RETURNS TEXT AS $$
    SELECT currency_code || ' ' || to_char(amount, '999,999.99');
$$ LANGUAGE sql IMMUTABLE;

-- Usage
SELECT format_currency(1234.56);                    -- $1,234.56
SELECT format_currency(1234.56, 'EUR');            -- EUR 1,234.56
```

### Default Parameters
```sql
CREATE OR REPLACE FUNCTION search_products(
    search_term TEXT,
    category_filter INTEGER DEFAULT NULL,
    min_price DECIMAL DEFAULT 0,
    max_price DECIMAL DEFAULT 999999,
    limit_count INTEGER DEFAULT 50
)
RETURNS TABLE(
    product_id INTEGER,
    name TEXT,
    price DECIMAL,
    category_name TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        p.product_id,
        p.name::TEXT,
        p.price,
        c.name::TEXT
    FROM products p
    JOIN categories c ON p.category_id = c.category_id
    WHERE 
        (search_term IS NULL OR p.name ILIKE '%' || search_term || '%')
        AND (category_filter IS NULL OR p.category_id = category_filter)
        AND p.price BETWEEN min_price AND max_price
    ORDER BY p.name
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql STABLE;

-- Usage with different parameter combinations
SELECT * FROM search_products('laptop');
SELECT * FROM search_products('laptop', 1);
SELECT * FROM search_products('laptop', min_price := 500);
SELECT * FROM search_products(search_term := 'laptop', limit_count := 10);
```

### Variadic Functions
```sql
CREATE OR REPLACE FUNCTION calculate_average(VARIADIC numbers DECIMAL[])
RETURNS DECIMAL AS $$
DECLARE
    total DECIMAL := 0;
    num DECIMAL;
BEGIN
    IF array_length(numbers, 1) IS NULL THEN
        RETURN NULL;
    END IF;
    
    FOREACH num IN ARRAY numbers LOOP
        total := total + num;
    END LOOP;
    
    RETURN total / array_length(numbers, 1);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Usage
SELECT calculate_average(10, 20, 30, 40, 50);
SELECT calculate_average(VARIADIC ARRAY[1.5, 2.5, 3.5]);
```

## Triggers and Trigger Functions

### Basic Trigger Functions
```sql
-- Audit trigger function
CREATE OR REPLACE FUNCTION audit_table_changes()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (table_name, operation, new_values, changed_by, changed_at)
        VALUES (TG_TABLE_NAME, TG_OP, row_to_json(NEW), current_user, now());
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (table_name, operation, old_values, new_values, changed_by, changed_at)
        VALUES (TG_TABLE_NAME, TG_OP, row_to_json(OLD), row_to_json(NEW), current_user, now());
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (table_name, operation, old_values, changed_by, changed_at)
        VALUES (TG_TABLE_NAME, TG_OP, row_to_json(OLD), current_user, now());
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to tables
CREATE TRIGGER tr_audit_customers
    AFTER INSERT OR UPDATE OR DELETE ON customers
    FOR EACH ROW EXECUTE FUNCTION audit_table_changes();

CREATE TRIGGER tr_audit_orders
    AFTER INSERT OR UPDATE OR DELETE ON orders
    FOR EACH ROW EXECUTE FUNCTION audit_table_changes();
```

### Data Validation Triggers
```sql
-- Validation trigger function
CREATE OR REPLACE FUNCTION validate_order_data()
RETURNS TRIGGER AS $$
BEGIN
    -- Validate order date is not in future
    IF NEW.order_date > CURRENT_DATE THEN
        RAISE EXCEPTION 'Order date cannot be in the future';
    END IF;
    
    -- Validate total amount is positive
    IF NEW.total_amount <= 0 THEN
        RAISE EXCEPTION 'Order total must be positive';
    END IF;
    
    -- Validate customer exists and is active
    IF NOT EXISTS (
        SELECT 1 FROM customers 
        WHERE customer_id = NEW.customer_id AND status = 'active'
    ) THEN
        RAISE EXCEPTION 'Customer does not exist or is not active';
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tr_validate_order
    BEFORE INSERT OR UPDATE ON orders
    FOR EACH ROW EXECUTE FUNCTION validate_order_data();
```

### Automatic Calculation Triggers
```sql
-- Automatically update order totals
CREATE OR REPLACE FUNCTION update_order_total()
RETURNS TRIGGER AS $$
DECLARE
    calculated_total DECIMAL;
BEGIN
    -- Calculate total from order items
    SELECT COALESCE(SUM(quantity * unit_price), 0)
    INTO calculated_total
    FROM order_items
    WHERE order_id = COALESCE(NEW.order_id, OLD.order_id);
    
    -- Update order total
    UPDATE orders
    SET total_amount = calculated_total,
        updated_at = CURRENT_TIMESTAMP
    WHERE order_id = COALESCE(NEW.order_id, OLD.order_id);
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tr_update_order_total
    AFTER INSERT OR UPDATE OR DELETE ON order_items
    FOR EACH ROW EXECUTE FUNCTION update_order_total();
```

## Performance Considerations

### Function Volatility
```sql
-- IMMUTABLE: Always returns same result for same input
CREATE OR REPLACE FUNCTION calculate_tax(amount DECIMAL, rate DECIMAL)
RETURNS DECIMAL AS $$
    SELECT amount * rate;
$$ LANGUAGE sql IMMUTABLE;

-- STABLE: Result can vary within single statement
CREATE OR REPLACE FUNCTION get_current_user_id()
RETURNS INTEGER AS $$
    SELECT user_id FROM users WHERE username = current_user;
$$ LANGUAGE sql STABLE;

-- VOLATILE: Result can change within single statement (default)
CREATE OR REPLACE FUNCTION insert_log_entry(message TEXT)
RETURNS TIMESTAMP AS $$
    INSERT INTO log_entries (message, created_at) 
    VALUES (message, now()) 
    RETURNING created_at;
$$ LANGUAGE sql; -- VOLATILE is default
```

### Parallel Safety
```sql
-- Mark function as parallel safe for better performance
CREATE OR REPLACE FUNCTION expensive_calculation(input_value DECIMAL)
RETURNS DECIMAL AS $$
    SELECT input_value * sin(input_value) + cos(input_value);
$$ LANGUAGE sql IMMUTABLE PARALLEL SAFE;
```

### Inlining SQL Functions
```sql
-- Simple SQL functions may be inlined for better performance
CREATE OR REPLACE FUNCTION is_adult(birth_date DATE)
RETURNS BOOLEAN AS $$
    SELECT age(birth_date) >= interval '18 years';
$$ LANGUAGE sql IMMUTABLE;

-- This query may have the function inlined:
SELECT name FROM customers WHERE is_adult(birth_date);
```

## Security and Permissions

### Function Security
```sql
-- SECURITY DEFINER functions run with creator's privileges
CREATE OR REPLACE FUNCTION admin_get_user_count()
RETURNS INTEGER
SECURITY DEFINER -- Runs with creator's permissions
SET search_path = public, pg_temp
AS $$
    SELECT COUNT(*) FROM users;
$$ LANGUAGE sql STABLE;

-- Grant execution to specific roles
GRANT EXECUTE ON FUNCTION admin_get_user_count() TO reporting_users;

-- SECURITY INVOKER functions run with caller's privileges (default)
CREATE OR REPLACE FUNCTION user_get_own_orders(user_id_param INTEGER)
RETURNS TABLE(order_id INTEGER, order_date DATE, total DECIMAL)
SECURITY INVOKER -- Runs with caller's permissions
AS $$
    SELECT o.order_id, o.order_date, o.total_amount
    FROM orders o
    WHERE o.customer_id = user_id_param;
$$ LANGUAGE sql STABLE;
```

## Function Management

### Viewing Function Information
```sql
-- List user-defined functions
SELECT 
    n.nspname as schema_name,
    p.proname as function_name,
    pg_get_function_identity_arguments(p.oid) as arguments,
    pg_get_functiondef(p.oid) as definition
FROM pg_proc p
JOIN pg_namespace n ON p.pronamespace = n.oid
WHERE n.nspname NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
ORDER BY schema_name, function_name;

-- Function dependencies
SELECT DISTINCT
    p.proname as function_name,
    pg_describe_object(d.classid, d.objid, d.objsubid) as depends_on
FROM pg_depend d
JOIN pg_proc p ON d.objid = p.oid
WHERE d.deptype = 'n' -- normal dependency
ORDER BY function_name;
```

### Function Maintenance
```sql
-- Drop function with specific signature
DROP FUNCTION IF EXISTS calculate_discount(DECIMAL, DECIMAL);

-- Drop function with all overloads
DROP FUNCTION calculate_discount CASCADE;

-- Rename function
ALTER FUNCTION calculate_discount(DECIMAL, DECIMAL) RENAME TO apply_discount;

-- Change function owner
ALTER FUNCTION apply_discount(DECIMAL, DECIMAL) OWNER TO new_owner;

-- Add comment
COMMENT ON FUNCTION apply_discount(DECIMAL, DECIMAL) 
IS 'Calculates discount amount based on price and discount rate';
```
