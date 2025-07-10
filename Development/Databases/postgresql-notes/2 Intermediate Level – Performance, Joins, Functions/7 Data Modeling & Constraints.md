### 🟡 Intermediate Level – Topic 7: **Data Modeling & Constraints**

## Advanced Data Modeling Concepts

### Conceptual vs Logical vs Physical Models

#### Conceptual Model
- High-level business view
- Entities and relationships
- No technical details
- Business rules and requirements

#### Logical Model
- Platform-independent design
- Detailed attributes and relationships
- Normalized structure
- Data types and constraints

#### Physical Model
- Database-specific implementation
- Indexes, partitions, storage
- Performance optimizations
- Physical storage considerations

## Entity Relationship Modeling

### Entity Types
```sql
-- Strong Entity (independent existence)
CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL
);

-- Weak Entity (depends on strong entity)
CREATE TABLE order_items (
    order_id INTEGER NOT NULL,
    item_sequence INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    PRIMARY KEY (order_id, item_sequence),
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
        ON DELETE CASCADE
);
```

### Relationship Types and Cardinality

#### One-to-One (1:1)
```sql
-- User and user profile (1:1)
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL
);

CREATE TABLE user_profiles (
    user_id INTEGER PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    bio TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
        ON DELETE CASCADE
);
```

#### One-to-Many (1:M)
```sql
-- Department and employees (1:M)
CREATE TABLE departments (
    department_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    manager_id INTEGER
);

CREATE TABLE employees (
    employee_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    department_id INTEGER NOT NULL,
    FOREIGN KEY (department_id) REFERENCES departments(department_id)
);
```

#### Many-to-Many (M:M)
```sql
-- Students and courses (M:M)
CREATE TABLE students (
    student_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

CREATE TABLE courses (
    course_id SERIAL PRIMARY KEY,
    title VARCHAR(100) NOT NULL,
    credits INTEGER
);

-- Junction/Bridge table
CREATE TABLE enrollments (
    student_id INTEGER,
    course_id INTEGER,
    enrollment_date DATE DEFAULT CURRENT_DATE,
    grade CHAR(2),
    PRIMARY KEY (student_id, course_id),
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (course_id) REFERENCES courses(course_id)
);
```

## Advanced Constraints

### Check Constraints
```sql
-- Basic check constraints
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    price DECIMAL(10,2) NOT NULL CHECK (price > 0),
    category VARCHAR(50) CHECK (category IN ('electronics', 'clothing', 'books')),
    weight DECIMAL(8,2) CHECK (weight > 0),
    
    -- Multiple column check
    CONSTRAINT valid_dimensions CHECK (
        (length > 0 AND width > 0 AND height > 0) OR
        (length IS NULL AND width IS NULL AND height IS NULL)
    )
);

-- Complex business rules
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date DATE NOT NULL DEFAULT CURRENT_DATE,
    ship_date DATE,
    
    -- Business rule: ship date must be after order date
    CONSTRAINT ship_after_order CHECK (ship_date >= order_date),
    
    -- Business rule: order date cannot be in future
    CONSTRAINT order_date_not_future CHECK (order_date <= CURRENT_DATE)
);
```

### Unique Constraints
```sql
-- Single column unique
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL
);

-- Composite unique constraint
CREATE TABLE course_sections (
    section_id SERIAL PRIMARY KEY,
    course_id INTEGER NOT NULL,
    section_number INTEGER NOT NULL,
    semester VARCHAR(20) NOT NULL,
    year INTEGER NOT NULL,
    
    -- A course can have only one section with same number per semester
    UNIQUE (course_id, section_number, semester, year)
);

-- Partial unique constraint (with conditions)
CREATE TABLE active_subscriptions (
    subscription_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    service_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    
    -- Only one active subscription per user per service type
    UNIQUE (user_id, service_type) WHERE status = 'active'
);
```

### Foreign Key Constraints with Actions
```sql
-- Referential actions
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    
    -- CASCADE: Delete orders when customer is deleted
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

CREATE TABLE order_items (
    item_id SERIAL PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    
    -- RESTRICT: Prevent deletion if referenced
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
        ON DELETE RESTRICT,
    
    -- SET NULL: Set to NULL when product is deleted
    FOREIGN KEY (product_id) REFERENCES products(product_id)
        ON DELETE SET NULL
);

-- SET DEFAULT example
CREATE TABLE employees (
    employee_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    department_id INTEGER DEFAULT 1,
    
    FOREIGN KEY (department_id) REFERENCES departments(department_id)
        ON DELETE SET DEFAULT
);
```

## Normalization Deep Dive

### First Normal Form (1NF)
```sql
-- Violation of 1NF (multiple values in single field)
-- DON'T DO THIS
CREATE TABLE bad_orders (
    order_id SERIAL PRIMARY KEY,
    customer_name VARCHAR(100),
    product_names TEXT  -- 'Product A, Product B, Product C'
);

-- Correct 1NF design
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_name VARCHAR(100)
);

CREATE TABLE order_items (
    order_id INTEGER,
    product_name VARCHAR(100),
    PRIMARY KEY (order_id, product_name),
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);
```

### Second Normal Form (2NF)
```sql
-- Violation of 2NF (partial dependency)
-- DON'T DO THIS
CREATE TABLE bad_order_items (
    order_id INTEGER,
    product_id INTEGER,
    product_name VARCHAR(100),  -- Depends only on product_id
    product_price DECIMAL(10,2), -- Depends only on product_id
    quantity INTEGER,
    PRIMARY KEY (order_id, product_id)
);

-- Correct 2NF design
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    product_name VARCHAR(100),
    product_price DECIMAL(10,2)
);

CREATE TABLE order_items (
    order_id INTEGER,
    product_id INTEGER,
    quantity INTEGER,
    PRIMARY KEY (order_id, product_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
```

### Third Normal Form (3NF)
```sql
-- Violation of 3NF (transitive dependency)
-- DON'T DO THIS
CREATE TABLE bad_employees (
    employee_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    department_id INTEGER,
    department_name VARCHAR(100),  -- Depends on department_id
    department_location VARCHAR(100) -- Depends on department_id
);

-- Correct 3NF design
CREATE TABLE departments (
    department_id SERIAL PRIMARY KEY,
    department_name VARCHAR(100),
    department_location VARCHAR(100)
);

CREATE TABLE employees (
    employee_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    department_id INTEGER,
    FOREIGN KEY (department_id) REFERENCES departments(department_id)
);
```

### Boyce-Codd Normal Form (BCNF)
```sql
-- Example requiring BCNF
CREATE TABLE course_instructors (
    course_id INTEGER,
    instructor_id INTEGER,
    classroom VARCHAR(50),
    
    -- Each instructor teaches in only one classroom
    -- But multiple instructors can teach same course
    PRIMARY KEY (course_id, instructor_id),
    UNIQUE (instructor_id, classroom)
);
```

## Advanced Modeling Patterns

### Inheritance Modeling

#### Table Per Hierarchy (Single Table)
```sql
-- All entity types in one table
CREATE TABLE contacts (
    contact_id SERIAL PRIMARY KEY,
    contact_type VARCHAR(20) NOT NULL, -- 'person', 'company'
    name VARCHAR(100) NOT NULL,
    
    -- Person-specific fields
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    birth_date DATE,
    
    -- Company-specific fields
    company_type VARCHAR(50),
    tax_id VARCHAR(20),
    
    CHECK (
        (contact_type = 'person' AND first_name IS NOT NULL AND last_name IS NOT NULL) OR
        (contact_type = 'company' AND company_type IS NOT NULL)
    )
);
```

#### Table Per Type (Joined Tables)
```sql
-- Base table
CREATE TABLE contacts (
    contact_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100),
    phone VARCHAR(20)
);

-- Person-specific table
CREATE TABLE persons (
    contact_id INTEGER PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    birth_date DATE,
    FOREIGN KEY (contact_id) REFERENCES contacts(contact_id)
);

-- Company-specific table
CREATE TABLE companies (
    contact_id INTEGER PRIMARY KEY,
    company_type VARCHAR(50),
    tax_id VARCHAR(20),
    employee_count INTEGER,
    FOREIGN KEY (contact_id) REFERENCES contacts(contact_id)
);
```

### Temporal Data Modeling

#### Valid Time (Business Time)
```sql
-- Track when data was valid in real world
CREATE TABLE employee_salaries (
    employee_id INTEGER NOT NULL,
    salary DECIMAL(10,2) NOT NULL,
    valid_from DATE NOT NULL,
    valid_to DATE NOT NULL DEFAULT '9999-12-31',
    
    PRIMARY KEY (employee_id, valid_from),
    
    CHECK (valid_from < valid_to),
    
    -- No overlapping periods for same employee
    EXCLUDE USING gist (
        employee_id WITH =,
        daterange(valid_from, valid_to, '[)') WITH &&
    )
);
```

#### Transaction Time (System Time)
```sql
-- Track when data was recorded in system
CREATE TABLE audit_log (
    log_id SERIAL PRIMARY KEY,
    table_name VARCHAR(50) NOT NULL,
    record_id INTEGER NOT NULL,
    operation VARCHAR(10) NOT NULL, -- INSERT, UPDATE, DELETE
    old_values JSONB,
    new_values JSONB,
    transaction_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    user_id INTEGER
);
```

### Flexible Schema Patterns

#### Entity-Attribute-Value (EAV)
```sql
-- For highly variable attributes
CREATE TABLE entities (
    entity_id SERIAL PRIMARY KEY,
    entity_type VARCHAR(50) NOT NULL
);

CREATE TABLE attributes (
    attribute_id SERIAL PRIMARY KEY,
    attribute_name VARCHAR(100) NOT NULL,
    data_type VARCHAR(20) NOT NULL
);

CREATE TABLE entity_attributes (
    entity_id INTEGER,
    attribute_id INTEGER,
    value_text TEXT,
    value_number DECIMAL,
    value_date DATE,
    value_boolean BOOLEAN,
    
    PRIMARY KEY (entity_id, attribute_id),
    FOREIGN KEY (entity_id) REFERENCES entities(entity_id),
    FOREIGN KEY (attribute_id) REFERENCES attributes(attribute_id)
);
```

#### JSON/Document Hybrid
```sql
-- Combine structured and unstructured data
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    category_id INTEGER NOT NULL,
    
    -- Flexible attributes as JSON
    attributes JSONB,
    
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);

-- Index on JSON attributes
CREATE INDEX idx_products_attributes_gin ON products USING gin(attributes);
CREATE INDEX idx_products_color ON products USING btree((attributes->>'color'));
```

## Data Quality and Constraints

### Domain Constraints
```sql
-- Create custom domains for reusability
CREATE DOMAIN email_type AS VARCHAR(320)
    CHECK (VALUE ~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$');

CREATE DOMAIN phone_type AS VARCHAR(20)
    CHECK (VALUE ~ '^\+?[1-9]\d{1,14}$');

CREATE DOMAIN currency_type AS DECIMAL(19,4)
    CHECK (VALUE >= 0);

-- Use domains in tables
CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email email_type,
    phone phone_type,
    credit_limit currency_type DEFAULT 0
);
```

### Assertion-like Constraints
```sql
-- Complex business rules using triggers
CREATE OR REPLACE FUNCTION check_enrollment_limit()
RETURNS TRIGGER AS $$
BEGIN
    IF (
        SELECT COUNT(*)
        FROM enrollments e
        JOIN courses c ON e.course_id = c.course_id
        WHERE e.student_id = NEW.student_id
    ) > 6 THEN
        RAISE EXCEPTION 'Student cannot enroll in more than 6 courses';
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tr_check_enrollment_limit
    BEFORE INSERT ON enrollments
    FOR EACH ROW
    EXECUTE FUNCTION check_enrollment_limit();
```

## Performance Considerations in Modeling

### Denormalization Strategies
```sql
-- Calculated/derived columns for performance
CREATE TABLE order_summaries (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date DATE NOT NULL,
    
    -- Denormalized totals (calculated from order_items)
    item_count INTEGER NOT NULL DEFAULT 0,
    subtotal DECIMAL(10,2) NOT NULL DEFAULT 0,
    tax_amount DECIMAL(10,2) NOT NULL DEFAULT 0,
    total_amount DECIMAL(10,2) NOT NULL DEFAULT 0,
    
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Maintain denormalized data with triggers
CREATE OR REPLACE FUNCTION update_order_summary()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE order_summaries
    SET 
        item_count = (
            SELECT COUNT(*)
            FROM order_items
            WHERE order_id = COALESCE(NEW.order_id, OLD.order_id)
        ),
        subtotal = (
            SELECT COALESCE(SUM(quantity * unit_price), 0)
            FROM order_items
            WHERE order_id = COALESCE(NEW.order_id, OLD.order_id)
        )
    WHERE order_id = COALESCE(NEW.order_id, OLD.order_id);
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;
```

### Partitioning-Friendly Design
```sql
-- Design with partitioning in mind
CREATE TABLE sales_transactions (
    transaction_id BIGINT,
    transaction_date DATE NOT NULL,
    customer_id INTEGER NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    
    -- Include partition key in primary key
    PRIMARY KEY (transaction_id, transaction_date)
) PARTITION BY RANGE (transaction_date);

-- Create partitions
CREATE TABLE sales_2023_q1 PARTITION OF sales_transactions
FOR VALUES FROM ('2023-01-01') TO ('2023-04-01');

CREATE TABLE sales_2023_q2 PARTITION OF sales_transactions
FOR VALUES FROM ('2023-04-01') TO ('2023-07-01');
```

## Model Documentation and Metadata

### Self-Documenting Schema
```sql
-- Add comments to tables and columns
CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE customers IS 'Customer master data table';
COMMENT ON COLUMN customers.customer_id IS 'Unique identifier for customer';
COMMENT ON COLUMN customers.name IS 'Customer full name or company name';
COMMENT ON COLUMN customers.email IS 'Primary email address for customer communication';

-- Document constraints
COMMENT ON CONSTRAINT customers_email_key ON customers 
IS 'Ensures email uniqueness across all customers';
```

### Metadata Tables
```sql
-- Track schema changes
CREATE TABLE schema_versions (
    version_id SERIAL PRIMARY KEY,
    version_number VARCHAR(20) NOT NULL,
    description TEXT,
    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    applied_by VARCHAR(100) NOT NULL
);

-- Document business rules
CREATE TABLE business_rules (
    rule_id SERIAL PRIMARY KEY,
    rule_name VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    table_name VARCHAR(100),
    column_name VARCHAR(100),
    rule_type VARCHAR(50), -- 'constraint', 'trigger', 'application'
    implementation TEXT
);
```
---

## ✅ 1. **Indexing Basics**

Indexes improve query performance, especially for large datasets.

### 🔹 Create a Basic Index

```sql
CREATE INDEX idx_customers_email ON customers(email);
```

### 🔹 Unique Index (enforced automatically with `UNIQUE` constraint)

```sql
CREATE UNIQUE INDEX idx_customers_username ON customers(username);
```

### 🔹 Drop an Index

```sql
DROP INDEX idx_customers_email;
```

---

## ✅ 2. **Types of Indexes in PostgreSQL**

| Index Type | Use Case                                      |
| ---------- | --------------------------------------------- |
| `B-TREE`   | Default; used for comparisons (`=`, `<`, etc) |
| `HASH`     | For exact matches only                        |
| `GIN`      | Full-text search, arrays, JSONB               |
| `GiST`     | Geospatial or custom range searches           |
| `BRIN`     | Large, append-only tables (e.g., logs)        |

---

## ✅ 3. **Composite Keys (Multi-column Primary Keys)**

Used when a single column can't uniquely identify a row.

```sql
CREATE TABLE enrollments (
    student_id INT,
    course_id INT,
    PRIMARY KEY (student_id, course_id)
);
```

* Both fields together must be unique.
* Often used in **many-to-many** relationship junction tables.

---

## ✅ 4. **CHECK Constraints**

Used to enforce custom logic on column values.

```sql
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    salary NUMERIC CHECK (salary > 0),
    age INT CHECK (age BETWEEN 18 AND 65)
);
```

---

## ✅ 5. **ENUMs and Custom Data Types**

### 🔹 Create an ENUM

```sql
CREATE TYPE job_role AS ENUM ('engineer', 'manager', 'analyst');
```

### 🔹 Use ENUM in a Table

```sql
CREATE TABLE staff (
    id SERIAL PRIMARY KEY,
    name TEXT,
    role job_role
);
```

---

## ✅ 6. **DEFAULT and GENERATED Columns**

### 🔹 Set Default Value

```sql
CREATE TABLE tasks (
    id SERIAL PRIMARY KEY,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 🔹 Generated Columns (PostgreSQL 12+)

```sql
CREATE TABLE sales (
    quantity INT,
    price NUMERIC,
    total NUMERIC GENERATED ALWAYS AS (quantity * price) STORED
);
```

---

## ✅ Summary

* Indexes are **critical** for performance — use them wisely based on query patterns.
* Constraints like **CHECK**, **DEFAULT**, and **ENUM** improve **data safety and integrity**.
* Composite keys and generated columns are powerful tools for accurate modeling.
