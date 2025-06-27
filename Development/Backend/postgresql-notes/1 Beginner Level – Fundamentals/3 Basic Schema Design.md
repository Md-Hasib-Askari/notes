### 🟢 Beginner Level – Topic 3: **Basic Schema Design**

## Schema Design Fundamentals

Schema design is the process of defining the structure, organization, and relationships of data in a database.

## Key Principles

### 1. Entity Identification
- Identify real-world objects (entities)
- Define attributes for each entity
- Determine relationships between entities

### 2. Attribute Definition
- Choose appropriate data types
- Define constraints and rules
- Consider nullable vs non-nullable fields

### 3. Relationship Modeling
- One-to-One (1:1)
- One-to-Many (1:M)
- Many-to-Many (M:M)

## Entity-Relationship (ER) Modeling

### Entities
Physical or conceptual objects that exist independently:
```
Customer: customer_id, name, email, phone
Product: product_id, name, price, description
Order: order_id, order_date, total_amount
```

### Attributes
Properties that describe entities:
- **Simple**: Cannot be divided (e.g., age)
- **Composite**: Can be subdivided (e.g., address)
- **Single-valued**: One value per entity (e.g., birth_date)
- **Multi-valued**: Multiple values per entity (e.g., phone_numbers)

### Relationships
Associations between entities:
- **One-to-One**: Each entity relates to exactly one other entity
- **One-to-Many**: One entity relates to many others
- **Many-to-Many**: Multiple entities relate to multiple others

## Common Design Patterns

### 1. User Management System
```sql
-- Users table
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User profiles table (1:1 relationship)
CREATE TABLE user_profiles (
    profile_id SERIAL PRIMARY KEY,
    user_id INTEGER UNIQUE REFERENCES users(user_id),
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    bio TEXT,
    avatar_url VARCHAR(255)
);
```

### 2. E-commerce System
```sql
-- Categories table
CREATE TABLE categories (
    category_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    parent_id INTEGER REFERENCES categories(category_id)
);

-- Products table
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    price DECIMAL(10,2) NOT NULL,
    category_id INTEGER REFERENCES categories(category_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Orders table
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'pending',
    total_amount DECIMAL(10,2)
);

-- Order items table (M:M relationship resolver)
CREATE TABLE order_items (
    order_item_id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(order_id),
    product_id INTEGER REFERENCES products(product_id),
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL
);
```

### 3. Blog System
```sql
-- Authors table
CREATE TABLE authors (
    author_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    bio TEXT
);

-- Posts table
CREATE TABLE posts (
    post_id SERIAL PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    content TEXT NOT NULL,
    author_id INTEGER REFERENCES authors(author_id),
    published_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tags table
CREATE TABLE tags (
    tag_id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL
);

-- Post tags table (M:M relationship)
CREATE TABLE post_tags (
    post_id INTEGER REFERENCES posts(post_id),
    tag_id INTEGER REFERENCES tags(tag_id),
    PRIMARY KEY (post_id, tag_id)
);
```

## Data Type Selection

### Text Data
- **CHAR(n)**: Fixed length, padded with spaces
- **VARCHAR(n)**: Variable length, up to n characters
- **TEXT**: Unlimited variable length

### Numeric Data
- **INTEGER**: 4-byte signed integer (-2M to +2M)
- **BIGINT**: 8-byte signed integer
- **DECIMAL(p,s)**: Exact numeric with precision and scale
- **REAL**: 4-byte floating point
- **DOUBLE PRECISION**: 8-byte floating point

### Date and Time
- **DATE**: Date only (YYYY-MM-DD)
- **TIME**: Time only (HH:MM:SS)
- **TIMESTAMP**: Date and time
- **TIMESTAMPTZ**: Timestamp with timezone

### Boolean and Other
- **BOOLEAN**: TRUE/FALSE/NULL
- **UUID**: Universally unique identifier
- **JSON/JSONB**: JSON data storage

## Naming Conventions

### Tables
- Use lowercase with underscores
- Use plural nouns: `users`, `products`, `orders`
- Be descriptive and consistent

### Columns
- Use lowercase with underscores
- Use singular nouns: `user_id`, `first_name`
- Prefix foreign keys with table name: `user_id`, `category_id`

### Constraints
- Primary keys: `pk_table_name`
- Foreign keys: `fk_table_column`
- Unique constraints: `uk_table_column`
- Check constraints: `ck_table_condition`

## Schema Design Best Practices

### 1. Start Simple
- Begin with core entities
- Add complexity gradually
- Iterate based on requirements

### 2. Normalize Appropriately
- Apply normalization rules
- Consider denormalization for performance
- Balance between flexibility and efficiency

### 3. Use Constraints
- Enforce data integrity at database level
- Use NOT NULL for required fields
- Add CHECK constraints for business rules

### 4. Plan for Growth
- Consider future requirements
- Leave room for extensibility
- Use appropriate data types for scale

### 5. Document Everything
- Comment tables and columns
- Maintain ER diagrams
- Document business rules

## Common Anti-Patterns to Avoid

### 1. Generic Tables
```sql
-- BAD: Generic entity-attribute-value table
CREATE TABLE entities (
    id SERIAL PRIMARY KEY,
    entity_type VARCHAR(50),
    attribute_name VARCHAR(50),
    attribute_value TEXT
);

-- GOOD: Specific tables for each entity
CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100)
);
```

### 2. Storing Multiple Values in Single Column
```sql
-- BAD: Comma-separated values
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    product_ids VARCHAR(255)  -- '1,2,3,4'
);

-- GOOD: Separate junction table
CREATE TABLE order_items (
    order_id INTEGER,
    product_id INTEGER,
    PRIMARY KEY (order_id, product_id)
);
```

### 3. Inappropriate NULL Usage
```sql
-- BAD: Using NULL for empty strings
INSERT INTO users (name) VALUES (NULL);

-- GOOD: Use empty string or make field required
ALTER TABLE users ALTER COLUMN name SET NOT NULL;
```

## Schema Evolution

### Version Control
- Track schema changes
- Use migration scripts
- Test migrations thoroughly

### Backward Compatibility
- Avoid breaking changes
- Add new columns as nullable
- Deprecate rather than drop immediately

### Rollback Strategy
- Always have rollback plans
- Test rollback procedures
- Keep backups before major changes


---

#### ✅ What is a Schema?

A **schema** is a logical namespace inside a database:

* Organizes tables, views, functions, etc.
* Default schema is `public`

**🧪 Create a schema:**

```sql
CREATE SCHEMA analytics;
```

**🧪 Use objects from a schema:**

```sql
SELECT * FROM analytics.sales;
```

---

### ✅ Creating Tables with Constraints

#### Basic Table Example:

```sql
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    age INTEGER CHECK (age >= 18),
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

#### ✅ Common Constraints

| Constraint    | Purpose                              | Example                              |
| ------------- | ------------------------------------ | ------------------------------------ |
| `PRIMARY KEY` | Uniquely identifies a row            | `id SERIAL PRIMARY KEY`              |
| `NOT NULL`    | Ensures column cannot be NULL        | `name TEXT NOT NULL`                 |
| `UNIQUE`      | Ensures unique values in a column    | `email TEXT UNIQUE`                  |
| `CHECK`       | Validates a condition                | `age INTEGER CHECK (age > 0)`        |
| `DEFAULT`     | Sets a default value if not provided | `created_at TIMESTAMP DEFAULT NOW()` |

---

#### ✅ Foreign Keys (Relationships)

Used to enforce relationships between tables.

**🧪 Create related tables:**

```sql
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    total NUMERIC(10, 2),
    order_date DATE DEFAULT CURRENT_DATE
);
```

* `customer_id` must exist in `customers.id`

---

### ✅ Normalization Basics

**Goal**: Reduce data redundancy and improve integrity.

#### 🔹 1NF (First Normal Form)

* Atomic values (no lists/arrays in a column)

#### 🔹 2NF (Second Normal Form)

* No partial dependencies (used with composite keys)

#### 🔹 3NF (Third Normal Form)

* No transitive dependencies (non-key fields should depend only on the primary key)

---

#### ✅ Describing a Table

In `psql`:

```sql
\d customers
```

---

### ✅ ERD (Entity Relationship Diagram)

**Not required to memorize**, but useful tools to design schema:

* dbdiagram.io
* QuickDBD
* pgModeler

---

### ✅ Summary

* Tables should be designed with **constraints** to enforce business rules.
* Use **foreign keys** for relationships between entities.
* Normalize tables to avoid redundancy and ensure clean data modeling.
