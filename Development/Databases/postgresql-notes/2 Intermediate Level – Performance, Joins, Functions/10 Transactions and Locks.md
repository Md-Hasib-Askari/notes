### 🟡 Intermediate Level – Topic 10: **Transactions and Locks**

## ✅ 1. **What is a Transaction?**

A **transaction** is a group of SQL operations that are treated as a single unit — either **all succeed** or **none do**.

---

### 🔹 Start and End a Transaction

```sql
BEGIN;

-- your queries
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;

COMMIT; -- makes changes permanent
```

```sql
ROLLBACK; -- undoes all changes in the transaction
```

> If something goes wrong before `COMMIT`, use `ROLLBACK` to undo everything.

---

## ✅ 2. **Transaction Control Summary**

| Command    | Purpose           |
| ---------- | ----------------- |
| `BEGIN`    | Start transaction |
| `COMMIT`   | Save changes      |
| `ROLLBACK` | Undo changes      |

---

## ✅ 3. **Savepoints**

Let you roll back **part** of a transaction.

```sql
BEGIN;
    UPDATE customers SET name = 'Hasib Updated' WHERE id = 1;
    SAVEPOINT before_email;
    UPDATE customers SET email = 'invalid email';
    ROLLBACK TO SAVEPOINT before_email;
COMMIT;
```

---

## ✅ 4. **Isolation Levels**

PostgreSQL supports 4 levels:

| Level              | Description                                 |
| ------------------ | ------------------------------------------- |
| `READ UNCOMMITTED` | Not supported; defaults to `READ COMMITTED` |
| `READ COMMITTED`   | Default; sees only committed changes        |
| `REPEATABLE READ`  | Prevents dirty and non-repeatable reads     |
| `SERIALIZABLE`     | Strictest; guarantees full isolation        |

### Example:

```sql
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
```

---

## ✅ 5. **Locks in PostgreSQL**

Locks prevent data conflicts when multiple users access the same data.

---

### 🔹 Implicit Locks (Handled Automatically)

* `SELECT FOR UPDATE` – locks rows for update

```sql
SELECT * FROM accounts WHERE id = 1 FOR UPDATE;
```

---

### 🔹 Explicit Locks

```sql
LOCK TABLE accounts IN EXCLUSIVE MODE;
```

| Lock Type       | Use Case                         |
| --------------- | -------------------------------- |
| `ACCESS SHARE`  | Reading data                     |
| `ROW EXCLUSIVE` | For `INSERT`, `UPDATE`, `DELETE` |
| `EXCLUSIVE`     | Block all reads/writes           |

---

## ✅ 6. **Deadlocks**

A **deadlock** occurs when two transactions wait on each other’s lock forever.

PostgreSQL will **detect and cancel** one of them.

**Avoid it by:**

* Locking tables in the same order across transactions
* Keeping transactions short

---

## ✅ Summary

* Transactions ensure **atomicity** — all-or-nothing changes.
* Use `SAVEPOINT` for partial rollbacks.
* Locks ensure **data integrity** in concurrent environments.
* Understand isolation levels to choose the right trade-off between performance and safety.

---

# Transactions and Locks

## Introduction to Transactions

Transactions are fundamental units of work that ensure data consistency and integrity. They follow ACID properties (Atomicity, Consistency, Isolation, Durability) to maintain database reliability.

## Basic Transaction Control

### Transaction Syntax
```sql
-- Basic transaction
BEGIN;
    INSERT INTO customers (name, email) VALUES ('John Doe', 'john@example.com');
    INSERT INTO orders (customer_id, total_amount) VALUES (1, 100.00);
COMMIT;

-- Transaction with rollback
BEGIN;
    UPDATE products SET price = price * 1.1 WHERE category_id = 1;
    -- Something goes wrong...
ROLLBACK;

-- Using SAVEPOINT for partial rollback
BEGIN;
    INSERT INTO customers (name, email) VALUES ('Jane Smith', 'jane@example.com');
    SAVEPOINT sp1;
    
    UPDATE products SET price = price * 0.9; -- Bulk price change
    -- Oops, this was wrong
    ROLLBACK TO SAVEPOINT sp1;
    
    INSERT INTO orders (customer_id, total_amount) VALUES (2, 75.00);
COMMIT;
```

### Transaction States
```sql
-- Check transaction status
SELECT 
    pid,
    state,
    query_start,
    state_change,
    query
FROM pg_stat_activity 
WHERE state IN ('active', 'idle in transaction');

-- Current transaction ID
SELECT txid_current();

-- Transaction snapshot info
SELECT txid_current_snapshot();
```

## Isolation Levels

### Understanding Isolation Levels
PostgreSQL supports four isolation levels that control how transactions interact with each other:

1. **READ UNCOMMITTED** (rarely used)
2. **READ COMMITTED** (default)
3. **REPEATABLE READ**
4. **SERIALIZABLE**

### READ COMMITTED (Default)
```sql
-- Session 1
BEGIN;
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
SELECT balance FROM accounts WHERE account_id = 1; -- Returns 1000
-- ... wait for Session 2 to commit ...
SELECT balance FROM accounts WHERE account_id = 1; -- Returns 900 (sees committed changes)
COMMIT;

-- Session 2 (concurrent)
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE account_id = 1;
COMMIT; -- This makes change visible to Session 1
```

### REPEATABLE READ
```sql
-- Session 1
BEGIN;
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SELECT balance FROM accounts WHERE account_id = 1; -- Returns 1000
-- ... Session 2 commits changes ...
SELECT balance FROM accounts WHERE account_id = 1; -- Still returns 1000 (snapshot isolation)
COMMIT;

-- Session 2 (concurrent)
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE account_id = 1;
COMMIT; -- Change not visible to Session 1 until it commits
```

### SERIALIZABLE
```sql
-- Session 1
BEGIN;
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
SELECT SUM(balance) FROM accounts; -- Total: 5000
INSERT INTO accounts (name, balance) VALUES ('New Account', 500);
COMMIT; -- May fail with serialization error if concurrent modifications

-- Session 2 (concurrent)
BEGIN;
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
SELECT SUM(balance) FROM accounts; -- Total: 5000
UPDATE accounts SET balance = balance + 100 WHERE account_id = 1;
COMMIT; -- One of these transactions will be rolled back
```

### Setting Isolation Levels
```sql
-- For single transaction
BEGIN;
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
-- ... transaction statements ...
COMMIT;

-- For session
SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ;

-- For specific transaction characteristics
BEGIN ISOLATION LEVEL SERIALIZABLE READ WRITE;
-- ... transaction statements ...
COMMIT;
```

## Concurrency Phenomena

### Dirty Read
```sql
-- Prevented at READ COMMITTED and higher levels
-- Session 1
BEGIN;
UPDATE products SET price = 999.99 WHERE product_id = 1;
-- Don't commit yet

-- Session 2 (at READ UNCOMMITTED - not recommended)
BEGIN;
SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;
SELECT price FROM products WHERE product_id = 1; -- Would see 999.99 (dirty read)
COMMIT;
```

### Non-repeatable Read
```sql
-- Prevented at REPEATABLE READ and higher levels
-- Session 1
BEGIN;
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
SELECT price FROM products WHERE product_id = 1; -- Returns 100.00
-- ... Session 2 commits price change ...
SELECT price FROM products WHERE product_id = 1; -- Returns 150.00 (non-repeatable read)
COMMIT;
```

### Phantom Read
```sql
-- Prevented at SERIALIZABLE level
-- Session 1
BEGIN;
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SELECT COUNT(*) FROM products WHERE category_id = 1; -- Returns 5
-- ... Session 2 inserts new product in category 1 ...
SELECT COUNT(*) FROM products WHERE category_id = 1; -- Still returns 5 in REPEATABLE READ
COMMIT;
```

## Locking Mechanisms

### Table-Level Locks
```sql
-- Explicit table locking
BEGIN;
LOCK TABLE products IN ACCESS EXCLUSIVE MODE; -- Blocks all access
-- Perform bulk operations
UPDATE products SET price = price * 1.1;
COMMIT;

-- Different lock modes
LOCK TABLE inventory IN SHARE MODE; -- Allows concurrent reads
LOCK TABLE orders IN ROW EXCLUSIVE MODE; -- Allows concurrent reads and row-level writes
```

### Row-Level Locks
```sql
-- SELECT FOR UPDATE (exclusive row lock)
BEGIN;
SELECT * FROM accounts 
WHERE account_id = 1 
FOR UPDATE; -- Locks this specific row

UPDATE accounts SET balance = balance - 100 WHERE account_id = 1;
COMMIT;

-- SELECT FOR SHARE (shared row lock)
BEGIN;
SELECT * FROM products 
WHERE product_id IN (1, 2, 3) 
FOR SHARE; -- Allows other shared locks, blocks exclusive

-- Verify inventory levels without modification
COMMIT;

-- NO WAIT option
BEGIN;
SELECT * FROM accounts 
WHERE account_id = 1 
FOR UPDATE NOWAIT; -- Fails immediately if row is locked
```

### Advisory Locks
```sql
-- Session-level advisory locks
SELECT pg_advisory_lock(12345); -- Acquires exclusive lock with ID 12345
-- ... critical section ...
SELECT pg_advisory_unlock(12345); -- Releases lock

-- Transaction-level advisory locks
BEGIN;
SELECT pg_advisory_xact_lock(12345); -- Lock released at transaction end
-- ... critical section ...
COMMIT; -- Lock automatically released

-- Try to acquire lock without waiting
SELECT pg_try_advisory_lock(12345); -- Returns true if acquired, false if not

-- Shared advisory locks
SELECT pg_advisory_lock_shared(12345); -- Multiple sessions can hold shared lock
```

## Deadlock Detection and Prevention

### Understanding Deadlocks
```sql
-- Example of potential deadlock scenario
-- Session 1
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE account_id = 1;
-- Now tries to update account 2...
UPDATE accounts SET balance = balance + 100 WHERE account_id = 2; -- May deadlock

-- Session 2 (concurrent)
BEGIN;
UPDATE accounts SET balance = balance - 50 WHERE account_id = 2;
-- Now tries to update account 1...
UPDATE accounts SET balance = balance + 50 WHERE account_id = 1; -- May deadlock
```

### Deadlock Prevention Strategies
```sql
-- Strategy 1: Consistent ordering
CREATE OR REPLACE FUNCTION transfer_funds(
    from_account INTEGER,
    to_account INTEGER,
    amount DECIMAL
)
RETURNS BOOLEAN AS $$
DECLARE
    account1 INTEGER;
    account2 INTEGER;
BEGIN
    -- Always lock accounts in consistent order (lower ID first)
    IF from_account < to_account THEN
        account1 := from_account;
        account2 := to_account;
    ELSE
        account1 := to_account;
        account2 := from_account;
    END IF;
    
    -- Lock in consistent order
    PERFORM balance FROM accounts WHERE account_id = account1 FOR UPDATE;
    PERFORM balance FROM accounts WHERE account_id = account2 FOR UPDATE;
    
    -- Perform transfer
    UPDATE accounts SET balance = balance - amount WHERE account_id = from_account;
    UPDATE accounts SET balance = balance + amount WHERE account_id = to_account;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Strategy 2: Timeout handling
SET deadlock_timeout = '1s'; -- Adjust deadlock detection sensitivity

-- Strategy 3: Using NOWAIT
BEGIN;
SELECT * FROM accounts WHERE account_id = 1 FOR UPDATE NOWAIT;
-- Handle lock acquisition failure gracefully
EXCEPTION
    WHEN lock_not_available THEN
        RAISE NOTICE 'Could not acquire lock, retrying later';
        ROLLBACK;
```

### Deadlock Monitoring
```sql
-- Monitor deadlocks
SELECT 
    datname,
    deadlocks
FROM pg_stat_database
WHERE deadlocks > 0;

-- Check current locks
SELECT 
    l.locktype,
    l.database,
    l.relation::regclass,
    l.page,
    l.tuple,
    l.virtualxid,
    l.transactionid,
    l.mode,
    l.granted,
    a.query
FROM pg_locks l
LEFT JOIN pg_stat_activity a ON l.pid = a.pid
WHERE NOT l.granted
ORDER BY l.pid;
```

## Transaction Patterns and Best Practices

### Optimistic Locking Pattern
```sql
-- Using version column for optimistic locking
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER,
    total_amount DECIMAL(10,2),
    version_number INTEGER DEFAULT 1,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Update with version check
CREATE OR REPLACE FUNCTION update_order_optimistic(
    p_order_id INTEGER,
    p_total_amount DECIMAL,
    p_expected_version INTEGER
)
RETURNS BOOLEAN AS $$
DECLARE
    rows_affected INTEGER;
BEGIN
    UPDATE orders 
    SET 
        total_amount = p_total_amount,
        version_number = version_number + 1,
        updated_at = CURRENT_TIMESTAMP
    WHERE 
        order_id = p_order_id 
        AND version_number = p_expected_version;
    
    GET DIAGNOSTICS rows_affected = ROW_COUNT;
    
    IF rows_affected = 0 THEN
        RAISE EXCEPTION 'Order has been modified by another user';
    END IF;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;
```

### Batch Processing with Chunking
```sql
-- Process large datasets in chunks to avoid long-running transactions
CREATE OR REPLACE FUNCTION process_orders_in_batches(batch_size INTEGER DEFAULT 1000)
RETURNS INTEGER AS $$
DECLARE
    processed_count INTEGER := 0;
    batch_count INTEGER;
BEGIN
    LOOP
        BEGIN;
            -- Process one batch
            WITH batch AS (
                SELECT order_id
                FROM orders
                WHERE status = 'pending'
                ORDER BY order_id
                LIMIT batch_size
                FOR UPDATE SKIP LOCKED
            )
            UPDATE orders
            SET 
                status = 'processed',
                processed_at = CURRENT_TIMESTAMP
            FROM batch
            WHERE orders.order_id = batch.order_id;
            
            GET DIAGNOSTICS batch_count = ROW_COUNT;
            processed_count := processed_count + batch_count;
            
            COMMIT;
            
            -- Exit if no more rows to process
            IF batch_count = 0 THEN
                EXIT;
            END IF;
            
            -- Optional: Add delay between batches
            PERFORM pg_sleep(0.1);
        EXCEPTION
            WHEN OTHERS THEN
                ROLLBACK;
                RAISE NOTICE 'Error processing batch: %', SQLERRM;
                EXIT;
        END;
    END LOOP;
    
    RETURN processed_count;
END;
$$ LANGUAGE plpgsql;
```

### Saga Pattern for Distributed Transactions
```sql
-- Compensating transaction pattern
CREATE TABLE saga_events (
    event_id SERIAL PRIMARY KEY,
    saga_id UUID,
    step_name TEXT,
    event_type TEXT, -- 'start', 'success', 'failure', 'compensate'
    event_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE OR REPLACE FUNCTION execute_order_saga(p_order_data JSONB)
RETURNS UUID AS $$
DECLARE
    saga_id UUID := gen_random_uuid();
    order_id INTEGER;
    payment_id INTEGER;
    inventory_reserved BOOLEAN := FALSE;
BEGIN
    -- Step 1: Create order
    INSERT INTO saga_events (saga_id, step_name, event_type, event_data)
    VALUES (saga_id, 'create_order', 'start', p_order_data);
    
    BEGIN
        INSERT INTO orders (customer_id, total_amount)
        VALUES ((p_order_data->>'customer_id')::INTEGER, (p_order_data->>'total_amount')::DECIMAL)
        RETURNING order_id INTO order_id;
        
        INSERT INTO saga_events (saga_id, step_name, event_type, event_data)
        VALUES (saga_id, 'create_order', 'success', jsonb_build_object('order_id', order_id));
    EXCEPTION
        WHEN OTHERS THEN
            INSERT INTO saga_events (saga_id, step_name, event_type, event_data)
            VALUES (saga_id, 'create_order', 'failure', jsonb_build_object('error', SQLERRM));
            RETURN saga_id;
    END;
    
    -- Step 2: Reserve inventory
    -- ... similar pattern for each step
    
    -- If any step fails, execute compensating transactions
    -- ... compensation logic
    
    RETURN saga_id;
END;
$$ LANGUAGE plpgsql;
```

## Performance Tuning for Transactions

### Configuration Parameters
```sql
-- View current transaction-related settings
SELECT name, setting, unit, short_desc
FROM pg_settings
WHERE name IN (
    'deadlock_timeout',
    'lock_timeout',
    'statement_timeout',
    'idle_in_transaction_session_timeout',
    'max_locks_per_transaction'
);

-- Adjust settings for your workload
-- In postgresql.conf or via SET
SET deadlock_timeout = '2s';
SET lock_timeout = '10s';
SET statement_timeout = '30s';
SET idle_in_transaction_session_timeout = '5min';
```

### Transaction Monitoring
```sql
-- Monitor long-running transactions
SELECT 
    pid,
    now() - xact_start as duration,
    state,
    query
FROM pg_stat_activity
WHERE xact_start IS NOT NULL
ORDER BY xact_start;

-- Monitor lock waits
SELECT 
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_statement,
    blocking_activity.query AS current_statement_in_blocking_process
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks 
    ON blocking_locks.locktype = blocked_locks.locktype
    AND blocking_locks.database IS NOT DISTINCT FROM blocked_locks.database
    AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
    AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
    AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
    AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
    AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
    AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
    AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
    AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
    AND blocking_locks.pid != blocked_locks.pid
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;
```

### Best Practices Summary

1. **Keep transactions short** - Minimize lock duration
2. **Consistent lock ordering** - Prevent deadlocks
3. **Use appropriate isolation levels** - Balance consistency and performance
4. **Handle deadlocks gracefully** - Implement retry logic
5. **Monitor lock waits** - Identify bottlenecks
6. **Use advisory locks** - For application-level coordination
7. **Batch large operations** - Process in smaller chunks
8. **Set appropriate timeouts** - Prevent hung transactions

## Common Transaction Anti-patterns

### Anti-pattern: Long-running transactions
```sql
-- DON'T: Keep transaction open while waiting for user input
BEGIN;
SELECT * FROM products FOR UPDATE;
-- ... wait for user to make decision ...
UPDATE products SET price = ? WHERE id = ?;
COMMIT;

-- DO: Keep transaction as short as possible
-- Get data first
SELECT * FROM products WHERE id = ?;
-- ... user makes decision ...
-- Then update in short transaction
BEGIN;
UPDATE products SET price = ? WHERE id = ? AND version = ?;
COMMIT;
```

### Anti-pattern: Unnecessary locking
```sql
-- DON'T: Lock more than necessary
BEGIN;
SELECT * FROM orders FOR UPDATE; -- Locks all orders
UPDATE orders SET status = 'processed' WHERE order_id = 123;
COMMIT;

-- DO: Lock only what you need
BEGIN;
SELECT * FROM orders WHERE order_id = 123 FOR UPDATE;
UPDATE orders SET status = 'processed' WHERE order_id = 123;
COMMIT;
```
