### 🔵 Advanced Level – Topic 13: **Security and Access Control**

---

PostgreSQL offers a rich and flexible security model to **control who can access what**, down to the row level.

---

## ✅ 1. **Roles and Privileges**

Roles = Users or Groups

### 🔹 Create a Role

```sql
CREATE ROLE analyst WITH LOGIN PASSWORD 'strongpass';
```

### 🔹 Grant Privileges

```sql
GRANT SELECT ON ALL TABLES IN SCHEMA public TO analyst;
```

### 🔹 Make a Role a Superuser (⚠ Dangerous)

```sql
ALTER ROLE devuser WITH SUPERUSER;
```

### 🔹 Check Permissions

```sql
\du  -- list roles
\z tablename  -- show table privileges
```

---

## ✅ 2. **Object-level Permissions**

| Command         | Grants Rights To       |
| --------------- | ---------------------- |
| `GRANT SELECT`  | Read from table/view   |
| `GRANT INSERT`  | Add data               |
| `GRANT UPDATE`  | Modify data            |
| `GRANT DELETE`  | Remove rows            |
| `GRANT USAGE`   | Use schema or sequence |
| `GRANT EXECUTE` | Call a function        |

### 🔹 Example

```sql
GRANT SELECT, INSERT ON orders TO analyst;
```

### 🔹 Revoke Permissions

```sql
REVOKE ALL ON orders FROM analyst;
```

---

## ✅ 3. **Schema Security**

By default, `public` schema is accessible by all roles.

🔐 Secure it:

```sql
REVOKE ALL ON SCHEMA public FROM PUBLIC;
```

Create isolated schema:

```sql
CREATE SCHEMA private_schema AUTHORIZATION secure_user;
```

---

## ✅ 4. **pg\_hba.conf – Authentication Control**

The file that controls **how users connect** to PostgreSQL.

### 🔹 Location (Ubuntu/Debian)

```
/etc/postgresql/<version>/main/pg_hba.conf
```

### 🔹 Common Methods:

| Method          | Description                   |
| --------------- | ----------------------------- |
| `trust`         | No password                   |
| `md5`           | Password required             |
| `scram-sha-256` | More secure password method   |
| `peer`          | System username = DB username |

After changes:

```bash
sudo systemctl reload postgresql
```

---

## ✅ 5. **SSL/TLS Encryption**

Enable encrypted connections:

* Configure `postgresql.conf` and `pg_hba.conf`
* Set `ssl = on`
* Provide `.crt` and `.key` files

🧠 Also supported by all PostgreSQL GUI tools

---

## ✅ 6. **Row-Level Security (RLS)**

PostgreSQL 9.5+ allows **per-row access control**.

### 🔹 Step-by-step:

```sql
-- 1. Enable RLS
ALTER TABLE customers ENABLE ROW LEVEL SECURITY;

-- 2. Add a policy
CREATE POLICY own_rows ON customers
  USING (id = current_setting('app.current_user')::INT);
```

```sql
-- 3. Set the variable
SET app.current_user = '1';
SELECT * FROM customers;  -- will only see their own rows
```

---

## ✅ 7. **Security Extensions**

| Extension  | Purpose                        |
| ---------- | ------------------------------ |
| `pgcrypto` | Column-level encryption        |
| `citext`   | Case-insensitive text          |
| `sepgsql`  | SELinux integration (advanced) |

---

## ✅ Summary

* Use roles and grants to control user access precisely.
* Lock down schemas, tables, and functions based on need.
* Use **pg\_hba.conf** and **SSL** for safe connections.
* RLS lets you enforce **per-user row visibility** — a rare feature in most DBMSs.

---

# Security and Access Control

## PostgreSQL Security Architecture

PostgreSQL implements a multi-layered security model that includes authentication, authorization, encryption, and auditing capabilities.

### Security Layers
1. **Network Security** - Connection filtering and SSL/TLS
2. **Authentication** - User identity verification
3. **Authorization** - Role-based access control (RBAC)
4. **Object-level Security** - Schema, table, column permissions
5. **Row-level Security** - Fine-grained data access
6. **Data Encryption** - At rest and in transit
7. **Auditing** - Activity monitoring and logging

## User Management and Roles

### Creating Users and Roles
```sql
-- Create a login role (user)
CREATE USER app_user WITH PASSWORD 'secure_password123';

-- Create a group role (cannot login by default)
CREATE ROLE app_readers;
CREATE ROLE app_writers;
CREATE ROLE app_admins;

-- Create role with specific attributes
CREATE ROLE data_analyst WITH 
    LOGIN 
    PASSWORD 'analyst_password'
    VALID UNTIL '2024-12-31'
    CONNECTION LIMIT 5;

-- Create role with advanced options
CREATE ROLE backup_user WITH
    LOGIN
    PASSWORD 'backup_pass'
    REPLICATION
    BYPASSRLS; -- Bypass row-level security

-- Modify existing roles
ALTER ROLE app_user WITH PASSWORD 'new_password';
ALTER ROLE app_user VALID UNTIL '2025-01-01';
ALTER ROLE app_user CONNECTION LIMIT 10;
```

### Role Hierarchies and Inheritance
```sql
-- Create hierarchical roles
CREATE ROLE base_user;
CREATE ROLE power_user INHERIT;
CREATE ROLE admin_user INHERIT;

-- Grant roles to other roles (inheritance)
GRANT base_user TO power_user;
GRANT power_user TO admin_user;

-- Grant role to specific user
GRANT app_readers TO app_user;
GRANT app_writers TO data_analyst;

-- Check role memberships
SELECT 
    r.rolname as role_name,
    m.rolname as member_name,
    g.rolname as granted_by
FROM pg_roles r
JOIN pg_auth_members am ON r.oid = am.roleid
JOIN pg_roles m ON am.member = m.oid
JOIN pg_roles g ON am.grantor = g.oid
ORDER BY r.rolname, m.rolname;
```

### Role Attributes and Capabilities
```sql
-- View role attributes
SELECT 
    rolname,
    rolsuper,
    rolinherit,
    rolcreaterole,
    rolcreatedb,
    rolcanlogin,
    rolreplication,
    rolbypassrls,
    rolconnlimit,
    rolvaliduntil
FROM pg_roles
ORDER BY rolname;

-- Create specialized roles
CREATE ROLE db_creator WITH CREATEDB LOGIN PASSWORD 'creator_pass';
CREATE ROLE role_manager WITH CREATEROLE LOGIN PASSWORD 'manager_pass';
CREATE ROLE superuser_role WITH SUPERUSER LOGIN PASSWORD 'super_pass';

-- Remove specific privileges
ALTER ROLE app_user NOCREATEDB;
ALTER ROLE app_user NOSUPERUSER;
```

## Authentication Methods

### pg_hba.conf Configuration
```conf
# TYPE  DATABASE        USER            ADDRESS                 METHOD

# Local connections
local   all             postgres                                peer
local   all             all                                     md5

# IPv4 local connections
host    all             all             127.0.0.1/32            md5
host    myapp           app_user        192.168.1.0/24          md5

# IPv6 local connections
host    all             all             ::1/128                 md5

# SSL connections only
hostssl all             all             0.0.0.0/0               md5

# Certificate authentication
hostssl all             all             192.168.1.0/24          cert

# LDAP authentication
host    all             all             10.0.0.0/8              ldap ldapserver=ldap.company.com ldapbasedn="dc=company,dc=com"

# Kerberos authentication
host    all             all             10.0.0.0/8              gss

# Reject specific connections
host    all             blocked_user    0.0.0.0/0               reject
```

### SSL/TLS Configuration
```sql
-- Check SSL status
SELECT ssl, version FROM pg_stat_ssl WHERE pid = pg_backend_pid();

-- Force SSL for specific users
ALTER USER app_user SET ssl TO on;

-- Certificate-based authentication setup
-- In postgresql.conf:
-- ssl = on
-- ssl_cert_file = 'server.crt'
-- ssl_key_file = 'server.key'
-- ssl_ca_file = 'ca.crt'

-- Client certificate mapping in pg_ident.conf:
-- MAPNAME       SYSTEM-USERNAME         PG-USERNAME
-- cert-map      john.doe                app_user
```

### Password Policies
```sql
-- Create extension for password checking
CREATE EXTENSION passwordcheck;

-- Set password encryption method
SET password_encryption = 'scram-sha-256';

-- Create user with encrypted password
CREATE USER secure_user WITH PASSWORD 'SecurePass123!';

-- Check password encryption method
SELECT usename, usepassword FROM pg_shadow WHERE usename = 'secure_user';
```

## Permissions and Privileges

### Database and Schema Privileges
```sql
-- Grant database privileges
GRANT CONNECT ON DATABASE myapp TO app_user;
GRANT TEMPORARY ON DATABASE myapp TO app_user;

-- Create schema with specific owner
CREATE SCHEMA app_data AUTHORIZATION app_owner;

-- Grant schema privileges
GRANT USAGE ON SCHEMA app_data TO app_readers;
GRANT CREATE ON SCHEMA app_data TO app_writers;
GRANT ALL ON SCHEMA app_data TO app_admins;

-- Default privileges for new objects
ALTER DEFAULT PRIVILEGES IN SCHEMA app_data 
GRANT SELECT ON TABLES TO app_readers;

ALTER DEFAULT PRIVILEGES IN SCHEMA app_data 
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO app_writers;

ALTER DEFAULT PRIVILEGES IN SCHEMA app_data 
GRANT USAGE ON SEQUENCES TO app_writers;
```

### Table and Column Privileges
```sql
-- Table-level privileges
GRANT SELECT ON customers TO app_readers;
GRANT SELECT, INSERT, UPDATE, DELETE ON orders TO app_writers;
GRANT ALL PRIVILEGES ON products TO app_admins;

-- Column-level privileges
GRANT SELECT (customer_id, name, email) ON customers TO customer_service;
GRANT UPDATE (status, updated_at) ON orders TO order_processor;

-- Revoke privileges
REVOKE DELETE ON customers FROM app_writers;
REVOKE ALL ON sensitive_table FROM public;

-- Grant with grant option
GRANT SELECT ON public_data TO data_analyst WITH GRANT OPTION;

-- Check privileges
SELECT 
    table_schema,
    table_name,
    privilege_type,
    grantee,
    is_grantable
FROM information_schema.table_privileges
WHERE table_schema = 'app_data'
ORDER BY table_name, grantee;
```

### Function and Sequence Privileges
```sql
-- Function privileges
GRANT EXECUTE ON FUNCTION calculate_discount(DECIMAL, DECIMAL) TO app_users;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA app_data TO app_writers;

-- Sequence privileges
GRANT USAGE ON SEQUENCE customers_id_seq TO app_writers;
GRANT ALL ON ALL SEQUENCES IN SCHEMA app_data TO app_admins;

-- View function permissions
SELECT 
    routine_schema,
    routine_name,
    privilege_type,
    grantee
FROM information_schema.routine_privileges
WHERE routine_schema = 'app_data';
```

## Row-Level Security (RLS)

### Basic Row-Level Security
```sql
-- Enable RLS on table
ALTER TABLE customers ENABLE ROW LEVEL SECURITY;

-- Create basic policy
CREATE POLICY customer_isolation ON customers
    FOR ALL
    TO app_users
    USING (customer_id = current_setting('app.current_customer_id')::INTEGER);

-- Users can only see their own records
SET app.current_customer_id = '123';
SELECT * FROM customers; -- Only shows customer 123's data

-- Create separate policies for different operations
CREATE POLICY customer_select ON customers
    FOR SELECT
    TO app_users
    USING (customer_id = current_setting('app.current_customer_id')::INTEGER);

CREATE POLICY customer_update ON customers
    FOR UPDATE
    TO app_users
    USING (customer_id = current_setting('app.current_customer_id')::INTEGER)
    WITH CHECK (customer_id = current_setting('app.current_customer_id')::INTEGER);
```

### Advanced RLS Policies
```sql
-- Multi-tenant application with company isolation
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;

CREATE POLICY company_isolation ON orders
    FOR ALL
    TO tenant_users
    USING (
        company_id = (
            SELECT company_id 
            FROM users 
            WHERE user_id = current_setting('app.current_user_id')::INTEGER
        )
    );

-- Role-based access with RLS
CREATE POLICY manager_access ON employees
    FOR ALL
    TO managers
    USING (
        department_id IN (
            SELECT department_id 
            FROM manager_departments 
            WHERE manager_id = current_setting('app.current_user_id')::INTEGER
        )
    );

-- Time-based access control
CREATE POLICY business_hours_access ON sensitive_data
    FOR SELECT
    TO business_users
    USING (
        EXTRACT(HOUR FROM CURRENT_TIME) BETWEEN 9 AND 17
        AND EXTRACT(DOW FROM CURRENT_DATE) BETWEEN 1 AND 5
    );

-- Hierarchical access control
CREATE POLICY hierarchical_access ON employee_data
    FOR SELECT
    TO hr_users
    USING (
        employee_level <= (
            SELECT max_viewable_level 
            FROM hr_permissions 
            WHERE hr_user_id = current_setting('app.current_user_id')::INTEGER
        )
    );
```

### RLS with Functions
```sql
-- Create security function
CREATE OR REPLACE FUNCTION user_has_access_to_customer(customer_id INTEGER)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 
        FROM user_customer_access uca
        WHERE uca.user_id = current_setting('app.current_user_id')::INTEGER
        AND uca.customer_id = $1
        AND uca.access_granted = true
        AND (uca.expires_at IS NULL OR uca.expires_at > CURRENT_TIMESTAMP)
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Use function in RLS policy
CREATE POLICY function_based_access ON customer_data
    FOR ALL
    TO app_users
    USING (user_has_access_to_customer(customer_id));

-- Bypass RLS for specific roles
ALTER TABLE customers FORCE ROW LEVEL SECURITY; -- Apply to table owners too
GRANT BYPASSRLS TO system_admin; -- Allow bypassing RLS
```

### RLS Monitoring and Debugging
```sql
-- Check RLS status
SELECT 
    schemaname,
    tablename,
    rowsecurity,
    forcerowsecurity
FROM pg_tables
WHERE schemaname = 'public';

-- View RLS policies
SELECT 
    schemaname,
    tablename,
    policyname,
    permissive,
    roles,
    cmd,
    qual,
    with_check
FROM pg_policies
WHERE schemaname = 'public';

-- Test RLS policies
SET ROLE app_user;
SET app.current_customer_id = '123';
EXPLAIN (COSTS OFF) SELECT * FROM customers; -- Shows RLS filter in plan
```

## Data Encryption

### Transparent Data Encryption (TDE)
```sql
-- PostgreSQL doesn't have built-in TDE, but you can use:
-- 1. File system encryption (LUKS, BitLocker)
-- 2. pgcrypto extension for column-level encryption

CREATE EXTENSION pgcrypto;

-- Column-level encryption
CREATE TABLE secure_customer_data (
    customer_id SERIAL PRIMARY KEY,
    name TEXT,
    encrypted_ssn BYTEA, -- Encrypted column
    encrypted_credit_card BYTEA
);

-- Insert encrypted data
INSERT INTO secure_customer_data (name, encrypted_ssn, encrypted_credit_card)
VALUES (
    'John Doe',
    pgp_sym_encrypt('123-45-6789', 'encryption_key'),
    pgp_sym_encrypt('4111-1111-1111-1111', 'encryption_key')
);

-- Query encrypted data
SELECT 
    customer_id,
    name,
    pgp_sym_decrypt(encrypted_ssn, 'encryption_key') as ssn,
    pgp_sym_decrypt(encrypted_credit_card, 'encryption_key') as credit_card
FROM secure_customer_data;
```

### Application-Level Encryption
```sql
-- Create encryption functions for application use
CREATE OR REPLACE FUNCTION encrypt_sensitive_data(data TEXT, key_id TEXT)
RETURNS BYTEA AS $$
DECLARE
    encryption_key TEXT;
BEGIN
    -- Retrieve key from secure key store (implement your key management)
    encryption_key := get_encryption_key(key_id);
    RETURN pgp_sym_encrypt(data, encryption_key);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE OR REPLACE FUNCTION decrypt_sensitive_data(encrypted_data BYTEA, key_id TEXT)
RETURNS TEXT AS $$
DECLARE
    encryption_key TEXT;
BEGIN
    encryption_key := get_encryption_key(key_id);
    RETURN pgp_sym_decrypt(encrypted_data, encryption_key);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Use encryption functions
CREATE TABLE encrypted_documents (
    doc_id SERIAL PRIMARY KEY,
    title TEXT,
    content BYTEA, -- Encrypted content
    key_id TEXT
);
```

## Security Auditing and Monitoring

### Audit Logging with pg_audit
```sql
-- Install pg_audit extension
CREATE EXTENSION pg_audit;

-- Configure audit settings in postgresql.conf:
-- shared_preload_libraries = 'pg_audit'
-- pg_audit.log = 'all'
-- pg_audit.log_catalog = off
-- pg_audit.log_client = on
-- pg_audit.log_level = log
-- pg_audit.log_parameter = on

-- Session-level audit settings
SET pg_audit.log = 'read,write,ddl';
SET pg_audit.log_relation = on;

-- Object-level auditing
SELECT pg_audit.log_object_access('SELECT', 'public.sensitive_table');

-- Review audit logs
-- Logs will appear in PostgreSQL log files with AUDIT prefix
```

### Custom Audit System
```sql
-- Create audit table
CREATE TABLE audit_log (
    audit_id BIGSERIAL PRIMARY KEY,
    table_name TEXT NOT NULL,
    operation TEXT NOT NULL,
    user_name TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    application_name TEXT
);

-- Audit trigger function
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO audit_log (
        table_name,
        operation,
        user_name,
        old_values,
        new_values,
        ip_address,
        application_name
    ) VALUES (
        TG_TABLE_NAME,
        TG_OP,
        session_user,
        CASE WHEN TG_OP = 'DELETE' OR TG_OP = 'UPDATE' THEN row_to_json(OLD) END,
        CASE WHEN TG_OP = 'INSERT' OR TG_OP = 'UPDATE' THEN row_to_json(NEW) END,
        inet_client_addr(),
        current_setting('application_name', true)
    );
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Apply audit trigger to sensitive tables
CREATE TRIGGER audit_customers
    AFTER INSERT OR UPDATE OR DELETE ON customers
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER audit_orders
    AFTER INSERT OR UPDATE OR DELETE ON orders
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
```

### Connection and Activity Monitoring
```sql
-- Monitor current connections
SELECT 
    pid,
    usename,
    application_name,
    client_addr,
    client_port,
    backend_start,
    state,
    query_start,
    query
FROM pg_stat_activity
WHERE state = 'active';

-- Monitor failed login attempts (requires log analysis)
-- Configure in postgresql.conf:
-- log_connections = on
-- log_disconnections = on
-- log_failed_connections = on

-- Track privilege escalations
CREATE TABLE privilege_changes (
    change_id SERIAL PRIMARY KEY,
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    changed_by TEXT,
    target_role TEXT,
    privilege_type TEXT,
    granted BOOLEAN
);

-- Function to log privilege changes
CREATE OR REPLACE FUNCTION log_privilege_change()
RETURNS EVENT_TRIGGER AS $$
BEGIN
    -- Log DDL commands that affect privileges
    INSERT INTO privilege_changes (changed_by, target_role, privilege_type, granted)
    SELECT 
        session_user,
        'extracted_from_command',
        'privilege_change',
        true;
END;
$$ LANGUAGE plpgsql;

-- Event trigger for DDL commands
CREATE EVENT TRIGGER log_ddl_privilege_changes
ON ddl_command_end
EXECUTE FUNCTION log_privilege_change();
```

## Security Best Practices

### Principle of Least Privilege
```sql
-- Create application-specific roles with minimal privileges
CREATE ROLE readonly_app WITH LOGIN PASSWORD 'secure_pass';
GRANT CONNECT ON DATABASE myapp TO readonly_app;
GRANT USAGE ON SCHEMA public TO readonly_app;
GRANT SELECT ON specific_tables TO readonly_app;

-- Separate roles for different functions
CREATE ROLE order_processor WITH LOGIN PASSWORD 'processor_pass';
GRANT USAGE ON SCHEMA orders TO order_processor;
GRANT SELECT, UPDATE ON orders TO order_processor;
GRANT UPDATE (status, processed_at) ON orders TO order_processor;

-- Avoid using superuser for applications
-- Create specific administrative roles instead
CREATE ROLE db_maintainer WITH 
    LOGIN 
    PASSWORD 'maintainer_pass'
    CREATEDB
    CREATEROLE;
```

### Secure Configuration Examples
```sql
-- Remove dangerous default privileges
REVOKE ALL ON SCHEMA public FROM public;
REVOKE CREATE ON SCHEMA public FROM public;

-- Grant only necessary privileges
GRANT USAGE ON SCHEMA public TO authenticated_users;

-- Use security-definer functions for privileged operations
CREATE OR REPLACE FUNCTION secure_user_creation(
    p_username TEXT,
    p_email TEXT,
    p_password TEXT
)
RETURNS INTEGER
SECURITY DEFINER -- Runs with definer's privileges
SET search_path = public, pg_temp
AS $$
DECLARE
    new_user_id INTEGER;
BEGIN
    -- Validate input
    IF LENGTH(p_password) < 8 THEN
        RAISE EXCEPTION 'Password must be at least 8 characters';
    END IF;
    
    -- Create user with encrypted password
    INSERT INTO users (username, email, password_hash)
    VALUES (p_username, p_email, crypt(p_password, gen_salt('bf')))
    RETURNING user_id INTO new_user_id;
    
    RETURN new_user_id;
END;
$$ LANGUAGE plpgsql;

-- Grant execution to specific roles
GRANT EXECUTE ON FUNCTION secure_user_creation(TEXT, TEXT, TEXT) TO app_registrar;
```

### Security Checklist
```sql
-- 1. Regular security audit queries
-- Check for users with excessive privileges
SELECT 
    rolname,
    rolsuper,
    rolcreaterole,
    rolcreatedb,
    rolcanlogin
FROM pg_roles
WHERE rolsuper = true OR rolcreaterole = true;

-- 2. Check for public schema access
SELECT 
    grantee,
    privilege_type
FROM information_schema.schema_privileges
WHERE schema_name = 'public' AND grantee = 'public';

-- 3. Review RLS policies
SELECT 
    schemaname,
    tablename,
    policyname,
    permissive,
    roles
FROM pg_policies;

-- 4. Check for unencrypted connections
SELECT 
    usename,
    application_name,
    client_addr,
    ssl
FROM pg_stat_activity
WHERE ssl = false AND client_addr IS NOT NULL;

-- 5. Monitor for unusual activity
SELECT 
    usename,
    COUNT(*) as connection_count,
    array_agg(DISTINCT client_addr) as source_ips
FROM pg_stat_activity
GROUP BY usename
HAVING COUNT(*) > 10; -- Adjust threshold as needed
```
