### 🟢 Beginner Level – Topic 5: **PostgreSQL Tools (CLI & GUI)**

## Command Line Tools

### psql - Interactive Terminal
The primary command-line interface for PostgreSQL.

#### Basic Connection
```bash
# Connect to local database
psql -d database_name

# Connect with specific user
psql -U username -d database_name

# Connect to remote server
psql -h hostname -p port -U username -d database_name

# Connect with password prompt
psql -U username -d database_name -W
```

#### Common psql Commands
```sql
-- List databases
\l

-- Connect to database
\c database_name

-- List tables
\dt

-- Describe table structure
\d table_name

-- List all relations (tables, views, sequences)
\d

-- List users/roles
\du

-- List schemas
\dn

-- Show current connection info
\conninfo

-- Execute SQL file
\i filename.sql

-- Turn on/off timing
\timing

-- Quit psql
\q

-- Get help
\?
```

#### psql Configuration
```sql
-- Set prompt
\set PROMPT1 '%n@%m:%>%x %/# '

-- Set pager off for small results
\pset pager off

-- Format output
\x  -- Expanded display toggle
\a  -- Aligned output toggle
\t  -- Tuples only toggle
```

### pg_dump - Database Backup
```bash
# Backup entire database
pg_dump database_name > backup.sql

# Backup with compression
pg_dump -Fc database_name > backup.dump

# Backup specific tables
pg_dump -t table_name database_name > table_backup.sql

# Backup schema only (no data)
pg_dump -s database_name > schema_only.sql

# Backup data only (no schema)
pg_dump -a database_name > data_only.sql

# Remote backup
pg_dump -h hostname -U username database_name > backup.sql
```

### pg_restore - Database Restore
```bash
# Restore from compressed dump
pg_restore -d database_name backup.dump

# Restore with verbose output
pg_restore -v -d database_name backup.dump

# Restore specific tables
pg_restore -t table_name -d database_name backup.dump

# Restore to different database
pg_restore -d new_database backup.dump
```

### createdb/dropdb - Database Management
```bash
# Create database
createdb new_database

# Create database with owner
createdb -O username new_database

# Drop database
dropdb database_name

# Create database with template
createdb -T template0 new_database
```

### pg_ctl - Server Control
```bash
# Start PostgreSQL server
pg_ctl start -D /path/to/data/directory

# Stop PostgreSQL server
pg_ctl stop -D /path/to/data/directory

# Restart PostgreSQL server
pg_ctl restart -D /path/to/data/directory

# Reload configuration
pg_ctl reload -D /path/to/data/directory

# Check server status
pg_ctl status -D /path/to/data/directory
```

## Graphical Tools

### pgAdmin
Web-based administration and development platform.

#### Features
- Database object browser
- Query editor with syntax highlighting
- Visual query builder
- Backup/restore tools
- User management
- Performance monitoring
- Database migration tools

#### Installation
```bash
# Ubuntu/Debian
sudo apt install pgadmin4

# macOS with Homebrew
brew install --cask pgadmin4

# Windows - Download from pgadmin.org
```

### DBeaver
Universal database tool with PostgreSQL support.

#### Features
- Multi-platform (Windows, macOS, Linux)
- SQL editor with autocomplete
- Data viewer and editor
- ER diagrams
- Import/export tools
- Connection management

### DataGrip
JetBrains IDE for databases.

#### Features
- Intelligent query console
- Efficient schema navigation
- Version control integration
- Code completion and refactoring
- Database introspection

## Development Tools

### Visual Studio Code Extensions
```
PostgreSQL by Chris Kolkman
SQLTools by Matheus Teixeira
```

#### Setup
1. Install PostgreSQL extension
2. Configure connection settings
3. Use integrated query runner

### Database Modeling Tools

#### dbdiagram.io
- Online ER diagram tool
- DBML (Database Markup Language)
- Export to SQL

#### pgModeler
- PostgreSQL-specific modeling tool
- Visual database design
- Code generation

### Migration Tools

#### Flyway
Database migration tool.

```bash
# Install Flyway
wget -qO- https://repo1.maven.org/maven2/org/flywaydb/flyway-commandline/8.5.13/flyway-commandline-8.5.13-linux-x64.tar.gz | tar xvz

# Configure flyway.conf
flyway.url=jdbc:postgresql://localhost:5432/mydb
flyway.user=myuser
flyway.password=mypassword

# Run migrations
flyway migrate
```

#### Liquibase
Database change management.

```xml
<!-- changelog.xml -->
<?xml version="1.0" encoding="UTF-8"?>
<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog">
    <changeSet id="1" author="developer">
        <createTable tableName="users">
            <column name="id" type="SERIAL">
                <constraints primaryKey="true"/>
            </column>
            <column name="username" type="VARCHAR(50)">
                <constraints nullable="false"/>
            </column>
        </createTable>
    </changeSet>
</databaseChangeLog>
```

## Monitoring and Performance Tools

### pg_stat_statements
Query performance tracking.

```sql
-- Enable extension
CREATE EXTENSION pg_stat_statements;

-- View query statistics
SELECT query, calls, total_time, mean_time
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;
```

### EXPLAIN and EXPLAIN ANALYZE
Query execution plan analysis.

```sql
-- Show execution plan
EXPLAIN SELECT * FROM users WHERE age > 25;

-- Show actual execution statistics
EXPLAIN ANALYZE SELECT * FROM users WHERE age > 25;

-- More detailed analysis
EXPLAIN (ANALYZE, BUFFERS, VERBOSE) 
SELECT * FROM users WHERE age > 25;
```

### pg_activity
Real-time activity monitoring.

```bash
# Install pg_activity
pip install pg_activity

# Run monitoring
pg_activity -h localhost -U username -d database_name
```

## Data Import/Export Tools

### COPY Command
```sql
-- Export to CSV
COPY users TO '/path/to/users.csv' DELIMITER ',' CSV HEADER;

-- Import from CSV
COPY users FROM '/path/to/users.csv' DELIMITER ',' CSV HEADER;

-- Export query results
COPY (SELECT * FROM users WHERE active = true) 
TO '/path/to/active_users.csv' CSV HEADER;
```

### \copy in psql
```sql
-- Export (client-side)
\copy users TO 'users.csv' CSV HEADER

-- Import (client-side)
\copy users FROM 'users.csv' CSV HEADER
```

## Configuration Management

### postgresql.conf
Main configuration file.

```conf
# Connection settings
listen_addresses = '*'
port = 5432
max_connections = 100

# Memory settings
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB

# WAL settings
wal_level = replica
max_wal_senders = 3
```

### pg_hba.conf
Client authentication configuration.

```conf
# Database administrative login by Unix domain socket
local   all             postgres                                peer

# "local" is for Unix domain socket connections only
local   all             all                                     peer

# IPv4 local connections:
host    all             all             127.0.0.1/32            md5

# IPv6 local connections:
host    all             all             ::1/128                 md5
```

## Useful Scripts and Aliases

### Bash Aliases
```bash
# Add to ~/.bashrc or ~/.zshrc
alias pgstart='sudo systemctl start postgresql'
alias pgstop='sudo systemctl stop postgresql'
alias pgrestart='sudo systemctl restart postgresql'
alias pgstatus='sudo systemctl status postgresql'

# Quick psql connection
alias psqldev='psql -h localhost -U developer -d development'
```

### Common SQL Scripts
```sql
-- Database size
SELECT pg_size_pretty(pg_database_size(current_database()));

-- Table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Active connections
SELECT 
    datname,
    usename,
    application_name,
    client_addr,
    state
FROM pg_stat_activity
WHERE state = 'active';
```

## Docker Tools

### PostgreSQL Docker Container
```bash
# Run PostgreSQL in Docker
docker run --name postgres-db \
  -e POSTGRES_PASSWORD=mypassword \
  -e POSTGRES_DB=mydatabase \
  -p 5432:5432 \
  -d postgres:14

# Connect to container
docker exec -it postgres-db psql -U postgres -d mydatabase

# Backup from container
docker exec postgres-db pg_dump -U postgres mydatabase > backup.sql

# Restore to container
docker exec -i postgres-db psql -U postgres mydatabase < backup.sql
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'
services:
  postgres:
    image: postgres:14
    environment:
      POSTGRES_PASSWORD: mypassword
      POSTGRES_DB: mydatabase
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql

volumes:
  postgres_data:
```

## Best Practices

### Tool Selection
- Use **psql** for quick queries and administration
- Use **pgAdmin** for complex database management
- Use **pg_dump/pg_restore** for regular backups
- Use migration tools for schema changes
- Use monitoring tools for performance optimization

### Security
- Keep tools updated
- Use encrypted connections
- Limit access permissions
- Regular security audits
- Backup encryption

### Performance
- Monitor query performance regularly
- Use appropriate indexes
- Regular maintenance tasks
- Connection pooling for applications
- Hardware monitoring

---

## ✅ 1. **psql – PostgreSQL CLI Tool**

`psql` is the default PostgreSQL command-line client. It's powerful and flexible.

---

### 🔹 Connecting to a Database

```bash
psql -U username -d databasename
```

If using the `postgres` user:

```bash
sudo -u postgres psql
```

**Switch database inside psql:**

```sql
\c another_database
```

---

### 🔹 Common `psql` Meta-commands

| Command        | Description                        |
| -------------- | ---------------------------------- |
| `\q`           | Quit                               |
| `\l`           | List databases                     |
| `\c dbname`    | Connect to a database              |
| `\dt`          | List tables in current schema      |
| `\d tablename` | Describe table schema              |
| `\du`          | List users/roles                   |
| `\x`           | Toggle expanded output (vertical)  |
| `\h command`   | Show SQL syntax (e.g. `\h SELECT`) |

---

### 🔹 Export & Import from CLI

```bash
psql -U user -d dbname -f file.sql     # Run SQL script
pg_dump -U user -d dbname > backup.sql # Export database
psql -U user -d dbname < backup.sql    # Import backup
```

---

## ✅ 2. **pgAdmin – Official GUI Tool**

* Web-based GUI for PostgreSQL
* Excellent for **visualizing schemas**, writing queries, and **managing roles**

### Features:

* SQL Editor with autocomplete
* Graphical ERD view
* Backup/restore interface
* User and permission management

🔹 To run:

```bash
sudo apt install pgadmin4
```

Or download from: [https://www.pgadmin.org/](https://www.pgadmin.org/)

---

## ✅ 3. **DBeaver – Universal DB GUI**

* Open-source GUI client for many databases
* Popular with developers for its speed and features

### Key Features:

* Query editor
* ERD diagrams
* Data export/import
* Schema browsing

🔗 Get it: [https://dbeaver.io/](https://dbeaver.io/)

---

## ✅ 4. **TablePlus / DataGrip (Optional)**

| Tool      | Platform       | Strength                         |
| --------- | -------------- | -------------------------------- |
| TablePlus | macOS, Windows | Lightweight, fast, modern UI     |
| DataGrip  | All platforms  | JetBrains IDE for DB power users |

---

## ✅ Summary

* `psql` is the go-to CLI for fast control and scripting.
* Use `pgAdmin` or `DBeaver` for GUI-based database inspection, especially during design or debugging.
* Master both CLI and GUI tools to be a productive PostgreSQL user.
