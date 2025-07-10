### 🔴 Expert Level – Topic 16: **Replication & High Availability**

## ✅ 1. **What is Replication?**

Replication = Copying data from one PostgreSQL instance (**primary**) to another (**standby**).

---

### 🔹 Key Types:

| Type                      | Description                                |
| ------------------------- | ------------------------------------------ |
| **Streaming Replication** | Real-time binary-level replication         |
| **Logical Replication**   | Table-level, row-based (can filter tables) |
| **Warm/Hot Standby**      | Replica available for read-only queries    |
| **Synchronous vs Async**  | Sync = safe, Async = fast                  |

---

## ✅ 2. **Streaming Replication (Binary-Level)**

### 🔹 Steps:

1. **Enable WAL Archiving** in `postgresql.conf` (on primary):

```conf
wal_level = replica
max_wal_senders = 10
archive_mode = on
archive_command = 'cp %p /var/lib/postgresql/wal_archive/%f'
```

2. **Allow Replication in `pg_hba.conf`:**

```conf
host replication all standby_ip/32 md5
```

3. **Base Backup to Standby:**

```bash
pg_basebackup -h primary_ip -D /var/lib/postgresql/data -U replicator -P -R
```

4. **Start Standby PostgreSQL:**
   Standby will automatically follow the primary.

---

### 🔹 Use Case:

* Real-time failover
* Load balancing for read-heavy systems (read from standby)

---

## ✅ 3. **Logical Replication (Table-Level)**

PostgreSQL 10+

### 🔹 On the **Publisher** (Primary):

```sql
CREATE PUBLICATION my_pub FOR TABLE users, orders;
```

### 🔹 On the **Subscriber** (Replica):

```sql
CREATE SUBSCRIPTION my_sub
  CONNECTION 'host=primary port=5432 user=replicator dbname=mydb'
  PUBLICATION my_pub;
```

✅ Fine-grained: replicate only selected tables
❌ No schema/DDL replication

---

## ✅ 4. **Synchronous vs Asynchronous Replication**

| Mode             | Description                                    |
| ---------------- | ---------------------------------------------- |
| **Asynchronous** | Default, primary does **not wait** for standby |
| **Synchronous**  | Primary **waits** for standby to confirm write |

```conf
synchronous_standby_names = 'standby1'
```

Good for **data safety**, bad for **write performance** if standby is slow.

---

## ✅ 5. **Failover and High Availability**

Use tools to detect failure and promote standby:

### 🔹 Manual Failover:

```sql
SELECT pg_promote();  -- Run on standby
```

### 🔹 HA Tools:

| Tool        | Description                        |
| ----------- | ---------------------------------- |
| `repmgr`    | Replication manager for PostgreSQL |
| `Patroni`   | HA via etcd/Consul + REST API      |
| `Pgpool-II` | Load balancing + failover manager  |
| `Pacemaker` | Cluster-based HA                   |

---

## ✅ 6. **Monitoring Replication**

### 🔹 Check Replication Status (Primary):

```sql
SELECT * FROM pg_stat_replication;
```

### 🔹 On Standby:

```sql
SELECT pg_last_wal_receive_lsn(), pg_last_wal_replay_lsn();
```

Use tools like:

* `pg_stat_activity`
* `pg_replication_slots`

---

## ✅ 7. **Replication Slots**

Keep WAL files until the replica consumes them.

```sql
SELECT * FROM pg_create_physical_replication_slot('standby_slot');
```

---

## ✅ Summary

| Feature               | Use Case                           |
| --------------------- | ---------------------------------- |
| Streaming Replication | Real-time binary syncing           |
| Logical Replication   | Selective table-level replication  |
| Synchronous Mode      | Strong consistency                 |
| Replication Slots     | WAL retention for lagging replicas |
| HA Tools              | Auto-failover and monitoring       |

PostgreSQL provides **enterprise-grade availability** out of the box — but tools like `Patroni` or `Pgpool-II` make production readiness even better.

---

# Replication and High Availability

## Introduction to PostgreSQL Replication

PostgreSQL offers several replication methods to ensure high availability, load distribution, and disaster recovery. Understanding these options is crucial for production deployments.

### Types of Replication
1. **Streaming Replication** - Real-time binary replication
2. **Logical Replication** - Table-level, cross-version replication
3. **Point-in-Time Recovery (PITR)** - Backup-based recovery
4. **Synchronous vs Asynchronous** - Consistency vs performance trade-offs

## Streaming Replication

### Setting Up Primary Server
```bash
# postgresql.conf configuration for primary
wal_level = replica
max_wal_senders = 5
max_replication_slots = 5
wal_keep_segments = 32  # Or wal_keep_size for PostgreSQL 13+
archive_mode = on
archive_command = 'cp %p /path/to/archive/%f'
hot_standby = on
```

```sql
-- Create replication user
CREATE USER replicator WITH REPLICATION ENCRYPTED PASSWORD 'replica_password';

-- Grant necessary permissions
GRANT CONNECT ON DATABASE postgres TO replicator;
```

```bash
# pg_hba.conf - Allow replication connections
host replication replicator 192.168.1.0/24 md5
hostssl replication replicator 0.0.0.0/0 md5
```

### Setting Up Standby Server
```bash
# Create base backup from primary
pg_basebackup -h primary_server -D /var/lib/postgresql/data \
  -U replicator -W -v -P -x

# Create recovery.conf (PostgreSQL < 12) or postgresql.conf settings
# For PostgreSQL 12+, add to postgresql.conf:
primary_conninfo = 'host=primary_server port=5432 user=replicator password=replica_password'
restore_command = 'cp /path/to/archive/%f %p'

# Create standby.signal file (PostgreSQL 12+)
touch /var/lib/postgresql/data/standby.signal
```

### Monitoring Replication
```sql
-- On primary: Check replication status
SELECT 
    client_addr,
    client_hostname,
    client_port,
    state,
    sent_lsn,
    write_lsn,
    flush_lsn,
    replay_lsn,
    write_lag,
    flush_lag,
    replay_lag,
    sync_state
FROM pg_stat_replication;

-- On standby: Check replication lag
SELECT 
    now() - pg_last_xact_replay_timestamp() AS replication_delay,
    pg_is_in_recovery() AS is_standby;

-- Check WAL receive status on standby
SELECT 
    pid,
    status,
    receive_start_lsn,
    receive_start_tli,
    received_lsn,
    received_tli,
    last_msg_send_time,
    last_msg_receipt_time,
    latest_end_lsn,
    latest_end_time,
    slot_name,
    sender_host,
    sender_port,
    conninfo
FROM pg_stat_wal_receiver;
```

## Synchronous Replication

### Configuration for Synchronous Replication
```bash
# postgresql.conf on primary
synchronous_standby_names = 'standby1,standby2'  # ANY 1 (standby1,standby2)
synchronous_commit = on

# For quorum-based synchronous replication
synchronous_standby_names = 'ANY 2 (standby1,standby2,standby3)'
```

### Application Names for Standby Servers
```bash
# On standby server in recovery.conf or postgresql.conf
primary_conninfo = 'host=primary port=5432 user=replicator application_name=standby1'
```

### Monitoring Synchronous Replication
```sql
-- Check synchronous replication status
SELECT 
    application_name,
    client_addr,
    state,
    sync_state,
    sync_priority,
    replay_lag
FROM pg_stat_replication
WHERE sync_state IN ('sync', 'potential');

-- Test synchronous behavior
BEGIN;
INSERT INTO test_table VALUES (1, 'sync test');
-- This will wait for standby confirmation before commit
COMMIT;
```

## Logical Replication

### Setting Up Logical Replication
```sql
-- On source database (publisher)
-- Enable logical replication
ALTER SYSTEM SET wal_level = logical;
-- Restart PostgreSQL

-- Create publication
CREATE PUBLICATION my_publication FOR ALL TABLES;
-- Or for specific tables
CREATE PUBLICATION orders_publication FOR TABLE orders, order_items;

-- Check publications
SELECT pubname, puballtables FROM pg_publication;
```

```sql
-- On target database (subscriber)
-- Create subscription
CREATE SUBSCRIPTION my_subscription 
CONNECTION 'host=source_host dbname=source_db user=repl_user password=password'
PUBLICATION my_publication;

-- Check subscription status
SELECT 
    subname,
    pid,
    received_lsn,
    latest_end_lsn,
    latest_end_time
FROM pg_stat_subscription;
```

### Managing Logical Replication
```sql
-- Add table to publication
ALTER PUBLICATION my_publication ADD TABLE new_table;

-- Remove table from publication
ALTER PUBLICATION my_publication DROP TABLE old_table;

-- Refresh subscription after schema changes
ALTER SUBSCRIPTION my_subscription REFRESH PUBLICATION;

-- Enable/disable subscription
ALTER SUBSCRIPTION my_subscription DISABLE;
ALTER SUBSCRIPTION my_subscription ENABLE;

-- Drop subscription
DROP SUBSCRIPTION my_subscription;
```

### Logical Replication Conflicts
```sql
-- Handle conflicts in logical replication
-- Check for conflicts
SELECT 
    slot_name,
    plugin,
    slot_type,
    database,
    active,
    active_pid,
    confirmed_flush_lsn
FROM pg_replication_slots;

-- Skip conflicting transaction (use with caution)
-- ALTER SUBSCRIPTION my_subscription SKIP (lsn = 'LSN_VALUE');

-- Monitor replication slot lag
SELECT 
    slot_name,
    pg_size_pretty(
        pg_wal_lsn_diff(pg_current_wal_lsn(), confirmed_flush_lsn)
    ) as replication_lag
FROM pg_replication_slots
WHERE slot_type = 'logical';
```

## Replication Slots

### Physical Replication Slots
```sql
-- Create replication slot on primary
SELECT pg_create_physical_replication_slot('standby_slot');

-- Use slot in standby configuration
-- primary_slot_name = 'standby_slot'

-- Monitor replication slots
SELECT 
    slot_name,
    plugin,
    slot_type,
    database,
    active,
    active_pid,
    confirmed_flush_lsn,
    restart_lsn,
    wal_status
FROM pg_replication_slots;

-- Drop replication slot
SELECT pg_drop_replication_slot('standby_slot');
```

### Logical Replication Slots
```sql
-- Create logical replication slot
SELECT pg_create_logical_replication_slot('logical_slot', 'pgoutput');

-- Read from logical slot
SELECT * FROM pg_logical_slot_get_changes('logical_slot', NULL, NULL);

-- Peek at changes without consuming them
SELECT * FROM pg_logical_slot_peek_changes('logical_slot', NULL, NULL);

-- Advance slot without processing
SELECT pg_replication_slot_advance('logical_slot', 'LSN_VALUE');
```

## Failover and Switchover

### Planned Switchover
```sql
-- 1. Stop writes to primary
-- 2. Wait for standby to catch up
SELECT pg_current_wal_lsn(); -- On primary
SELECT pg_last_wal_replay_lsn(); -- On standby - should match

-- 3. Promote standby
-- Create trigger file or use pg_promote()
SELECT pg_promote();

-- 4. Reconfigure old primary as new standby
-- Update configuration and restart
```

### Automatic Failover Setup
```bash
# Using pg_auto_failover
# Install pg_auto_failover extension

# Create monitor node
pg_autoctl create monitor --hostname monitor.example.com --pgdata /var/lib/postgresql/monitor

# Create primary node
pg_autoctl create postgres --hostname primary.example.com --pgdata /var/lib/postgresql/primary --monitor postgres://autoctl_node@monitor.example.com/pg_auto_failover

# Create standby node
pg_autoctl create postgres --hostname standby.example.com --pgdata /var/lib/postgresql/standby --monitor postgres://autoctl_node@monitor.example.com/pg_auto_failover

# Run nodes
pg_autoctl run --pgdata /var/lib/postgresql/primary &
pg_autoctl run --pgdata /var/lib/postgresql/standby &
```

### Manual Failover Process
```sql
-- 1. Check replication status before failover
SELECT 
    client_addr,
    state,
    sync_state,
    replay_lag
FROM pg_stat_replication;

-- 2. On standby, check last received WAL
SELECT pg_last_wal_replay_lsn();

-- 3. Promote standby to primary
SELECT pg_promote();

-- 4. Update application connection strings
-- Point applications to new primary

-- 5. Set up old primary as new standby (if recoverable)
-- Requires pg_rewind if timelines diverged
pg_rewind --target-pgdata=/var/lib/postgresql/data --source-server='host=new_primary user=postgres'
```

## Connection Pooling and Load Balancing

### pgBouncer Configuration
```ini
# pgbouncer.ini
[databases]
mydb = host=primary_server port=5432 dbname=mydb
mydb_readonly = host=standby_server port=5432 dbname=mydb

[pgbouncer]
listen_port = 6432
listen_addr = *
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25
reserve_pool_size = 5
server_round_robin = 1
```

### HAProxy for Load Balancing
```
# haproxy.cfg
global
    maxconn 4096

defaults
    mode tcp
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms

frontend postgres_frontend
    bind *:5432
    default_backend postgres_backend

backend postgres_backend
    balance roundrobin
    option tcp-check
    tcp-check expect string "accepts connections"
    server primary primary_server:5432 check
    server standby1 standby1_server:5432 check backup
    server standby2 standby2_server:5432 check backup
```

### Read-Write Splitting
```sql
-- Application-level read-write splitting
-- Write operations go to primary
INSERT INTO orders (customer_id, total) VALUES (1, 100.00);
UPDATE orders SET status = 'shipped' WHERE order_id = 123;

-- Read operations can go to standby
SELECT * FROM orders WHERE customer_id = 1;
SELECT COUNT(*) FROM products WHERE category = 'electronics';

-- Use connection parameters for routing
-- postgresql://user:pass@primary:5432/db?target_session_attrs=read-write
-- postgresql://user:pass@standby:5432/db?target_session_attrs=read-only
```

## Backup and Recovery in Replication Environment

### Point-in-Time Recovery (PITR)
```bash
# Configure continuous archiving
archive_mode = on
archive_command = 'cp %p /backup/archive/%f'

# Take base backup
pg_basebackup -h primary -D /backup/base -U backup_user -v -P -x

# Restore to specific point in time
# 1. Restore base backup
cp -R /backup/base/* /var/lib/postgresql/data/

# 2. Configure recovery
# recovery.conf (PostgreSQL < 12) or postgresql.conf
restore_command = 'cp /backup/archive/%f %p'
recovery_target_time = '2023-12-01 14:30:00'

# 3. Start PostgreSQL
systemctl start postgresql
```

### Backup Strategies with Replication
```bash
# Backup from standby to reduce primary load
pg_dump -h standby_server -U backup_user mydb > backup.sql

# Backup with pg_basebackup from standby
pg_basebackup -h standby_server -D /backup/standby_backup -U replicator -v -P

# Verify backup consistency
pg_controldata /backup/standby_backup
```

## Monitoring and Alerting

### Replication Health Checks
```sql
-- Create monitoring function
CREATE OR REPLACE FUNCTION check_replication_health()
RETURNS TABLE(
    server_type TEXT,
    is_primary BOOLEAN,
    replication_lag INTERVAL,
    wal_lag_bytes BIGINT,
    status TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        CASE 
            WHEN pg_is_in_recovery() THEN 'Standby'
            ELSE 'Primary'
        END as server_type,
        NOT pg_is_in_recovery() as is_primary,
        CASE 
            WHEN pg_is_in_recovery() THEN now() - pg_last_xact_replay_timestamp()
            ELSE NULL
        END as replication_lag,
        CASE 
            WHEN NOT pg_is_in_recovery() THEN 
                pg_wal_lsn_diff(pg_current_wal_lsn(), 
                    COALESCE((SELECT flush_lsn FROM pg_stat_replication LIMIT 1), pg_current_wal_lsn()))
            ELSE NULL
        END as wal_lag_bytes,
        CASE 
            WHEN pg_is_in_recovery() AND now() - pg_last_xact_replay_timestamp() > INTERVAL '1 minute' 
                THEN 'ALERT: High replication lag'
            WHEN NOT pg_is_in_recovery() AND NOT EXISTS (SELECT 1 FROM pg_stat_replication)
                THEN 'WARNING: No standby servers'
            ELSE 'OK'
        END as status;
END;
$$ LANGUAGE plpgsql;

-- Use monitoring function
SELECT * FROM check_replication_health();
```

### Automated Monitoring Script
```bash
#!/bin/bash
# replication_monitor.sh

# Check if server is primary or standby
IS_PRIMARY=$(psql -t -c "SELECT NOT pg_is_in_recovery();" | tr -d ' ')

if [ "$IS_PRIMARY" = "t" ]; then
    # Primary server checks
    echo "Checking primary server..."
    
    # Check connected standbys
    STANDBY_COUNT=$(psql -t -c "SELECT COUNT(*) FROM pg_stat_replication;" | tr -d ' ')
    echo "Connected standbys: $STANDBY_COUNT"
    
    # Check replication lag
    psql -c "SELECT client_addr, state, replay_lag FROM pg_stat_replication;"
    
    # Alert if no standbys connected
    if [ "$STANDBY_COUNT" -eq 0 ]; then
        echo "ALERT: No standby servers connected!" | mail -s "PostgreSQL Replication Alert" admin@company.com
    fi
else
    # Standby server checks
    echo "Checking standby server..."
    
    # Check replication lag
    LAG=$(psql -t -c "SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp()));" | tr -d ' ')
    echo "Replication lag: ${LAG} seconds"
    
    # Alert if lag is too high
    if (( $(echo "$LAG > 300" | bc -l) )); then
        echo "ALERT: Replication lag is ${LAG} seconds!" | mail -s "PostgreSQL Replication Lag Alert" admin@company.com
    fi
fi
```

## Best Practices and Troubleshooting

### Performance Optimization
```sql
-- Optimize replication performance
-- On primary
wal_compression = on
wal_writer_delay = 10ms
wal_writer_flush_after = 1MB

-- On standby
hot_standby_feedback = on  -- Reduces query conflicts
max_standby_streaming_delay = 30s
max_standby_archive_delay = 30s
```

### Common Issues and Solutions
```sql
-- Issue: Replication slot growing too large
-- Solution: Monitor and clean up inactive slots
SELECT 
    slot_name,
    active,
    pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn)) as lag_size
FROM pg_replication_slots;

-- Drop inactive slots
SELECT pg_drop_replication_slot('inactive_slot');

-- Issue: Standby queries being cancelled
-- Solution: Adjust conflict resolution settings
max_standby_streaming_delay = -1  -- Never cancel queries
hot_standby_feedback = on         -- Prevent vacuum conflicts

-- Issue: Synchronous replication blocking
-- Check synchronous standby availability
SELECT application_name, sync_state FROM pg_stat_replication;

-- Temporarily switch to asynchronous if needed
ALTER SYSTEM SET synchronous_standby_names = '';
SELECT pg_reload_conf();
```

### Disaster Recovery Procedures
```sql
-- Document recovery procedures
-- 1. Assess primary server status
-- 2. Check data integrity on standbys
-- 3. Promote most up-to-date standby
-- 4. Update application configurations
-- 5. Set up new standby servers
-- 6. Test application functionality

-- Recovery validation queries
SELECT pg_is_in_recovery();  -- Should be false on new primary
SELECT pg_last_wal_replay_lsn();  -- Last applied WAL
SELECT COUNT(*) FROM critical_table;  -- Data verification
```
