# 20. PostgreSQL in Production

## Overview
Deploying and maintaining PostgreSQL in production requires careful planning, robust configuration, and ongoing operational excellence. This guide covers production deployment strategies, configuration optimization, backup strategies, disaster recovery, and operational best practices.

## Table of Contents
- [Production Architecture](#production-architecture)
- [Configuration Optimization](#configuration-optimization)
- [Backup and Recovery](#backup-and-recovery)
- [High Availability](#high-availability)
- [Security Hardening](#security-hardening)
- [Performance Tuning](#performance-tuning)
- [Operational Procedures](#operational-procedures)
- [Troubleshooting](#troubleshooting)

## Production Architecture

### Typical Production Setup

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Application   │    │   Application   │
│    (HAProxy)    │────┤     Server 1    │    │     Server 2    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐             │
         └──────────────┤  Connection     ├─────────────┘
                        │  Pool (PgBouncer)│
                        └─────────────────┘
                                 │
                        ┌─────────────────┐
                        │   PostgreSQL    │
                        │   Primary       │
                        │   (Master)      │
                        └─────────────────┘
                                 │
                   ┌─────────────┼─────────────┐
          ┌─────────────────┐           ┌─────────────────┐
          │   PostgreSQL    │           │   PostgreSQL    │
          │   Standby 1     │           │   Standby 2     │
          │   (Sync)        │           │   (Async)       │
          └─────────────────┘           └─────────────────┘
```

### Infrastructure Components

1. **Primary Database Server**
   - Master PostgreSQL instance
   - Handles all write operations
   - Source for replication

2. **Standby Servers**
   - Read replicas for load distribution
   - Failover targets for high availability
   - Backup sources

3. **Connection Pooling**
   - PgBouncer for connection management
   - Reduces connection overhead
   - Connection routing and load balancing

4. **Load Balancing**
   - HAProxy or similar for traffic distribution
   - Health checks and failover
   - SSL termination

## Configuration Optimization

### postgresql.conf for Production

```ini
# Memory Configuration
shared_buffers = 4GB                    # 25% of total RAM
effective_cache_size = 12GB            # 75% of total RAM
work_mem = 64MB                        # Per-query memory
maintenance_work_mem = 1GB             # Maintenance operations
effective_io_concurrency = 200         # SSD configuration

# Write-Ahead Logging (WAL)
wal_level = replica                    # For replication
max_wal_senders = 10                   # Number of replication connections
wal_keep_segments = 64                 # Keep WAL segments for replication
archive_mode = on                      # Enable WAL archiving
archive_command = 'cp %p /backup/wal/%f'  # Archive command

# Checkpoints
checkpoint_completion_target = 0.9     # Spread checkpoint I/O
checkpoint_timeout = 15min             # Maximum time between checkpoints
max_wal_size = 4GB                     # Maximum WAL size
min_wal_size = 512MB                   # Minimum WAL size

# Connection Settings
max_connections = 200                  # Maximum concurrent connections
superuser_reserved_connections = 3     # Reserved for superusers

# Query Planner
random_page_cost = 1.1                 # For SSD storage
cpu_tuple_cost = 0.01                  # CPU cost per tuple
cpu_index_tuple_cost = 0.005           # CPU cost per index tuple
cpu_operator_cost = 0.0025             # CPU cost per operator

# Parallel Processing
max_parallel_workers_per_gather = 4    # Parallel workers per query
max_parallel_workers = 8               # Total parallel workers
max_parallel_maintenance_workers = 4   # Parallel maintenance workers

# Logging
log_min_duration_statement = 1000      # Log slow queries (1 second)
log_checkpoints = on                   # Log checkpoint activity
log_connections = on                   # Log connections
log_disconnections = on                # Log disconnections
log_lock_waits = on                    # Log lock waits
log_temp_files = 0                     # Log temp files
log_autovacuum_min_duration = 0        # Log autovacuum activity

# Autovacuum
autovacuum = on                        # Enable autovacuum
autovacuum_max_workers = 3             # Number of autovacuum workers
autovacuum_naptime = 15s               # Time between autovacuum runs
autovacuum_vacuum_threshold = 50       # Minimum tuples for vacuum
autovacuum_analyze_threshold = 50      # Minimum tuples for analyze
autovacuum_vacuum_scale_factor = 0.1   # Fraction of table for vacuum
autovacuum_analyze_scale_factor = 0.05 # Fraction of table for analyze

# Statistics
track_activities = on                  # Track query activity
track_counts = on                      # Track table statistics
track_io_timing = on                   # Track I/O timing
track_functions = pl                   # Track function calls
```

### OS-Level Optimizations

```bash
# /etc/sysctl.conf optimizations for PostgreSQL

# Shared memory settings
kernel.shmmax = 68719476736          # 64GB
kernel.shmall = 4294967296           # 16TB
kernel.shmmni = 4096

# Memory management
vm.swappiness = 1                    # Minimize swapping
vm.dirty_background_ratio = 3        # Background write threshold
vm.dirty_ratio = 10                  # Foreground write threshold
vm.dirty_expire_centisecs = 500      # Dirty page expiration
vm.dirty_writeback_centisecs = 100   # Writeback frequency

# Network settings
net.core.rmem_max = 134217728        # Maximum receive buffer
net.core.wmem_max = 134217728        # Maximum send buffer
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728

# File system
fs.file-max = 65536                  # Maximum file descriptors
```

### Storage Configuration

```bash
# File system mount options for PostgreSQL data directory
/dev/sdb1 /var/lib/postgresql ext4 noatime,nobarrier,data=writeback 0 0

# I/O scheduler optimization (for SSDs)
echo noop > /sys/block/sdb/queue/scheduler

# Disable unnecessary services
systemctl disable bluetooth
systemctl disable cups
systemctl disable avahi-daemon
```

## Backup and Recovery

### Comprehensive Backup Strategy

#### 1. Physical Backups (pg_basebackup)

```bash
#!/bin/bash
# full_backup.sh - Complete database backup script

BACKUP_DIR="/backup/postgresql"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="$BACKUP_DIR/basebackup_$DATE"
LOG_FILE="/var/log/postgresql/backup.log"

# Create backup directory
mkdir -p $BACKUP_PATH

# Perform base backup
pg_basebackup -D $BACKUP_PATH \
              -Ft \
              -z \
              -P \
              -U backup_user \
              -h localhost \
              -p 5432 \
              -W \
              --checkpoint=fast \
              --write-recovery-conf

# Verify backup
if [ $? -eq 0 ]; then
    echo "$(date): Backup completed successfully: $BACKUP_PATH" >> $LOG_FILE
    
    # Cleanup old backups (keep last 7 days)
    find $BACKUP_DIR -type d -name "basebackup_*" -mtime +7 -exec rm -rf {} \;
else
    echo "$(date): Backup failed!" >> $LOG_FILE
    exit 1
fi
```

#### 2. WAL Archiving

```bash
#!/bin/bash
# wal_archive.sh - WAL archiving script

WAL_FILE=$1
WAL_PATH=$2
ARCHIVE_DIR="/backup/wal"
S3_BUCKET="your-backup-bucket"

# Local archive
cp "$WAL_PATH" "$ARCHIVE_DIR/$WAL_FILE"

# S3 archive (optional)
aws s3 cp "$WAL_PATH" "s3://$S3_BUCKET/wal/$WAL_FILE"

# Verify archive
if [ -f "$ARCHIVE_DIR/$WAL_FILE" ]; then
    exit 0
else
    exit 1
fi
```

#### 3. Point-in-Time Recovery Setup

```bash
# postgresql.conf settings for PITR
archive_mode = on
archive_command = '/path/to/wal_archive.sh %f %p'
wal_level = replica
max_wal_senders = 3
```

#### 4. Logical Backups

```bash
#!/bin/bash
# logical_backup.sh - Logical backup with pg_dump

DATABASE="production_db"
BACKUP_DIR="/backup/logical"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/${DATABASE}_$DATE.sql.gz"

# Create compressed dump
pg_dump -h localhost \
        -U backup_user \
        -d $DATABASE \
        --verbose \
        --format=custom \
        --compress=9 \
        --file="$BACKUP_FILE"

# Verify backup file
if [ -f "$BACKUP_FILE" ] && [ -s "$BACKUP_FILE" ]; then
    echo "Logical backup completed: $BACKUP_FILE"
    
    # Test restore to verify backup integrity
    pg_restore --list "$BACKUP_FILE" > /dev/null
    if [ $? -eq 0 ]; then
        echo "Backup verification successful"
    else
        echo "Backup verification failed!"
        exit 1
    fi
else
    echo "Backup failed!"
    exit 1
fi
```

### Automated Backup Scheduling

```bash
# /etc/cron.d/postgresql-backup

# Full backup every day at 2 AM
0 2 * * * postgres /scripts/full_backup.sh

# Logical backup every 6 hours
0 */6 * * * postgres /scripts/logical_backup.sh

# WAL backup verification every hour
0 * * * * postgres /scripts/verify_wal_archive.sh
```

## High Availability

### Streaming Replication Setup

#### Primary Server Configuration

```ini
# postgresql.conf on primary
listen_addresses = '*'
wal_level = replica
max_wal_senders = 10
max_replication_slots = 10
synchronous_standby_names = 'standby1'
```

```bash
# pg_hba.conf on primary
host replication replica_user standby_ip/32 md5
```

#### Standby Server Setup

```bash
# Stop standby PostgreSQL
systemctl stop postgresql

# Remove existing data directory
rm -rf /var/lib/postgresql/14/main/*

# Create base backup from primary
pg_basebackup -h primary_ip \
              -D /var/lib/postgresql/14/main \
              -U replica_user \
              -P \
              -W \
              --write-recovery-conf

# Create standby.signal file
touch /var/lib/postgresql/14/main/standby.signal

# Configure recovery
cat >> /var/lib/postgresql/14/main/postgresql.auto.conf << EOF
primary_conninfo = 'host=primary_ip port=5432 user=replica_user'
promote_trigger_file = '/tmp/promote_standby'
EOF

# Start standby
systemctl start postgresql
```

### Failover Procedures

#### Automatic Failover with Patroni

```yaml
# patroni.yml configuration
scope: postgres-cluster
namespace: /service/
name: node1

restapi:
  listen: 0.0.0.0:8008
  connect_address: node1_ip:8008

etcd:
  hosts: etcd1:2379,etcd2:2379,etcd3:2379

bootstrap:
  dcs:
    ttl: 30
    loop_wait: 10
    retry_timeout: 30
    maximum_lag_on_failover: 1048576
    postgresql:
      use_pg_rewind: true
      use_slots: true
      parameters:
        max_connections: 200
        shared_buffers: 4GB
        effective_cache_size: 12GB

postgresql:
  listen: 0.0.0.0:5432
  connect_address: node1_ip:5432
  data_dir: /var/lib/postgresql/14/main
  bin_dir: /usr/lib/postgresql/14/bin
  pgpass: /tmp/pgpass
  authentication:
    replication:
      username: replica_user
      password: replica_password
    superuser:
      username: postgres
      password: postgres_password

tags:
  nofailover: false
  noloadbalance: false
  clonefrom: false
  nosync: false
```

#### Manual Failover Script

```bash
#!/bin/bash
# manual_failover.sh - Manual failover procedure

PRIMARY_IP="10.0.1.10"
STANDBY_IP="10.0.1.11"
PROMOTE_FILE="/tmp/promote_standby"

echo "Starting manual failover procedure..."

# 1. Check primary status
if ping -c 3 $PRIMARY_IP > /dev/null; then
    echo "WARNING: Primary server is still reachable!"
    read -p "Continue with failover? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Failover cancelled"
        exit 1
    fi
fi

# 2. Promote standby
echo "Promoting standby server..."
ssh postgres@$STANDBY_IP "touch $PROMOTE_FILE"

# 3. Wait for promotion
echo "Waiting for promotion to complete..."
sleep 10

# 4. Verify new primary
if psql -h $STANDBY_IP -U postgres -c "SELECT pg_is_in_recovery();" | grep -q "f"; then
    echo "Failover completed successfully!"
    echo "New primary: $STANDBY_IP"
else
    echo "Failover failed!"
    exit 1
fi

# 5. Update application configuration
echo "Update application connection strings to point to: $STANDBY_IP"
```

## Security Hardening

### Network Security

```bash
# pg_hba.conf - Restrictive access control
# TYPE  DATABASE        USER            ADDRESS                 METHOD

# Local connections
local   all             postgres                                peer
local   all             all                                     md5

# IPv4 connections
host    all             all             127.0.0.1/32            md5
host    production_db   app_user        10.0.1.0/24            md5
host    replication     replica_user    10.0.1.0/24            md5

# Reject all other connections
host    all             all             0.0.0.0/0               reject
```

### SSL Configuration

```ini
# postgresql.conf SSL settings
ssl = on
ssl_cert_file = 'server.crt'
ssl_key_file = 'server.key'
ssl_ca_file = 'ca.crt'
ssl_ciphers = 'HIGH:!aNULL:!MD5'
ssl_prefer_server_ciphers = on
ssl_min_protocol_version = 'TLSv1.2'
```

### User and Role Management

```sql
-- Create application user with minimal privileges
CREATE ROLE app_user LOGIN PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE production_db TO app_user;
GRANT USAGE ON SCHEMA public TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_user;

-- Create read-only user for reporting
CREATE ROLE readonly_user LOGIN PASSWORD 'readonly_password';
GRANT CONNECT ON DATABASE production_db TO readonly_user;
GRANT USAGE ON SCHEMA public TO readonly_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;

-- Create backup user
CREATE ROLE backup_user LOGIN PASSWORD 'backup_password' REPLICATION;

-- Remove default public schema permissions
REVOKE ALL ON SCHEMA public FROM public;
```

### Audit Logging

```sql
-- Enable pgaudit extension
CREATE EXTENSION IF NOT EXISTS pgaudit;

-- Configure audit logging
SET pgaudit.log = 'all';
SET pgaudit.log_catalog = off;
SET pgaudit.log_parameter = on;
SET pgaudit.log_statement_once = on;
```

## Performance Tuning

### Query Optimization

```sql
-- Enable query statistics collection
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Identify slow queries
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    max_time,
    stddev_time,
    rows
FROM pg_stat_statements
WHERE mean_time > 1000  -- Queries taking more than 1 second
ORDER BY total_time DESC
LIMIT 20;

-- Index usage analysis
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch,
    pg_size_pretty(pg_relation_size(indexrelid)) as size
FROM pg_stat_user_indexes
WHERE idx_scan = 0  -- Unused indexes
ORDER BY pg_relation_size(indexrelid) DESC;
```

### Automatic Performance Tuning

```sql
-- Auto-tuning function for work_mem
CREATE OR REPLACE FUNCTION tune_work_mem()
RETURNS TEXT AS $$
DECLARE
    avg_query_mem BIGINT;
    recommended_work_mem TEXT;
BEGIN
    -- Calculate average memory usage from pg_stat_statements
    SELECT avg(temp_blks_written * 8192) INTO avg_query_mem
    FROM pg_stat_statements
    WHERE temp_blks_written > 0;
    
    IF avg_query_mem IS NULL THEN
        RETURN 'No temporary file usage detected';
    END IF;
    
    recommended_work_mem := pg_size_pretty(avg_query_mem * 2);
    
    RETURN 'Recommended work_mem: ' || recommended_work_mem;
END;
$$ LANGUAGE plpgsql;
```

## Operational Procedures

### Deployment Procedures

```bash
#!/bin/bash
# deploy_schema_changes.sh - Safe schema deployment

DATABASE="production_db"
MIGRATION_FILE="$1"
BACKUP_PREFIX="pre_migration"

if [ -z "$MIGRATION_FILE" ]; then
    echo "Usage: $0 <migration_file>"
    exit 1
fi

echo "Starting deployment procedure..."

# 1. Create backup
echo "Creating backup..."
pg_dump -h localhost -U postgres -d $DATABASE \
        --schema-only \
        --file="${BACKUP_PREFIX}_$(date +%Y%m%d_%H%M%S).sql"

# 2. Begin transaction
echo "Applying migration in transaction..."
psql -h localhost -U postgres -d $DATABASE << EOF
BEGIN;

-- Set lock timeout to prevent long-running locks
SET lock_timeout = '30s';

-- Apply migration
\i $MIGRATION_FILE

-- Verify migration
SELECT 'Migration completed successfully' as status;

COMMIT;
EOF

if [ $? -eq 0 ]; then
    echo "Migration completed successfully!"
else
    echo "Migration failed! Check logs and rollback if necessary."
    exit 1
fi
```

### Maintenance Procedures

```bash
#!/bin/bash
# maintenance_window.sh - Scheduled maintenance tasks

echo "Starting maintenance window at $(date)"

# 1. Update table statistics
echo "Updating statistics..."
psql -d production_db -c "ANALYZE;"

# 2. Vacuum tables if needed
echo "Running vacuum analysis..."
psql -d production_db << EOF
SELECT 
    schemaname,
    tablename,
    n_dead_tup,
    n_live_tup,
    round(n_dead_tup::float / (n_live_tup + n_dead_tup) * 100, 2) as dead_ratio
FROM pg_stat_user_tables
WHERE n_live_tup > 0
  AND n_dead_tup::float / (n_live_tup + n_dead_tup) > 0.1
ORDER BY dead_ratio DESC;
EOF

# 3. Reindex if fragmentation is high
echo "Checking index bloat..."
psql -d production_db -c "REINDEX INDEX CONCURRENTLY idx_heavily_used;"

# 4. Clean up old log files
echo "Cleaning up log files..."
find /var/log/postgresql -name "*.log" -mtime +30 -delete

echo "Maintenance window completed at $(date)"
```

### Health Check Scripts

```python
#!/usr/bin/env python3
# health_check.py - Comprehensive health check

import psycopg2
import sys
import time
from datetime import datetime, timedelta

class PostgreSQLHealthCheck:
    def __init__(self, connection_string):
        self.conn_string = connection_string
        self.checks = []
        
    def add_check(self, name, check_function):
        self.checks.append((name, check_function))
    
    def run_check(self, name, check_function):
        try:
            start_time = time.time()
            result = check_function()
            end_time = time.time()
            
            return {
                'name': name,
                'status': 'PASS' if result['success'] else 'FAIL',
                'message': result['message'],
                'duration': round(end_time - start_time, 3)
            }
        except Exception as e:
            return {
                'name': name,
                'status': 'ERROR',
                'message': str(e),
                'duration': 0
            }
    
    def check_connection(self):
        try:
            conn = psycopg2.connect(self.conn_string)
            conn.close()
            return {'success': True, 'message': 'Database connection successful'}
        except Exception as e:
            return {'success': False, 'message': f'Connection failed: {e}'}
    
    def check_replication_lag(self):
        try:
            conn = psycopg2.connect(self.conn_string)
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        client_addr,
                        state,
                        pg_wal_lsn_diff(pg_current_wal_lsn(), flush_lsn) as lag_bytes
                    FROM pg_stat_replication
                """)
                replicas = cur.fetchall()
                
                if not replicas:
                    return {'success': True, 'message': 'No replicas configured'}
                
                max_lag = max(replica[2] for replica in replicas if replica[2])
                if max_lag > 1024 * 1024:  # 1MB
                    return {'success': False, 'message': f'High replication lag: {max_lag} bytes'}
                else:
                    return {'success': True, 'message': f'Replication lag OK: {max_lag} bytes'}
                    
        except Exception as e:
            return {'success': False, 'message': f'Replication check failed: {e}'}
    
    def check_long_queries(self):
        try:
            conn = psycopg2.connect(self.conn_string)
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT count(*) 
                    FROM pg_stat_activity 
                    WHERE state != 'idle' 
                      AND now() - query_start > interval '5 minutes'
                """)
                long_queries = cur.fetchone()[0]
                
                if long_queries > 5:
                    return {'success': False, 'message': f'{long_queries} long-running queries detected'}
                else:
                    return {'success': True, 'message': f'{long_queries} long-running queries'}
                    
        except Exception as e:
            return {'success': False, 'message': f'Long query check failed: {e}'}
    
    def run_all_checks(self):
        results = []
        
        # Add all checks
        self.add_check('Database Connection', self.check_connection)
        self.add_check('Replication Lag', self.check_replication_lag)
        self.add_check('Long Running Queries', self.check_long_queries)
        
        # Run checks
        for name, check_func in self.checks:
            result = self.run_check(name, check_func)
            results.append(result)
            
        return results

if __name__ == "__main__":
    conn_string = "postgresql://monitor_user:password@localhost:5432/production_db"
    health_check = PostgreSQLHealthCheck(conn_string)
    
    results = health_check.run_all_checks()
    
    print(f"Health Check Report - {datetime.now()}")
    print("=" * 50)
    
    all_passed = True
    for result in results:
        status_symbol = "✓" if result['status'] == 'PASS' else "✗"
        print(f"{status_symbol} {result['name']}: {result['message']} ({result['duration']}s)")
        
        if result['status'] != 'PASS':
            all_passed = False
    
    print("=" * 50)
    print(f"Overall Status: {'HEALTHY' if all_passed else 'UNHEALTHY'}")
    
    sys.exit(0 if all_passed else 1)
```

## Troubleshooting

### Common Production Issues

#### 1. Connection Pool Exhaustion

```sql
-- Identify connection usage
SELECT 
    datname,
    usename,
    application_name,
    state,
    count(*) as connections
FROM pg_stat_activity
GROUP BY datname, usename, application_name, state
ORDER BY connections DESC;

-- Kill idle connections
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE state = 'idle'
  AND state_change < now() - interval '1 hour';
```

#### 2. Lock Contention

```sql
-- Identify blocking queries
SELECT 
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_statement,
    blocking_activity.query AS blocking_statement
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.GRANTED;
```

#### 3. High CPU Usage

```sql
-- Find CPU-intensive queries
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    (total_time / sum(total_time) OVER ()) * 100 AS percentage
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;
```

### Emergency Procedures

```bash
#!/bin/bash
# emergency_response.sh - Emergency response procedures

EMERGENCY_TYPE="$1"

case $EMERGENCY_TYPE in
    "high_connections")
        echo "Terminating idle connections..."
        psql -d postgres -c "
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE state = 'idle'
              AND state_change < now() - interval '30 minutes';"
        ;;
        
    "disk_full")
        echo "Cleaning up temporary files and old logs..."
        find /tmp -name "pgsql_tmp*" -mtime +1 -delete
        find /var/log/postgresql -name "*.log" -mtime +7 -delete
        ;;
        
    "replication_broken")
        echo "Attempting to restart replication..."
        systemctl restart postgresql
        ;;
        
    *)
        echo "Unknown emergency type: $EMERGENCY_TYPE"
        echo "Available types: high_connections, disk_full, replication_broken"
        exit 1
        ;;
esac
```

## Summary

Running PostgreSQL in production successfully requires:

1. **Proper Architecture**: Well-designed infrastructure with redundancy
2. **Optimized Configuration**: Tuned for your specific workload and hardware
3. **Robust Backup Strategy**: Multiple backup types with tested recovery procedures
4. **High Availability**: Replication and failover mechanisms
5. **Security Hardening**: Network, authentication, and audit controls
6. **Performance Monitoring**: Continuous monitoring and alerting
7. **Operational Procedures**: Documented processes for deployments and maintenance
8. **Emergency Response**: Prepared procedures for common issues

Key success factors:
- Test all procedures in staging environments
- Monitor proactively, not reactively
- Document everything and train your team
- Plan for growth and scalability
- Regular security audits and updates
- Continuous performance optimization

Production PostgreSQL requires ongoing attention and maintenance, but with proper planning and procedures, it can provide reliable, high-performance database services for critical applications.
