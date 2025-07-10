# 19. Monitoring and Logging

## Overview
Effective monitoring and logging are crucial for maintaining PostgreSQL database performance, identifying issues, and ensuring system reliability. This guide covers comprehensive monitoring strategies, log analysis, and alerting systems.

## Table of Contents
- [Database Monitoring Fundamentals](#database-monitoring-fundamentals)
- [Built-in Monitoring Views](#built-in-monitoring-views)
- [Logging Configuration](#logging-configuration)
- [Log Analysis](#log-analysis)
- [Performance Monitoring](#performance-monitoring)
- [External Monitoring Tools](#external-monitoring-tools)
- [Alerting and Notifications](#alerting-and-notifications)
- [Monitoring Best Practices](#monitoring-best-practices)

## Database Monitoring Fundamentals

### Key Metrics to Monitor

1. **Performance Metrics**
   - Query execution time
   - Throughput (transactions per second)
   - Connection count
   - Cache hit ratios

2. **Resource Usage**
   - CPU utilization
   - Memory usage
   - Disk I/O
   - Network traffic

3. **Database Health**
   - Replication lag
   - Lock contention
   - Deadlocks
   - Error rates

## Built-in Monitoring Views

### pg_stat_activity

Monitor current database activity:

```sql
-- Current active connections and queries
SELECT 
    pid,
    usename,
    application_name,
    client_addr,
    state,
    query_start,
    state_change,
    query
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY query_start;

-- Long-running queries
SELECT 
    pid,
    now() - query_start AS duration,
    query
FROM pg_stat_activity
WHERE state != 'idle'
  AND now() - query_start > interval '5 minutes'
ORDER BY duration DESC;

-- Blocked queries
SELECT 
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_statement,
    blocking_activity.query AS blocking_statement
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity 
    ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks 
    ON blocking_locks.locktype = blocked_locks.locktype
    AND blocking_locks.DATABASE IS NOT DISTINCT FROM blocked_locks.DATABASE
    AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
    AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
    AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
    AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
    AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
    AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
    AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
    AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
    AND blocking_locks.pid != blocked_locks.pid
JOIN pg_catalog.pg_stat_activity blocking_activity 
    ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.GRANTED;
```

### pg_stat_database

Database-level statistics:

```sql
-- Database statistics
SELECT 
    datname,
    numbackends,
    xact_commit,
    xact_rollback,
    blks_read,
    blks_hit,
    tup_returned,
    tup_fetched,
    tup_inserted,
    tup_updated,
    tup_deleted
FROM pg_stat_database
WHERE datname = 'your_database_name';

-- Cache hit ratio by database
SELECT 
    datname,
    blks_hit,
    blks_read,
    round(
        (blks_hit::float / (blks_hit + blks_read) * 100)::numeric, 2
    ) AS cache_hit_ratio
FROM pg_stat_database
WHERE blks_read > 0;
```

### pg_stat_user_tables

Table-level statistics:

```sql
-- Table usage statistics
SELECT 
    schemaname,
    tablename,
    n_tup_ins,
    n_tup_upd,
    n_tup_del,
    n_live_tup,
    n_dead_tup,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables
ORDER BY n_tup_ins + n_tup_upd + n_tup_del DESC;

-- Tables needing vacuum/analyze
SELECT 
    schemaname,
    tablename,
    n_dead_tup,
    n_live_tup,
    round(n_dead_tup::float / (n_live_tup + n_dead_tup) * 100, 2) AS dead_tuple_ratio,
    last_autovacuum,
    last_autoanalyze
FROM pg_stat_user_tables
WHERE n_live_tup > 0
  AND n_dead_tup::float / (n_live_tup + n_dead_tup) > 0.1
ORDER BY dead_tuple_ratio DESC;
```

### pg_stat_statements

Query performance monitoring (requires pg_stat_statements extension):

```sql
-- Enable pg_stat_statements
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Top queries by total time
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    max_time,
    rows,
    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;

-- Slowest queries by average time
SELECT 
    query,
    calls,
    mean_time,
    max_time,
    stddev_time
FROM pg_stat_statements
WHERE calls > 10
ORDER BY mean_time DESC
LIMIT 10;

-- Most frequently executed queries
SELECT 
    query,
    calls,
    total_time,
    mean_time
FROM pg_stat_statements
ORDER BY calls DESC
LIMIT 10;
```

## Logging Configuration

### postgresql.conf Logging Settings

```ini
# Enable logging
logging_collector = on
log_destination = 'stderr'
log_directory = 'log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_file_mode = 0600
log_rotation_age = 1d
log_rotation_size = 100MB
log_truncate_on_rotation = on

# What to log
log_min_messages = warning
log_min_error_statement = error
log_min_duration_statement = 1000  # Log queries taking > 1 second

# Connection logging
log_connections = on
log_disconnections = on
log_hostname = off

# Statement logging
log_statement = 'ddl'  # 'none', 'ddl', 'mod', 'all'
log_duration = off
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '

# Lock logging
log_lock_waits = on
deadlock_timeout = 1s

# Checkpoint logging
log_checkpoints = on
log_temp_files = 0

# Auto-vacuum logging
log_autovacuum_min_duration = 0
```

### Advanced Logging Configuration

```ini
# Detailed logging for troubleshooting
log_statement = 'all'
log_duration = on
log_min_duration_statement = 0
log_executor_stats = on
log_parser_stats = on
log_planner_stats = on

# CSV logging for analysis
log_destination = 'csvlog'
```

## Log Analysis

### Common Log Patterns

```bash
# Find slow queries
grep "duration:" /var/log/postgresql/postgresql.log | \
awk '{print $4, $0}' | sort -nr | head -10

# Find connection errors
grep "FATAL" /var/log/postgresql/postgresql.log

# Find lock waits
grep "still waiting for" /var/log/postgresql/postgresql.log

# Find checkpoints taking too long
grep "checkpoint complete" /var/log/postgresql/postgresql.log
```

### Log Analysis with pgBadger

```bash
# Install pgBadger
cpan App::pgBadger
# or
apt-get install pgbadger

# Generate report
pgbadger /var/log/postgresql/postgresql.log -o report.html

# Real-time analysis
pgbadger --incremental --outdir /var/www/html/pgbadger/ /var/log/postgresql/
```

### Custom Log Analysis Scripts

```bash
#!/bin/bash
# slow_queries.sh - Find slow queries in logs

LOG_FILE="/var/log/postgresql/postgresql.log"
THRESHOLD=5000  # 5 seconds

echo "Queries taking longer than ${THRESHOLD}ms:"
grep "duration:" $LOG_FILE | \
awk -v threshold=$THRESHOLD '
{
    duration_pos = index($0, "duration: ") + 10
    duration_end = index(substr($0, duration_pos), " ms")
    duration = substr($0, duration_pos, duration_end - 1)
    
    if (duration > threshold) {
        print duration "ms: " $0
    }
}' | sort -nr | head -20
```

## Performance Monitoring

### System-Level Monitoring

```sql
-- Database size monitoring
SELECT 
    datname,
    pg_size_pretty(pg_database_size(datname)) as size
FROM pg_database
ORDER BY pg_database_size(datname) DESC;

-- Table size monitoring
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) as index_size
FROM pg_tables
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
LIMIT 20;

-- Index usage statistics
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_tup_read,
    idx_tup_fetch,
    idx_scan,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
FROM pg_stat_user_indexes
ORDER BY idx_scan ASC;
```

### Real-time Performance Monitoring

```sql
-- Create monitoring views
CREATE OR REPLACE VIEW active_queries AS
SELECT 
    pid,
    usename,
    application_name,
    client_addr,
    state,
    query_start,
    now() - query_start AS duration,
    waiting,
    query
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY query_start;

-- Connection monitoring
CREATE OR REPLACE VIEW connection_stats AS
SELECT 
    datname,
    usename,
    application_name,
    state,
    count(*) as connections
FROM pg_stat_activity
GROUP BY datname, usename, application_name, state
ORDER BY connections DESC;
```

### Performance Monitoring Functions

```sql
-- Function to get top queries
CREATE OR REPLACE FUNCTION get_top_queries(limit_count INTEGER DEFAULT 10)
RETURNS TABLE(
    query TEXT,
    calls BIGINT,
    total_time DOUBLE PRECISION,
    mean_time DOUBLE PRECISION,
    max_time DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pss.query,
        pss.calls,
        pss.total_time,
        pss.mean_time,
        pss.max_time
    FROM pg_stat_statements pss
    ORDER BY pss.total_time DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Usage
SELECT * FROM get_top_queries(5);
```

## External Monitoring Tools

### Prometheus + Grafana Setup

```yaml
# docker-compose.yml for monitoring stack
version: '3.8'
services:
  postgres_exporter:
    image: prometheuscommunity/postgres-exporter
    environment:
      DATA_SOURCE_NAME: "postgresql://username:password@postgres:5432/dbname?sslmode=disable"
    ports:
      - "9187:9187"
    
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres_exporter:9187']
```

### pgMonitor Setup

```bash
# Install pgMonitor
git clone https://github.com/CrunchyData/pgmonitor.git
cd pgmonitor

# Setup Prometheus
cp prometheus/linux/crunchy-prometheus.yml /etc/prometheus/
systemctl restart prometheus

# Setup Grafana dashboards
# Import dashboards from grafana/linux/ directory
```

### Custom Monitoring Script

```python
#!/usr/bin/env python3
# postgresql_monitor.py

import psycopg2
import time
import json
import logging
from datetime import datetime

class PostgreSQLMonitor:
    def __init__(self, connection_string):
        self.conn_string = connection_string
        
    def get_connection(self):
        return psycopg2.connect(self.conn_string)
    
    def check_connections(self):
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        count(*) as total_connections,
                        count(*) FILTER (WHERE state = 'active') as active,
                        count(*) FILTER (WHERE state = 'idle') as idle,
                        count(*) FILTER (WHERE state = 'idle in transaction') as idle_in_transaction
                    FROM pg_stat_activity
                """)
                return dict(zip([desc[0] for desc in cur.description], cur.fetchone()))
    
    def check_slow_queries(self, threshold_seconds=5):
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        pid,
                        usename,
                        application_name,
                        state,
                        now() - query_start as duration,
                        query
                    FROM pg_stat_activity
                    WHERE state != 'idle'
                      AND now() - query_start > interval '%s seconds'
                    ORDER BY duration DESC
                """, (threshold_seconds,))
                
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]
    
    def check_locks(self):
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        mode,
                        locktype,
                        count(*) as count
                    FROM pg_locks
                    GROUP BY mode, locktype
                    ORDER BY count DESC
                """)
                
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]
    
    def monitor(self):
        timestamp = datetime.now().isoformat()
        
        metrics = {
            'timestamp': timestamp,
            'connections': self.check_connections(),
            'slow_queries': self.check_slow_queries(),
            'locks': self.check_locks()
        }
        
        return metrics

if __name__ == "__main__":
    monitor = PostgreSQLMonitor("postgresql://user:password@localhost:5432/dbname")
    
    while True:
        try:
            metrics = monitor.monitor()
            print(json.dumps(metrics, indent=2, default=str))
            
            # Alert on slow queries
            if metrics['slow_queries']:
                logging.warning(f"Found {len(metrics['slow_queries'])} slow queries")
            
            # Alert on high connection count
            if metrics['connections']['total_connections'] > 80:
                logging.warning(f"High connection count: {metrics['connections']['total_connections']}")
                
        except Exception as e:
            logging.error(f"Monitoring error: {e}")
        
        time.sleep(60)  # Check every minute
```

## Alerting and Notifications

### Slack Integration

```python
import requests
import json

def send_slack_alert(webhook_url, message):
    payload = {
        'text': message,
        'username': 'PostgreSQL Monitor',
        'icon_emoji': ':warning:'
    }
    
    response = requests.post(webhook_url, json=payload)
    return response.status_code == 200

# Usage
webhook_url = "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
send_slack_alert(webhook_url, "PostgreSQL: High connection count detected!")
```

### Email Alerts

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email_alert(smtp_server, port, username, password, to_email, subject, message):
    msg = MIMEMultipart()
    msg['From'] = username
    msg['To'] = to_email
    msg['Subject'] = subject
    
    msg.attach(MIMEText(message, 'plain'))
    
    try:
        server = smtplib.SMTP(smtp_server, port)
        server.starttls()
        server.login(username, password)
        text = msg.as_string()
        server.sendmail(username, to_email, text)
        server.quit()
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False
```

### Alert Rules

```sql
-- Create alert functions
CREATE OR REPLACE FUNCTION check_connection_limit()
RETURNS BOOLEAN AS $$
DECLARE
    conn_count INTEGER;
    max_conn INTEGER;
BEGIN
    SELECT count(*) INTO conn_count FROM pg_stat_activity;
    SHOW max_connections INTO max_conn;
    
    RETURN conn_count > (max_conn * 0.8);
END;
$$ LANGUAGE plpgsql;

-- Check for long-running transactions
CREATE OR REPLACE FUNCTION check_long_transactions()
RETURNS TABLE(pid INTEGER, duration INTERVAL, query TEXT) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pg_stat_activity.pid,
        now() - xact_start as duration,
        pg_stat_activity.query
    FROM pg_stat_activity
    WHERE xact_start IS NOT NULL
      AND now() - xact_start > interval '1 hour';
END;
$$ LANGUAGE plpgsql;
```

## Monitoring Best Practices

### Monitoring Checklist

1. **Database Health**
   - [ ] Connection count and limits
   - [ ] Query performance trends
   - [ ] Lock contention
   - [ ] Replication lag (if applicable)

2. **Resource Usage**
   - [ ] CPU utilization
   - [ ] Memory usage
   - [ ] Disk space and I/O
   - [ ] Network traffic

3. **Query Performance**
   - [ ] Slow query identification
   - [ ] Index usage efficiency
   - [ ] Cache hit ratios
   - [ ] Query plan changes

4. **Maintenance Operations**
   - [ ] Vacuum and analyze status
   - [ ] Backup completion
   - [ ] Log rotation
   - [ ] Statistics updates

### Automated Health Checks

```bash
#!/bin/bash
# postgresql_health_check.sh

DB_NAME="your_database"
DB_USER="monitor_user"
ALERT_EMAIL="admin@example.com"

# Function to send alert
send_alert() {
    echo "$1" | mail -s "PostgreSQL Alert" $ALERT_EMAIL
}

# Check if PostgreSQL is running
if ! pgrep -x "postgres" > /dev/null; then
    send_alert "PostgreSQL is not running!"
    exit 1
fi

# Check connection count
CONN_COUNT=$(psql -d $DB_NAME -U $DB_USER -t -c "SELECT count(*) FROM pg_stat_activity;")
if [ $CONN_COUNT -gt 80 ]; then
    send_alert "High connection count: $CONN_COUNT"
fi

# Check for slow queries
SLOW_QUERIES=$(psql -d $DB_NAME -U $DB_USER -t -c "
    SELECT count(*) FROM pg_stat_activity 
    WHERE state != 'idle' 
      AND now() - query_start > interval '5 minutes';
")
if [ $SLOW_QUERIES -gt 0 ]; then
    send_alert "Found $SLOW_QUERIES slow queries"
fi

# Check disk space
DISK_USAGE=$(df /var/lib/postgresql | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 85 ]; then
    send_alert "High disk usage: ${DISK_USAGE}%"
fi

echo "Health check completed at $(date)"
```

### Performance Baseline

```sql
-- Create baseline performance metrics
CREATE TABLE performance_baseline (
    timestamp TIMESTAMP DEFAULT now(),
    metric_name TEXT,
    metric_value NUMERIC,
    notes TEXT
);

-- Collect baseline metrics
INSERT INTO performance_baseline (metric_name, metric_value, notes)
VALUES 
    ('avg_query_time', (SELECT avg(mean_time) FROM pg_stat_statements), 'Average query execution time'),
    ('cache_hit_ratio', (SELECT sum(blks_hit)::float / sum(blks_hit + blks_read) * 100 FROM pg_stat_database), 'Buffer cache hit ratio'),
    ('total_connections', (SELECT count(*) FROM pg_stat_activity), 'Current connection count');
```

## Summary

Effective PostgreSQL monitoring requires:

1. **Comprehensive Coverage**: Monitor all aspects - performance, resources, and health
2. **Proactive Alerting**: Set up alerts before problems become critical
3. **Historical Analysis**: Keep historical data for trend analysis
4. **Automation**: Automate routine monitoring tasks
5. **Documentation**: Document normal baselines and escalation procedures

Key monitoring areas:
- Query performance and slow queries
- Connection management and limits
- Resource utilization (CPU, memory, disk)
- Database health indicators
- Log analysis and error tracking

Regular monitoring helps prevent issues, optimize performance, and ensure database reliability in production environments.
