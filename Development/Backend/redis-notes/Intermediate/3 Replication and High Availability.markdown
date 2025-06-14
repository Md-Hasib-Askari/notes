# 🌟 Replication and High Availability

## 🚀 Overview
- **Goal**: Learn how to set up Redis replication for data redundancy and use Redis Sentinel for automatic failover to ensure high availability.
- **Purpose**: Enable fault-tolerant Redis deployments for production environments, minimizing downtime and data loss.
- **Tools**: `redis-cli`, Redis configuration file (`redis.conf`), multiple Redis instances, Redis Sentinel.

## 🖇️ Replication
- **Description**: Redis replication allows a master instance to replicate its data to one or more slave (replica) instances in real-time.
- **How It Works**:
  - Master accepts writes and propagates changes to slaves.
  - Slaves are read-only, serving queries to reduce load on the master.
  - Asynchronous replication: Slaves lag slightly behind the master.
- **Use Case**: Scale read operations, provide data redundancy, or prepare for failover.
- **Configuration**:
  - **On Slave**: Use the `SLAVEOF` command or `replicaof` in `redis.conf`.
  - Example (`redis.conf` for slave):
    ```
    replicaof 127.0.0.1 6379  # Master’s host and port
    ```
  - **Dynamic Setup**:
    ```bash
    127.0.0.1:6380> SLAVEOF 127.0.0.1 6379
    OK
    ```
- **Commands**:
  - `INFO REPLICATION`: Check replication status on master or slave.
  - `SLAVEOF NO ONE`: Promote a slave to master (e.g., during failover).
- **Example** (Set up one master, one slave):
  1. Start master: `redis-server --port 6379`.
  2. Start slave: `redis-server --port 6380`.
  3. On slave: `redis-cli -p 6380 SLAVEOF 127.0.0.1 6379`.
  4. On master, set data:
     ```bash
     127.0.0.1:6379> SET key1 "value1"
     OK
     ```
  5. On slave, verify:
     ```bash
     127.0.0.1:6380> GET key1
     "value1"
     ```
  6. Check status:
     ```bash
     127.0.0.1:6379> INFO REPLICATION
     # Replication
     role:master
     connected_slaves:1
     ...
     127.0.0.1:6380> INFO REPLICATION
     # Replication
     role:slave
     master_host:127.0.0.1
     master_port:6379
     ...
     ```

## 🛡️ Redis Sentinel
- **Description**: A distributed system for monitoring Redis instances, detecting master failures, and performing automatic failover to a slave.
- **How It Works**:
  - Sentinels monitor master and slaves, electing a new master if the current one fails.
  - Clients connect to Sentinel to discover the current master.
  - Requires at least three Sentinel instances for quorum (majority agreement).
- **Use Case**: Ensure high availability in production (e.g., for session storage or caching).
- **Configuration** (in `sentinel.conf`):
  - Example:
    ```
    port 26379
    sentinel monitor mymaster 127.0.0.1 6379 2  # Monitor master, quorum=2
    sentinel down-after-milliseconds mymaster 5000
    sentinel failover-timeout mymaster 60000
    ```
  - Key settings:
    - `sentinel monitor <name> <host> <port> <quorum>`: Name master and set quorum.
    - `down-after-milliseconds`: Time to mark master as down.
    - `failover-timeout`: Time before retrying failover.
- **Starting Sentinel**:
  ```bash
  redis-sentinel /path/to/sentinel.conf
  # OR
  redis-server /path/to/sentinel.conf --sentinel
  ```
- **Commands**:
  - `SENTINEL MASTERS`: List monitored masters.
  - `SENTINEL GET-MASTER-ADDR-BY-NAME <name>`: Get current master’s address.
  - `SENTINEL FAILOVER <name>`: Trigger manual failover.
- **Example** (Set up master, slave, and Sentinel):
  1. Start master: `redis-server --port 6379`.
  2. Start slave: `redis-server --port 6380 --replicaof 127.0.0.1 6379`.
  3. Create `sentinel.conf`:
     ```
     port 26379
     sentinel monitor mymaster 127.0.0.1 6379 1
     sentinel down-after-milliseconds mymaster 5000
     ```
  4. Start Sentinel: `redis-sentinel sentinel.conf`.
  5. Test failover:
     - Stop master: `redis-cli -p 6379 SHUTDOWN`.
     - Check Sentinel: `redis-cli -p 26379 SENTINEL GET-MASTER-ADDR-BY-NAME mymaster`.
     - Expected: Slave (port 6380) becomes master.
  6. Verify:
     ```bash
     127.0.0.1:26379> SENTINEL MASTERS
     ...
     127.0.0.1:26379> SENTINEL GET-MASTER-ADDR-BY-NAME mymaster
     1) "127.0.0.1"
     2) "6380"
     ```

## 🛠️ Practice Setup
- **Local Environment**:
  - Run multiple Redis instances on different ports (e.g., 6379 for master, 6380/6381 for slaves).
  - Use `redis.conf` for each instance to set `port` and `replicaof`.
  - Run at least one Sentinel instance (port 26379).
- **Docker (Alternative)**:
  ```bash
  # Master
  docker run -d -p 6379:6379 --name redis-master redis
  # Slave
  docker run -d -p 6380:6379 --name redis-slave redis redis-server --replicaof redis-master 6379
  # Sentinel
  docker run -d -p 26379:26379 --name redis-sentinel redis redis-sentinel /etc/redis/sentinel.conf
  ```
- **Verify Replication**: Set data on master, read from slave.
- **Test Failover**: Shut down master, check if Sentinel promotes a slave.

## 📚 Resources
- **Official Docs**: [redis.io/docs/management/replication/](https://redis.io/docs/management/replication/), [redis.io/docs/management/sentinel/](https://redis.io/docs/management/sentinel/)
- **Command Reference**: [redis.io/commands/](https://redis.io/commands/) (e.g., `SLAVEOF`, `SENTINEL`)
- **Redis University**: [university.redis.com](https://university.redis.com) (free “Redis Administration” course)
- **Redis Labs Blog**: [redis.com/blog](https://redis.com/blog) (search for “replication” or “sentinel”)
- **TryRedis**: [try.redis.io](https://try.redis.io) (to test `INFO REPLICATION`)

## 🔑 Key Takeaways
- Replication provides read scalability and data redundancy with master-slave setups.
- Redis Sentinel ensures high availability by monitoring and handling failover.
- Configure replication with `replicaof` and Sentinel with `sentinel.conf`.
- Test failover scenarios to ensure reliability in production.