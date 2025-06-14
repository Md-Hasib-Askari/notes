# 🌟 Redis Cluster

## 🚀 Overview
- **Goal**: Learn to set up and manage Redis Cluster for horizontal scaling, data sharding, and high availability.
- **Purpose**: Enable Redis to handle large datasets and high traffic by distributing data across multiple nodes, with built-in fault tolerance.
- **Tools**: `redis-cli`, Redis configuration file (`redis.conf`), RedisInsight, multiple Redis instances or Docker.

## 🧩 What is Redis Cluster?
- **Description**: A distributed implementation of Redis that shards data across multiple nodes, providing scalability and high availability.
- **Key Features**:
  - **Sharding**: Data is partitioned into 16,384 hash slots, distributed across nodes.
  - **Replication**: Each node can have replicas for fault tolerance.
  - **Automatic Failover**: Nodes communicate to detect failures and promote replicas.
  - **Decentralized**: No central coordinator; nodes manage the cluster via gossip protocol.
- **Use Case**: Large-scale applications (e.g., caching, analytics) requiring high throughput and data distribution.

## 🔑 How Redis Cluster Works
- **Hash Slots**: Keys are hashed (using CRC16) to one of 16,384 slots, assigned to nodes.
  - Example: Key `user:1` hashes to slot 1234, handled by a specific node.
- **Master and Replica Nodes**:
  - Masters handle writes and a subset of slots.
  - Replicas replicate master data for redundancy and read scaling.
- **Failover**: If a master fails, a replica is promoted by cluster consensus.
- **Client Interaction**: Clients query any node; nodes redirect to the correct node if needed.

## 🛠️ Setting Up Redis Cluster
- **Requirements**:
  - Minimum 3 master nodes (each with optional replicas) for fault tolerance.
  - Each node runs a Redis instance with `cluster-enabled yes`.
- **Configuration** (in `redis.conf` for each node):
  ```conf
  port 7000  # Unique port per node (e.g., 7000, 7001, 7002)
  cluster-enabled yes
  cluster-config-file nodes.conf  # Auto-generated node config
  cluster-node-timeout 5000  # Timeout for node failure detection
  ```
- **Manual Setup** (6 nodes: 3 masters, 3 replicas):
  1. Create 6 directories (e.g., `redis-7000`, `redis-7001`, ..., `redis-7005`).
  2. Create `redis.conf` in each with unique ports (7000–7005) and `cluster-enabled yes`.
  3. Start each instance:
     ```bash
     redis-server redis-7000/redis.conf
     redis-server redis-7001/redis.conf
     # ... and so on
     ```
  4. Create cluster:
     ```bash
     redis-cli --cluster create 127.0.0.1:7000 127.0.0.1:7001 127.0.0.1:7002 \
       127.0.0.1:7003 127.0.0.1:7004 127.0.0.1:7005 \
       --cluster-replicas 1
     ```
     - Assigns 3 masters (7000–7002), 3 replicas (7003–7005).
     - Distributes 16,384 slots across masters.
  5. Verify:
     ```bash
     127.0.0.1:7000> CLUSTER INFO
     cluster_state:ok
     cluster_slots_assigned:16384
     ...
     127.0.0.1:7000> CLUSTER NODES
     # Lists nodes, roles, and slots
     ```

- **Docker Setup** (Alternative):
  ```bash
  # Start 6 nodes
  for port in {7000..7005}; do
    docker run -d -p $port:6379 --name redis-$port redis redis-server --cluster-enabled yes --port 6379
  done
  # Create cluster
  redis-cli --cluster create 127.0.0.1:7000 127.0.0.1:7001 127.0.0.1:7002 \
    127.0.0.1:7003 127.0.0.1:7004 127.0.0.1:7005 \
    --cluster-replicas 1
  ```

## 🔍 Cluster Commands
- **CLUSTER INFO**: Show cluster status (e.g., `cluster_state:ok`).
- **CLUSTER NODES**: List nodes, their roles, and assigned slots.
- **CLUSTER SLOTS**: Show slot distribution across nodes.
- **CLUSTER ADDSLOTS slot [slot ...]**: Assign slots to a node (manual setup).
- **CLUSTER DELSLOTS slot [slot ...]**: Remove slots from a node.
- **CLUSTER FAILOVER**: Trigger manual failover on a replica.
- **Example**:
  ```bash
  127.0.0.1:7000> SET user:1 "Alice"
  OK
  127.0.0.1:7000> CLUSTER SLOTS
  1) 1) (integer) 0
     2) (integer) 5460
     3) 1) "127.0.0.1"
        2) (integer) 7000
     ...
  127.0.0.1:7000> CLUSTER NODES
  # Shows node IDs, roles, and slots
  ```

## 🐍 Using Redis Cluster with Python
- **Library**: Use `redis-py` with cluster support (`redis-py-cluster` or `redis.RedisCluster`).
- **Setup**:
  ```bash
  pip install redis
  ```
- **Example**:
  ```python
  from redis.cluster import RedisCluster

  # Connect to cluster
  rc = RedisCluster(startup_nodes=[{"host": "127.0.0.1", "port": "7000"}])

  # Basic operations
  rc.set("user:1", "Alice")
  print(rc.get("user:1"))  # Output: b'Alice'

  # Cluster info
  print(rc.cluster_info())  # Shows cluster status
  ```
- **Note**: `RedisCluster` automatically handles slot redirection.

## 🛡️ Testing Failover
- **Steps**:
  1. Set data: `SET key1 "value1"` on any node.
  2. Stop a master (e.g., `redis-cli -p 7000 SHUTDOWN`).
  3. Verify failover:
     ```bash
     127.0.0.1:7001> CLUSTER NODES
     # Check if a replica (e.g., 7003) is now master
     127.0.0.1:7001> GET key1
     "value1"
     ```
- **Monitor**: Use RedisInsight to visualize node roles and slot distribution.

## 🛠️ Practice Tips
- Set up a 6-node cluster locally or with Docker.
- Add data and verify it’s sharded across nodes (`CLUSTER SLOTS`).
- Simulate a master failure and confirm failover.
- Use `redis-py` to interact with the clade.
- Monitor cluster status with RedisInsight or `CLUSTER INFO`.

## 📚 Resources
- **Official Docs**: [redis.io/docs/management/scaling/](https://redis.io/docs/management/scaling/)
- **Command Reference**: [redis.io/commands/](https://redis.io/commands/) (e.g., `CLUSTER`)
- **Redis University**: [university.redis.com](https://university.redis.com) (free “Redis Cluster” course)
- **Redis Labs Blog**: [redis.com/blog](https://redis.com/blog) (search for “cluster”)
- **TryRedis**: [try.redis.io](https://try.redis.io) (to test cluster commands)

## 🔑 Key Takeaways
- Redis Cluster shards data across nodes for scalability and high availability.
- Requires at least 3 masters with optional replicas for fault tolerance.
- Use `redis-cli --cluster` to create and manage clusters.
- Client libraries like `redis-py` simplify cluster interactions.
- Test failover to ensure reliability in production.