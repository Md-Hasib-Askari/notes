# 🌟 Performance Optimization

## 🚀 Overview
- **Goal**: Learn to optimize Redis performance by managing memory, configuring eviction policies, and monitoring bottlenecks using built-in tools.
- **Purpose**: Ensure Redis runs efficiently in production, handling large datasets and high throughput with minimal latency.
- **Tools**: `redis-cli`, Redis configuration file (`redis.conf`), RedisInsight, monitoring tools (e.g., Prometheus).

## 🧠 Memory Management
- **Description**: Redis is an in-memory database, so managing memory usage is critical for performance and stability.
- **Key Commands**:
  - `INFO MEMORY`: Display memory usage statistics.
  - `MEMORY USAGE key`: Estimate memory used by a specific key.
  - `MEMORY STATS`: Provide detailed memory metrics.
- **Configuration** (in `redis.conf`):
  - `maxmemory <bytes>`: Set maximum memory Redis can use (e.g., `maxmemory 1gb`).
  - Default: No limit (uses all available system memory, risking crashes).
- **Example**:
  ```bash
  127.0.0.1:6379> INFO MEMORY
  # Memory
  used_memory:1048576
  used_memory_human:1.00M
  maxmemory:1073741824
  maxmemory_human:1.00G
  ...
  127.0.0.1:6379> MEMORY USAGE user:1:name
  (integer) 64  # Bytes used by key
  ```
- **Tips**:
  - Set `maxmemory` to 50–75% of system RAM to leave room for OS and other processes.
  - Monitor `used_memory` to avoid hitting `maxmemory`.

## 🗑️ Eviction Policies
- **Description**: Define how Redis removes keys when `maxmemory` is reached.
- **Configuration** (in `redis.conf` or via `CONFIG SET`):
  - `maxmemory-policy <policy>`: Choose a policy for key eviction.
  - Common Policies:
    - `noeviction`: Reject writes when memory is full (default).
    - `allkeys-lru`: Evict least recently used keys (good for caching).
    - `allkeys-random`: Evict random keys.
    - `volatile-lru`: Evict LRU keys with an expiration set.
    - `volatile-ttl`: Evict keys with shortest TTL.
    - `volatile-random`: Evict random keys with an expiration set.
  - Example: `maxmemory-policy allkeys-lru`.
- **Dynamic Configuration**:
  ```bash
  127.0.0.1:6379> CONFIG SET maxmemory 100mb
  OK
  127.0.0.1:6379> CONFIG SET maxmemory-policy allkeys-lru
  OK
  ```
- **Example** (Test eviction):
  ```bash
  127.0.0.1:6379> CONFIG SET maxmemory 1mb
  OK
  127.0.0.1:6379> CONFIG SET maxmemory-policy allkeys-lru
  OK
  127.0.0.1:6379> SET key1 "value1"
  OK
  127.0.0.1:6379> SET key2 "value2"
  OK
  # Add large data to trigger eviction
  127.0.0.1:6379> SET bigkey (large string...)
  OK
  127.0.0.1:6379> GET key1
  (nil)  # key1 evicted due to LRU
  ```

## 🔍 Monitoring Performance
- **Description**: Use Redis tools to identify and resolve performance bottlenecks.
- **Key Commands**:
  - `SLOWLOG GET [n]`: List recent slow queries (default threshold: 10ms).
  - `SLOWLOG LEN`: Get number of slow queries logged.
  - `SLOWLOG RESET`: Clear slow log.
  - `MONITOR`: Stream all commands in real-time (use cautiously, high overhead).
- **Configuration**:
  - `slowlog-log-slower-than <microseconds>`: Log queries slower than this (e.g., 10000 for 10ms).
  - `slowlog-max-len <count>`: Maximum number of slow queries to log.
  - Example in `redis.conf`:
    ```
    slowlog-log-slower-than 10000
    slowlog-max-len 128
    ```
- **Example** (Check slow queries):
  ```bash
  127.0.0.1:6379> SLOWLOG GET 2
  1) 1) (integer) 0  # Entry ID
     2) (integer) 1743443580  # Timestamp
     3) (integer) 15000  # Duration (microseconds)
     4) 1) "LRANGE"  # Command
        2) "mylist"
        3) "0"
        4) "-1"
  127.0.0.1:6379> SLOWLOG LEN
  (integer) 1
  ```
- **MONITOR Example** (Run in one terminal, execute commands in another):
  ```bash
  127.0.0.1:6379> MONITOR
  OK
  1743443580.123456 [0 127.0.0.1:12345] "SET" "key1" "value1"
  1743443580.123789 [0 127.0.0.1:12345] "GET" "key1"
  ```

## 🛠️ Optimization Best Practices
- **Use Efficient Data Structures**:
  - Prefer hashes over multiple keys for structured data (e.g., `HSET user:1 name "Alice" age "30"` vs. separate keys).
  - Use bitmaps for binary flags to save memory.
- **Batch Operations**: Use pipelines in client libraries to reduce round-trips:
  ```python
  import redis
  r = redis.Redis(host='localhost', port=6379, db=0)
  with r.pipeline() as pipe:
      pipe.set('key1', 'value1')
      pipe.set('key2', 'value2')
      pipe.execute()
  ```
- **Avoid Costly Commands**: Commands like `KEYS *` or `LRANGE mylist 0 -1` on large datasets are slow; use `SCAN` or paginate ranges.
- **Monitor and Tune**:
  - Regularly check `INFO MEMORY` and `SLOWLOG`.
  - Adjust `maxmemory` and `maxmemory-policy` based on workload.
  - Use RedisInsight for visual monitoring of memory and performance.

## 🛠️ Practice Tips
- Set `maxmemory` and test eviction with `allkeys-lru` by adding data until memory is full.
- Configure `slowlog-log-slower-than` to 1000 (1ms) and generate slow queries (e.g., large `LRANGE`).
- Use `MONITOR` briefly to observe command patterns, then stop to avoid overhead.
- Write a Python script to monitor memory usage:
  ```python
  import redis
  r = redis.Redis(host='localhost', port=6379, db=0)
  print(r.info('memory'))
  ```
- Use RedisInsight to visualize memory usage and slow queries.

## 📚 Resources
- **Official Docs**: [redis.io/docs/management/optimization/](https://redis.io/docs/management/optimization/)
- **Command Reference**: [redis.io/commands/](https://redis.io/commands/) (e.g., `INFO`, `SLOWLOG`, `MEMORY`)
- **Redis University**: [university.redis.com](https://university.redis.com) (free “Redis Administration” course)
- **Redis Labs Blog**: [redis.com/blog](https://redis.com/blog) (search for “performance” or “memory management”)
- **TryRedis**: [try.redis.io](https://try.redis.io) (to test `INFO` and `SLOWLOG`)

## 🔑 Key Takeaways
- Memory management with `maxmemory` and `INFO MEMORY` prevents crashes and optimizes resource usage.
- Eviction policies like `allkeys-lru` handle memory limits gracefully.
- Monitoring tools (`SLOWLOG`, `MONITOR`) identify performance bottlenecks.
- Efficient data structures and batching reduce latency and memory footprint.