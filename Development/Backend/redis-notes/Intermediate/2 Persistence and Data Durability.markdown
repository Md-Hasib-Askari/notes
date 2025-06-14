# 🌟 Persistence and Data Durability

## 🚀 Overview
- **Goal**: Understand Redis’s persistence options—RDB (snapshotting) and AOF (append-only file)—and how to configure them to ensure data durability in case of crashes or restarts.
- **Purpose**: Enable reliable data storage for applications where data loss is unacceptable, balancing performance and durability.
- **Tools**: `redis-cli`, Redis configuration file (`redis.conf`), RedisInsight for monitoring.

## 🛡️ Why Persistence?
- Redis is an in-memory database, meaning data is lost on server restart or crash unless persisted to disk.
- Persistence ensures data can be recovered, making Redis suitable for more than just caching (e.g., session storage, analytics).

## 📸 RDB (Snapshotting)
- **Description**: Periodically saves a snapshot of the entire dataset to disk as a compact binary file (`.rdb`).
- **How It Works**: Saves data at specified intervals based on changes (e.g., “save after 1000 writes in 60 seconds”).
- **Pros**:
  - Fast recovery (restores entire dataset from a single file).
  - Minimal disk space usage due to compression.
- **Cons**:
  - Potential data loss (data since last snapshot is lost on crash).
  - Snapshotting can cause brief performance hiccups on large datasets.
- **Configuration** (in `redis.conf`):
  - `save <seconds> <changes>`: Triggers snapshot after `<changes>` writes in `<seconds>`.
  - Example: `save 60 1000` (save if 1000 keys changed in 60 seconds).
  - Default:
    ```
    save 900 1    # After 15 min, if 1 key changed
    save 300 10   # After 5 min, if 10 keys changed
    save 60 10000 # After 1 min, if 10,000 keys changed
    ```
  - Disable RDB: `save ""`.
  - File location: Set by `dir` and `dbfilename` (e.g., `dir /var/lib/redis/`, `dbfilename dump.rdb`).
- **Commands**:
  - `SAVE`: Create a snapshot synchronously (blocks server).
  - `BGSAVE`: Create a snapshot asynchronously (non-blocking).
  - `LASTSAVE`: Get timestamp of last successful save.
- **Example**:
  ```bash
  127.0.0.1:6379> SET user:1:name "Alice"
  OK
  127.0.0.1:6379> BGSAVE
  Background saving started
  127.0.0.1:6379> LASTSAVE
  (integer) 1743443580  # Unix timestamp
  ```

## 📜 AOF (Append-Only File)
- **Description**: Logs every write operation (e.g., `SET`, `LPUSH`) to a file, allowing full data reconstruction.
- **How It Works**: Appends commands to a file (`.aof`), which can be replayed on startup to rebuild the dataset.
- **Pros**:
  - Higher durability (less data loss, especially with frequent syncing).
  - Easier to understand (human-readable log).
- **Cons**:
  - Larger file size than RDB.
  - Slower recovery (replays all commands).
- **Configuration** (in `redis.conf`):
  - Enable AOF: `appendonly yes`.
  - File location: Set by `dir` and `appendfilename` (e.g., `appendfilename "appendonly.aof"`).
  - Sync options (`appendfsync`):
    - `always`: Sync every write (most durable, slowest).
    - `everysec`: Sync every second (good balance, default).
    - `no`: Let OS decide (fastest, least durable).
  - Example:
    ```
    appendonly yes
    appendfsync everysec
    ```
- **AOF Rewrite**:
  - Over time, AOF files grow large; rewriting compacts them.
  - `BGREWRITEAOF`: Rewrite AOF file in the background.
- **Example**:
  ```bash
  127.0.0.1:6379> CONFIG SET appendonly yes
  OK
  127.0.0.1:6379> SET user:1:name "Bob"
  OK
  127.0.0.1:6379> BGREWRITEAOF
  Background append only file rewriting started
  ```

## ⚖️ RDB vs. AOF
- **RDB**: Best for backups, faster recovery, less disk usage; risks losing recent data.
- **AOF**: Best for durability, minimal data loss; larger files, slower recovery.
- **Hybrid Approach**: Use both (AOF for durability, RDB for faster recovery).
- **Common Setup**: Enable AOF with `appendfsync everysec` and keep default RDB settings.

## 🛠️ Configuring Persistence
- **Edit redis.conf**:
  - Find file (e.g., `/etc/redis/redis.conf` or in Redis source directory).
  - Example configuration:
    ```
    dir /var/lib/redis/
    dbfilename dump.rdb
    save 60 1000
    appendonly yes
    appendfilename "appendonly.aof"
    appendfsync everysec
    ```
- **Apply Changes**:
  - Restart Redis: `redis-server /path/to/redis.conf`.
  - Or dynamically: `CONFIG SET appendonly yes`, `CONFIG SET appendfsync everysec`.
- **Verify**:
  - Check persistence status: `INFO PERSISTENCE`.
  - Example:
    ```bash
    127.0.0.1:6379> INFO PERSISTENCE
    # Persistence
    rdb_changes_since_last_save:0
    aof_enabled:1
    aof_current_size:1024
    ```

## 🔍 Testing Data Recovery
- **Steps**:
  1. Set some data: `SET key1 "value1"`.
  2. Trigger a snapshot: `BGSAVE` or enable AOF.
  3. Stop Redis: `redis-cli SHUTDOWN`.
  4. Delete in-memory data (e.g., restart server).
  5. Start Redis: `redis-server /path/to/redis.conf`.
  6. Check data: `GET key1` (should return `value1`).
- **Example**:
  ```bash
  127.0.0.1:6379> SET user:1:name "Alice"
  OK
  127.0.0.1:6379> BGSAVE
  Background saving started
  127.0.0.1:6379> SHUTDOWN
  # Restart Redis
  $ redis-server /path/to/redis.conf
  127.0.0.1:6379> GET user:1:name
  "Alice"
  ```

## 🛠️ Practice Tips
- Edit `redis.conf` to enable AOF and tweak RDB settings.
- Use `INFO PERSISTENCE` to monitor snapshot and AOF status.
- Test data recovery by simulating a crash (stop Redis, restart, check data).
- Use RedisInsight to inspect persisted data after recovery.
- Experiment with `appendfsync` options (`always`, `everysec`, `no`) to observe performance.

## 📚 Resources
- **Official Docs**: [redis.io/docs/management/persistence/](https://redis.io/docs/management/persistence/)
- **Command Reference**: [redis.io/commands/](https://redis.io/commands/) (e.g., `SAVE`, `BGSAVE`, `BGREWRITEAOF`)
- **Redis University**: [university.redis.com](https://university.redis.com) (free “Redis Administration” course)
- **Redis Labs Blog**: [redis.com/blog](https://redis.com/blog) (search for “persistence”)
- **TryRedis**: [try.redis.io](https://try.redis.io) (to test commands like `CONFIG SET`)

## 🔑 Key Takeaways
- RDB (snapshotting) is compact and fast for backups but risks losing recent data.
- AOF (append-only file) ensures higher durability with minimal data loss.
- Configure persistence via `redis.conf` or `CONFIG SET` for dynamic changes.
- Test recovery to ensure data durability in production scenarios.