# 🌟 Core Data Structures

## 🚀 Overview
- **Goal**: Learn Redis’s core data structures (strings, lists, sets, hashes, sorted sets) and their basic commands for storing and retrieving data.
- **Purpose**: Understand how Redis stores data beyond simple key-value pairs, enabling versatile applications like caching, queues, and rankings.
- **Tool**: Use `redis-cli` or RedisInsight to practice commands.

## 🗝️ Strings
- **Description**: The simplest Redis data type; stores text, numbers, or binary data (up to 512 MB).
- **Common Commands**:
  - `SET key value`: Store a value at a key.
  - `GET key`: Retrieve the value of a key.
  - `INCR key`: Increment a numeric value by 1.
  - `DECR key`: Decrement a numeric value by 1.
  - `APPEND key value`: Append a value to an existing string.
- **Example**:
  ```bash
  127.0.0.1:6379> SET user:1:name "Alice"
  OK
  127.0.0.1:6379> GET user:1:name
  "Alice"
  127.0.0.1:6379> INCR counter
  (integer) 1
  127.0.0.1:6379> INCR counter
  (integer) 2
  ```
- **Use Case**: Store user profiles, counters (e.g., page views), or JSON strings.

## 📋 Lists
- **Description**: Ordered collections of strings, acting like doubly-linked lists (efficient for insertions at both ends).
- **Common Commands**:
  - `LPUSH key value`: Add value to the left (head) of the list.
  - `RPUSH key value`: Add value to the right (tail) of the list.
  - `LPOP key`: Remove and return the leftmost value.
  - `RPOP key`: Remove and return the rightmost value.
  - `LRANGE key start end`: Get a range of elements (e.g., `LRANGE mylist 0 -1` for all).
- **Example**:
  ```bash
  127.0.0.1:6379> LPUSH tasks "Write code"
  (integer) 1
  127.0.0.1:6379> RPUSH tasks "Test app"
  (integer) 2
  127.0.0.1:6379> LRANGE tasks 0 -1
  1) "Write code"
  2) "Test app"
  127.0.0.1:6379> LPOP tasks
  "Write code"
  ```
- **Use Case**: Task queues, recent activity logs, or message queues.

## 🧳 Sets
- **Description**: Unordered collections of unique strings (no duplicates).
- **Common Commands**:
  - `SADD key member`: Add one or more members to a set.
  - `SMEMBERS key`: Get all members of a set.
  - `SINTER key1 key2`: Get intersection of two sets.
  - `SUNION key1 key2`: Get union of two sets.
  - `SREM key member`: Remove a member from a set.
- **Example**:
  ```bash
  127.0.0.1:6379> SADD user:1:tags "developer" "python"
  (integer) 2
  127.0.0.1:6379> SADD user:2:tags "developer" "java"
  (integer) 2
  127.0.0.1:6379> SINTER user:1:tags user:2:tags
  1) "developer"
  127.0.0.1:6379> SMEMBERS user:1:tags
  1) "developer"
  2) "python"
  ```
- **Use Case**: Tagging systems, unique visitor tracking, or friend lists.

## 🗄️ Hashes
- **Description**: Maps between string fields and string values, ideal for structured data (like objects).
- **Common Commands**:
  - `HSET key field value`: Set a field-value pair in a hash.
  - `HGET key field`: Get the value of a field.
  - `HGETALL key`: Get all fields and values in a hash.
  - `HDEL key field`: Delete a field from a hash.
- **Example**:
  ```bash
  127.0.0.1:6379> HSET user:1 name "Bob" age "30"
  (integer) 2
  127.0.0.1:6379> HGET user:1 name
  "Bob"
  127.0.0.1:6379> HGETALL user:1
  1) "name"
  2) "Bob"
  3) "age"
  4) "30"
  ```
- **Use Case**: Store user profiles, product details, or configuration settings.

## 🏅 Sorted Sets
- **Description**: Sets where each member has an associated score (floating-point number), sorted by score.
- **Common Commands**:
  - `ZADD key score member`: Add a member with a score.
  - `ZRANGE key start end [WITHSCORES]`: Get members in rank order.
  - `ZSCORE key member`: Get the score of a member.
  - `ZREM key member`: Remove a member.
- **Example**:
  ```bash
  127.0.0.1:6379> ZADD leaderboard 100 "Alice" 200 "Bob"
  (integer) 2
  127.0.0.1:6379> ZRANGE leaderboard 0 -1 WITHSCORES
  1) "Alice"
  2) "100"
  3) "Bob"
  4) "200"
  127.0.0.1:6379> ZSCORE leaderboard Bob
  "200"
  ```
- **Use Case**: Leaderboards, ranking systems, or priority queues.

## 🛠️ Practice Tips
- Use `redis-cli` to run commands interactively.
- Experiment with RedisInsight to visualize data structures.
- Combine data types (e.g., use a hash for user data and a sorted set for rankings).
- Common pattern: Use `:` in keys (e.g., `user:1:name`) for namespacing.

## 📚 Resources
- **Official Docs**: [redis.io/docs/data-types/](https://redis.io/docs/data-types/)
- **Interactive Tutorial**: [try.redis.io](https://try.redis.io) (browser-based Redis CLI)
- **Redis University**: [university.redis.com](https://university.redis.com) (free “Redis Data Types” course)
- **Command Reference**: [redis.io/commands/](https://redis.io/commands/)

## 🔑 Key Takeaways
- Redis supports five core data structures: strings, lists, sets, hashes, and sorted sets.
- Each structure serves specific use cases (e.g., lists for queues, sorted sets for rankings).
- Commands are intuitive and case-insensitive (e.g., `SET` or `set`).
- Practice with `redis-cli` builds confidence for real-world applications.