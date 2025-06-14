# 🌈 Basic Commands and Operations

## 🚀 Overview
- **Goal**: Learn essential Redis commands for managing keys, handling transactions, and exploring basic publish/subscribe (pub/sub) messaging.
- **Purpose**: Build a foundation for manipulating data, ensuring data expiration, and enabling simple messaging patterns.
- **Tool**: Use `redis-cli` or RedisInsight to practice commands.

## 🔑 Key Management Commands
- **Description**: Commands to create, delete, check, and manage the lifecycle of keys in Redis.
- **Common Commands**:
  - `DEL key [key ...]`: Delete one or more keys.
  - `EXISTS key [key ...]`: Check if one or more keys exist (returns count of existing keys).
  - `EXPIRE key seconds`: Set a key’s expiration time in seconds.
  - `TTL key`: Get the remaining time to live (TTL) of a key in seconds (-1 for no expiration, -2 if key doesn’t exist).
- **Example**:
  ```bash
  127.0.0.1:6379> SET session:123 "user-data"
  OK
  127.0.0.1:6379> EXISTS session:123
  (integer) 1
  127.0.0.1:6379> EXPIRE session:123 60
  (integer) 1
  127.0.0.1:6379> TTL session:123
  (integer) 58
  127.0.0.1:6379> DEL session:123
  (integer) 1
  127.0.0.1:6379> EXISTS session:123
  (integer) 0
  ```
- **Use Case**: Manage temporary data (e.g., user sessions) or clean up unused keys.

## 🔗 Transactions
- **Description**: Group multiple commands to execute atomically, ensuring no interference from other clients.
- **Common Commands**:
  - `MULTI`: Start a transaction block.
  - `EXEC`: Execute all commands in the transaction block.
  - `DISCARD`: Cancel the transaction and discard queued commands.
- **How It Works**: Commands in a `MULTI` block are queued and executed as a single atomic operation when `EXEC` is called.
- **Example**:
  ```bash
  127.0.0.1:6379> MULTI
  OK
  127.0.0.1:6379> SET user:1:name "Alice"
  QUEUED
  127.0.0.1:6379> INCR user:1:visits
  QUEUED
  127.0.0.1:6379> EXEC
  1) OK
  2) (integer) 1
  127.0.0.1:6379> GET user:1:name
  "Alice"
  127.0.0.1:6379> GET user:1:visits
  "1"
  ```
- **Use Case**: Update multiple keys atomically (e.g., updating user data and visit count together).

## 📬 Basic Pub/Sub
- **Description**: Publish/subscribe messaging for real-time communication, where clients subscribe to channels and receive messages published to them.
- **Common Commands**:
  - `PUBLISH channel message`: Send a message to a channel.
  - `SUBSCRIBE channel [channel ...]`: Subscribe to one or more channels to receive messages.
  - `UNSUBSCRIBE [channel ...]`: Unsubscribe from channels.
- **How It Works**: Subscribers listen to channels, and publishers send messages without knowing who’s listening.
- **Example** (Open two `redis-cli` instances):
  - **Terminal 1 (Subscriber)**:
    ```bash
    127.0.0.1:6379> SUBSCRIBE news
    Reading messages... (press Ctrl-C to quit)
    1) "subscribe"
    2) "news"
    3) (integer) 1
    ```
  - **Terminal 2 (Publisher)**:
    ```bash
    127.0.0.1:6379> PUBLISH news "Breaking: Redis 7.2 released!"
    (integer) 1
    ```
  - **Terminal 1 Output**:
    ```bash
    1) "message"
    2) "news"
    3) "Breaking: Redis 7.2 released!"
    ```
- **Use Case**: Real-time notifications (e.g., chat messages, live updates).

## 🛠️ Practice Tips
- Use `redis-cli` to run commands interactively and observe outputs.
- Combine key management with data structures (e.g., set a hash and add an expiration).
- Test transactions with multiple commands to ensure atomicity.
- Experiment with pub/sub in two terminal windows to simulate messaging.
- Use RedisInsight to monitor keys and their TTLs visually.

## 📚 Resources
- **Official Docs**: [redis.io/docs/management/](https://redis.io/docs/management/) (Key Management), [redis.io/docs/interact/transactions/](https://redis.io/docs/interact/transactions/), [redis.io/docs/interact/pubsub/](https://redis.io/docs/interact/pubsub/)
- **Interactive Tutorial**: [try.redis.io](https://try.redis.io) (practice commands in browser)
- **Redis University**: [university.redis.com](https://university.redis.com) (free “Redis for Beginners” course)
- **Command Reference**: [redis.io/commands/](https://redis.io/commands/)

## 🔑 Key Takeaways
- Key management commands (`DEL`, `EXISTS`, `EXPIRE`, `TTL`) control the lifecycle of data.
- Transactions (`MULTI`, `EXEC`, `DISCARD`) ensure atomic operations for multiple commands.
- Pub/sub (`PUBLISH`, `SUBSCRIBE`) enables simple real-time messaging.
- These commands are foundational for building Redis-based applications like session stores or notification systems.