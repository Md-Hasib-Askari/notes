# 🌟 Advanced Data Structures

## 🚀 Overview
- **Goal**: Learn Redis’s advanced data structures—Bitmaps, HyperLogLog, and Geospatial Indexes—and their commands for specialized use cases like compact storage, approximate counting, and location-based queries.
- **Purpose**: Enable efficient handling of complex data scenarios, such as tracking user activity, estimating unique counts, or performing geographic searches.
- **Tools**: `redis-cli`, RedisInsight, and a client library (e.g., `redis-py`).

## 🖼️ Bitmaps
- **Description**: Strings treated as arrays of bits, allowing compact storage and manipulation of binary data (up to 512 MB).
- **Use Case**: Track user activity (e.g., daily logins), flags, or binary states efficiently.
- **Common Commands**:
  - `SETBIT key offset value`: Set a bit (0 or 1) at a specific offset.
  - `GETBIT key offset`: Get the bit value at an offset.
  - `BITCOUNT key [start end]`: Count set bits (1s) in a range or entire bitmap.
  - `BITOP operation destkey key [key ...]`: Perform bitwise operations (AND, OR, XOR, NOT).
- **Example** (Track user logins on specific days):
  ```bash
  127.0.0.1:6379> SETBIT user:1:logins 0 1  # User 1 logged in on day 0
  (integer) 0
  127.0.0.1:6379> SETBIT user:1:logins 2 1  # User 1 logged in on day 2
  (integer) 0
  127.0.0.1:6379> GETBIT user:1:logins 0
  (integer) 1
  127.0.0.1:6379> BITCOUNT user:1:logins
  (integer) 2  # User logged in on 2 days
  ```
- **Python with redis-py**:
  ```python
  import redis
  r = redis.Redis(host='localhost', port=6379, db=0)
  r.setbit('user:1:logins', 0, 1)  # Day 0 login
  r.setbit('user:1:logins', 2, 1)  # Day 2 login
  print(r.getbit('user:1:logins', 0))  # Output: 1
  print(r.bitcount('user:1:logins'))  # Output: 2
  ```

## 📊 HyperLogLog
- **Description**: Probabilistic data structure for estimating the cardinality (unique count) of a set with minimal memory (12 KB per HyperLogLog).
- **Use Case**: Count unique visitors, search queries, or events with high scalability.
- **Common Commands**:
  - `PFADD key element [element ...]`: Add elements to a HyperLogLog.
  - `PFCOUNT key [key ...]`: Estimate the number of unique elements.
  - `PFMERGE destkey sourcekey [sourcekey ...]`: Merge multiple HyperLogLogs.
- **Example** (Track unique website visitors):
  ```bash
  127.0.0.1:6379> PFADD visitors:2025-05-30 "user1" "user2" "user1"
  (integer) 1
  127.0.0.1:6379> PFCOUNT visitors:2025-05-30
  (integer) 2  # Approx. 2 unique visitors
  127.0.0.1:6379> PFADD visitors:2025-05-31 "user2" "user3"
  (integer) 1
  127.0.0.1:6379> PFMERGE all_visitors visitors:2025-05-30 visitors:2025-05-31
  OK
  127.0.0.1:6379> PFCOUNT all_visitors
  (integer) 3  # Approx. 3 unique visitors
  ```
- **Python with redis-py**:
  ```python
  r.pfadd('visitors:2025-05-30', 'user1', 'user2', 'user1')
  print(r.pfcount('visitors:2025-05-30'))  # Output: ~2
  ```
- **Note**: HyperLogLog is probabilistic with ~0.81% error rate but extremely memory-efficient.

## 🌍 Geospatial Indexes
- **Description**: Sorted sets storing longitude/latitude pairs for location-based queries (e.g., find nearby points).
- **Use Case**: Location-based services, such as finding nearby stores or users.
- **Common Commands**:
  - `GEOADD key longitude latitude member`: Add a location with coordinates.
  - `GEORADIUS key longitude latitude radius unit [WITHCOORD] [WITHDIST]`: Find members within a radius (unit: `m`, `km`, `mi`, `ft`).
  - `GEODIST key member1 member2 [unit]`: Calculate distance between two members.
- **Example** (Find nearby coffee shops):
  ```bash
  127.0.0.1:6379> GEOADD shops -122.4194 37.7749 "San Francisco"
  (integer) 1
  127.0.0.1:6379> GEOADD shops -122.4064 37.7850 "Oakland"
  (integer) 1
  127.0.0.1:6379> GEORADIUS shops -122.4 37.78 10 mi
  1) "San Francisco"
  2) "Oakland"
  127.0.0.1:6379> GEODIST shops San Francisco Oakland mi
  "8.5062"  # Distance in miles
  ```
- **Python with redis-py**:
  ```python
  r.geoadd('shops', (-122.4194, 37.7749, 'San Francisco'))
  r.geoadd('shops', (-122.4064, 37.7850, 'Oakland'))
  nearby = r.georadius('shops', -122.4, 37.78, 10, 'mi')
  print(nearby)  # Output: [b'San Francisco', b'Oakland']
  print(r.geodist('shops', 'San Francisco', 'Oakland', 'mi'))  # Output: 8.5062
  ```

## 🛠️ Practice Tips
- Use `redis-cli` to experiment with commands interactively.
- Visualize data in RedisInsight (e.g., check bitmap bits or geospatial coordinates).
- Write Python scripts to automate tasks (e.g., track logins with bitmaps).
- Combine data structures (e.g., use HyperLogLog for unique counts and hashes for user details).
- Test edge cases (e.g., empty HyperLogLog, invalid coordinates).

## 📚 Resources
- **Official Docs**: [redis.io/docs/data-types/](https://redis.io/docs/data-types/) (Bitmaps, HyperLogLog, Geospatial)
- **Command Reference**: [redis.io/commands/](https://redis.io/commands/) (e.g., `SETBIT`, `PFADD`, `GEOADD`)
- **Redis University**: [university.redis.com](https://university.redis.com) (free “Redis Data Types” course)
- **TryRedis**: [try.redis.io](https://try.redis.io) (browser-based practice)
- **Redis Labs Blog**: [redis.com/blog](https://redis.com/blog) (search for “HyperLogLog” or “Geospatial”)

## 🔑 Key Takeaways
- Bitmaps enable compact storage for binary data (e.g., user activity tracking).
- HyperLogLog provides memory-efficient, approximate counting for large datasets.
- Geospatial Indexes support location-based queries for proximity searches.
- These structures expand Redis’s versatility for advanced applications.