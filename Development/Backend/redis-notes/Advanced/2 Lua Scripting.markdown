# 🌟 Lua Scripting

## 🚀 Overview
- **Goal**: Learn to write and execute Lua scripts in Redis to perform atomic, custom operations, reducing round-trips and enhancing performance.
- **Purpose**: Enable complex logic (e.g., conditional updates, rate limiting) in a single atomic operation, leveraging Redis’s built-in Lua interpreter.
- **Tools**: `redis-cli`, RedisInsight, a client library (e.g., `redis-py`), and basic Lua knowledge.

## 🧠 Why Lua Scripting?
- **Atomicity**: Redis executes Lua scripts as a single, uninterruptible operation, ensuring data consistency.
- **Performance**: Reduces network round-trips by combining multiple commands into one script.
- **Flexibility**: Allows custom logic not possible with standard Redis commands.
- **Use Case**: Implement rate limiting, conditional updates, or complex data manipulations.

## 🛠️ Core Lua Scripting Commands
- **EVAL script numkeys key [key ...] arg [arg ...]**: Execute a Lua script, specifying the number of keys and arguments.
  - `script`: Lua code as a string.
  - `numkeys`: Number of keys accessed in the script.
  - `key`: Keys used in the script (accessible in Lua as `KEYS[1]`, `KEYS[2]`, etc.).
  - `arg`: Additional arguments (accessible in Lua as `ARGV[1]`, `ARGV[2]`, etc.).
- **EVALSHA sha1 numkeys key [key ...] arg [arg ...]**: Execute a cached script by its SHA1 hash.
- **SCRIPT LOAD script**: Load a script into Redis, returning its SHA1 hash.
- **SCRIPT EXISTS sha1 [sha1 ...]**: Check if a script is cached.
- **SCRIPT FLUSH**: Clear all cached scripts.

## 📜 Writing Lua Scripts
- **Basics**:
  - Redis uses Lua 5.1, embedded in the server.
  - Use `redis.call(command, key, arg, ...)` to execute Redis commands (e.g., `redis.call('SET', KEYS[1], ARGV[1])`).
  - Return values from Lua are converted to Redis protocol responses.
- **Example** (Simple set and get):
  ```lua
  redis.call('SET', KEYS[1], ARGV[1])
  return redis.call('GET', KEYS[1])
  ```
  Execute in `redis-cli`:
  ```bash
  127.0.0.1:6379> EVAL "redis.call('SET', KEYS[1], ARGV[1]) return redis.call('GET', KEYS[1])" 1 user:1 Alice
  "Alice"
  ```
- **Caching Script**:
  ```bash
  127.0.0.1:6379> SCRIPT LOAD "redis.call('SET', KEYS[1], ARGV[1]) return redis.call('GET', KEYS[1])"
  "f3c4e7d2a8b9c1d0e2f3a4b5c6d7e8f9a0b1c2d3"
  127.0.0.1:6379> EVALSHA f3c4e7d2a8b9c1d0e2f3a4b5c6d7e8f9a0b1c2d3 1 user:1 Bob
  "Bob"
  ```

## 🛡️ Rate Limiting with Lua
- **Objective**: Implement a rate limiter that checks and increments a counter atomically.
- **Script** (`ratelimit.lua`):
  ```lua
  local key = KEYS[1]
  local limit = tonumber(ARGV[1])
  local window = tonumber(ARGV[2])
  local count = redis.call('INCR', key)
  if count == 1 then
      redis.call('EXPIRE', key, window)
  end
  if count > limit then
      return 0  -- Rate limit exceeded
  end
  return 1  -- Allowed
  ```
- **Execute**:
  ```bash
  127.0.0.1:6379> EVAL "local key = KEYS[1] local limit = tonumber(ARGV[1]) local window = tonumber(ARGV[2]) local count = redis.call('INCR', key) if count == 1 then redis.call('EXPIRE', key, window) end if count > limit then return 0 end return 1" 1 ratelimit:user1 5 60
  (integer) 1
  # Run multiple times; after 5th call:
  127.0.0.1:6379> EVAL ... 1 ratelimit:user1 5 60
  (integer) 0
  ```
- **Python with redis-py**:
  ```python
  import redis

  r = redis.Redis(host='localhost', port=6379, db=0)
  script = """
  local key = KEYS[1]
  local limit = tonumber(ARGV[1])
  local window = tonumber(ARGV[2])
  local count = redis.call('INCR', key)
  if count == 1 then
      redis.call('EXPIRE', key, window)
  end
  if count > limit then
      return 0
  end
  return 1
  """
  sha = r.script_load(script)
  result = r.evalsha(sha, 1, "ratelimit:user1", 5, 60)
  print("Allowed" if result == 1 else "Rate limit exceeded")  # Output: Allowed (first 5 calls)
  ```

## 🛠️ Practice Tips
- Write simple Lua scripts to combine commands (e.g., set and increment).
- Cache scripts with `SCRIPT LOAD` and use `EVALSHA` for efficiency.
- Test scripts in `redis-cli` before integrating with Python.
- Use RedisInsight to debug script effects (e.g., check keys modified).
- Experiment with error handling in Lua (e.g., `redis.pcall` for safe calls).

## 📚 Resources
- **Official Docs**: [redis.io/docs/interact/programmability/lua-scripting/](https://redis.io/docs/interact/programmability/lua-scripting/)
- **Command Reference**: [redis.io/commands/](https://redis.io/commands/) (e.g., `EVAL`, `EVALSHA`)
- **Redis University**: [university.redis.com](https://university.redis.com) (free “Redis for Developers” course)
- **Redis Labs Blog**: [redis.com/blog](https://redis.com/blog) (search for “Lua scripting”)
- **Lua Tutorial**: [lua.org/manual/5.1/](https://www.lua.org/manual/5.1/) (basic Lua syntax)

## 🔑 Key Takeaways
- Lua scripts enable atomic, custom operations in Redis, reducing network overhead.
- Use `EVAL` for ad-hoc scripts and `EVALSHA` for cached scripts.
- Scripts are ideal for complex tasks like rate limiting or conditional updates.
- Practice with simple scripts before tackling production-grade logic.