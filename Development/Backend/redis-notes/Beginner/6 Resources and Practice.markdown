# 🌈 Resources and Practice

## 🚀 Overview
- **Goal**: Discover high-quality resources to deepen your Redis knowledge and apply your skills through hands-on practice, including a milestone project.
- **Purpose**: Reinforce beginner-level concepts (data structures, commands, programming) by exploring tutorials, courses, and building a practical application.
- **Tools**: `redis-cli`, RedisInsight, a programming language (e.g., Python), and online resources.

## 📚 Key Resources
- **Official Redis Website** ([redis.io](https://redis.io)):
  - Comprehensive documentation covering installation, commands, and data types.
  - Sections: “Getting Started,” “Data Types,” “Commands.”
  - Use for reference when exploring new commands or configurations.
- **Redis University** ([university.redis.com](https://university.redis.com)):
  - Free courses like “RU101: Introduction to Redis” and “Redis for Developers.”
  - Interactive lessons with quizzes and hands-on labs.
  - Ideal for structured learning (1–2 hours per course).
- **TryRedis** ([try.redis.io](https://try.redis.io)):
  - Browser-based Redis CLI for practicing commands without local setup.
  - Great for quick experimentation with data structures and commands.
- **Redis Labs Blog** ([redis.com/blog](https://redis.com/blog)):
  - Articles on use cases, best practices, and new features.
  - Example: Learn how Redis is used for caching or real-time analytics.
- **Command Reference** ([redis.io/commands/](https://redis.io/commands/)):
  - Complete list of Redis commands with syntax and examples.
  - Use to look up commands like `SET`, `LPUSH`, or `PUBLISH`.
- **Community**:
  - **GitHub**: [github.com/redis/redis](https://github.com/redis/redis) for source code and issues.
  - **Stack Overflow**: Search for “redis” tag to find solutions to common problems.
  - **Redis Discord**: Join via [redis.com/community](https://redis.com/community) for discussions.
- **Video Tutorials**:
  - YouTube: Search for “Redis tutorial” (e.g., Redis Labs channel or “Tech With Tim”).
  - Focus on beginner-friendly videos (~10–20 minutes) covering basics.

## 🛠️ Practice Strategies
- **Use redis-cli**: Run commands daily to build muscle memory (e.g., `SET`, `LPUSH`, `HSET`).
- **Experiment with RedisInsight**: Visualize data structures and monitor key changes.
- **Write Code**: Use a client library (e.g., `redis-py`) to automate tasks like storing user data or publishing messages.
- **Mini-Projects**: Build small applications to apply concepts (e.g., a counter, task queue, or notification system).
- **Combine Concepts**: Use multiple data structures (e.g., hashes for user data, sorted sets for rankings) in one project.

## 🏆 Milestone Project: To-Do List Application
- **Objective**: Build a simple to-do list application using Redis lists to store and manage tasks.
- **Requirements**:
  - Store tasks in a Redis list (e.g., `tasks:user:1`).
  - Support adding tasks (to head or tail), viewing all tasks, and removing completed tasks.
  - Use a programming language (Python recommended) with a Redis client library.
- **Steps**:
  1. **Setup**: Ensure Redis is running (`redis-cli ping` returns `PONG`).
  2. **Python Script** (save as `todo.py`):
     ```python
     import redis

     # Connect to Redis
     r = redis.Redis(host='localhost', port=6379, db=0)

     def add_task(user_id, task):
         r.rpush(f'tasks:user:{user_id}', task)
         print(f"Added task: {task}")

     def view_tasks(user_id):
         tasks = r.lrange(f'tasks:user:{user_id}', 0, -1)
         print(f"Tasks for user {user_id}:")
         for i, task in enumerate(tasks, 1):
             print(f"{i}. {task.decode()}")

     def complete_task(user_id):
         task = r.lpop(f'tasks:user:{user_id}')
         if task:
             print(f"Completed task: {task.decode()}")
         else:
             print("No tasks to complete")

     # Example usage
     user_id = 1
     add_task(user_id, "Write code")
     add_task(user_id, "Test app")
     view_tasks(user_id)
     complete_task(user_id)
     view_tasks(user_id)
     ```
  3. **Run**: Execute the script (`python todo.py`) and observe the output:
     ```
     Added task: Write code
     Added task: Test app
     Tasks for user 1:
     1. Write code
     2. Test app
     Completed task: Write code
     Tasks for user 1:
     1. Test app
     ```
  4. **Verify**: Use RedisInsight or `redis-cli` to check the list (`LRANGE tasks:user:1 0 -1`).
- **Extensions** (Optional):
  - Add task priorities using a sorted set.
  - Set an expiration on tasks with `EXPIRE` (e.g., tasks expire after 24 hours).
  - Publish a notification with `PUBLISH` when a task is completed.
- **Learning Outcomes**:
  - Practice list commands (`LPUSH`, `RPUSH`, `LPOP`, `LRANGE`).
  - Integrate Redis with Python using `redis-py`.
  - Understand key naming conventions (e.g., `tasks:user:1`).

## 📊 Practice Tips
- Run the to-do list script and modify it to suit your needs (e.g., add a `delete_task` function).
- Use `redis-cli` to inspect the list data (`LRANGE`, `LLEN`).
- Experiment with errors (e.g., pop from an empty list) and handle them in your code.
- Share your script output or issues for feedback.

## 📚 Resources (Repeated for Convenience)
- **Official Docs**: [redis.io/docs/](https://redis.io/docs/)
- **Redis University**: [university.redis.com](https://university.redis.com)
- **TryRedis**: [try.redis.io](https://try.redis.io)
- **Redis Labs Blog**: [redis.com/blog](https://redis.com/blog)
- **Command Reference**: [redis.io/commands/](https://redis.io/commands/)

## 🔑 Key Takeaways
- Quality resources (Redis docs, Redis University, TryRedis) accelerate learning.
- Hands-on practice with commands and coding is essential for mastery.
- The to-do list project combines lists, programming, and Redis commands for practical experience.
- Regular practice builds confidence for more complex applications.