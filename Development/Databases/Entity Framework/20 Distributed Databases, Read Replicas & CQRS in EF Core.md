## 🧩 Step 20: Distributed Databases, Read Replicas & CQRS in EF Core

### 📘 Overview

As systems grow, a single EF Core context connected to one database isn’t enough.
You’ll need to:

* **Distribute load** between read and write databases
* **Scale horizontally** across regions
* **Separate reads and writes** (CQRS)
* **Support eventual consistency**

Let’s break down how to design, configure, and deploy these patterns using EF Core.

---

## ⚙️ 20.1 Read/Write Separation (Read Replicas)

Many relational databases (e.g., SQL Server, PostgreSQL, MySQL) allow **replication** — one master (for writes) and multiple read-only replicas.

EF Core doesn’t have built-in replica routing, but you can implement it easily.

---

### 💡 Pattern: Read/Write Context Separation

You use **two DbContexts**:

* One for **writes** (points to the primary database)
* One for **reads** (points to replicas)

**Example:**

```csharp
public class WriteDbContext : DbContext
{
    public WriteDbContext(DbContextOptions<WriteDbContext> options)
        : base(options) { }
}

public class ReadDbContext : DbContext
{
    public ReadDbContext(DbContextOptions<ReadDbContext> options)
        : base(options) { }
}
```

**Program.cs:**

```csharp
builder.Services.AddDbContext<WriteDbContext>(
    o => o.UseSqlServer("Server=primary-db;Database=Prod;Trusted_Connection=True;"));

builder.Services.AddDbContext<ReadDbContext>(
    o => o.UseSqlServer("Server=read-replica;Database=Prod;Trusted_Connection=True;"));
```

**Usage:**

```csharp
// Write operations
await using var writer = scope.ServiceProvider.GetRequiredService<WriteDbContext>();
writer.Users.Add(new User { Name = "Hasib" });
await writer.SaveChangesAsync();

// Read operations
await using var reader = scope.ServiceProvider.GetRequiredService<ReadDbContext>();
var users = await reader.Users.AsNoTracking().ToListAsync();
```

✅ Writes go to master; reads go to replica — zero contention, higher throughput.

---

### ⚡ Tip:

If your ORM queries are *purely read-only*, always use **AsNoTracking()** to skip tracking overhead.

---

## 🧩 20.2 Automatic Routing (Smart Context Factory)

You can dynamically switch between contexts based on the operation type.

```csharp
public class DbContextRouter
{
    private readonly WriteDbContext _writeDb;
    private readonly ReadDbContext _readDb;

    public DbContextRouter(WriteDbContext writeDb, ReadDbContext readDb)
    {
        _writeDb = writeDb;
        _readDb = readDb;
    }

    public DbContext Get(bool readOnly = false) => readOnly ? _readDb : _writeDb;
}
```

Usage:

```csharp
var db = router.Get(readOnly: true);
var data = await db.Users.ToListAsync();
```

✅ Clean separation of responsibilities.

---

## 🧩 20.3 Distributed Transactions (Caution!)

In microservices or distributed systems, you can’t always rely on a single database transaction across services.

Instead of **2PC (Two-Phase Commit)** (which is slow and complex), you typically use:

* **Eventual consistency**
* **Outbox pattern**

---

### 💡 Outbox Pattern Example

Store events in a local table alongside your EF transaction, then publish asynchronously.

```csharp
public class OutboxEvent
{
    public int Id { get; set; }
    public string EventType { get; set; }
    public string Payload { get; set; }
    public bool Processed { get; set; }
}
```

**When saving data:**

```csharp
await db.Users.AddAsync(new User { Name = "Hasib" });
await db.OutboxEvents.AddAsync(new OutboxEvent
{
    EventType = "UserCreated",
    Payload = JsonSerializer.Serialize(new { Name = "Hasib" })
});
await db.SaveChangesAsync();
```

Then a background service periodically reads unprocessed outbox events and publishes them to message brokers (Kafka, RabbitMQ, etc.).

✅ Ensures atomic writes without distributed transactions.

---

## 🧩 20.4 CQRS (Command–Query Responsibility Segregation)

### 🧠 Concept

Split application logic into two clear paths:

| Type        | Purpose                               | Example           |
| ----------- | ------------------------------------- | ----------------- |
| **Command** | Change state (insert, update, delete) | CreateUserCommand |
| **Query**   | Read data                             | GetUserByIdQuery  |

Each can have **separate models, DbContexts, and databases** for scaling and security.

---

### 💡 Example Implementation

**Command Model:**

```csharp
public class CommandDbContext : DbContext
{
    public DbSet<User> Users { get; set; }
}
```

**Query Model:**

```csharp
public class QueryDbContext : DbContext
{
    public DbSet<UserReadModel> Users { get; set; }
}
```

**Command Handler:**

```csharp
public class CreateUserCommandHandler
{
    private readonly CommandDbContext _db;
    public CreateUserCommandHandler(CommandDbContext db) => _db = db;

    public async Task Handle(string name)
    {
        _db.Users.Add(new User { Name = name });
        await _db.SaveChangesAsync();
    }
}
```

**Query Handler:**

```csharp
public class GetUsersQueryHandler
{
    private readonly QueryDbContext _db;
    public GetUsersQueryHandler(QueryDbContext db) => _db = db;

    public async Task<List<UserReadModel>> Handle()
        => await _db.Users.AsNoTracking().ToListAsync();
}
```

✅ You can deploy read models separately on read replicas or even cache them with Redis for instant reads.

---

## 🧩 20.5 Read Model Synchronization

After every command, publish a message (e.g., “UserCreated”) that updates the read database asynchronously.

* Commands → go to **primary DB**
* Queries → use **replica or denormalized read DB**

Example sync flow:

```
[Command API] → [Primary DB + Outbox] → [Message Broker] → [Read API] → [Read DB]
```

✅ Enables horizontal scaling and microservice separation.

---

## 🧩 20.6 EF Core in Distributed Microservices

When EF Core is used in multiple services:

* Each service owns its own **bounded context** (schema + model).
* No cross-service database sharing.
* Use events for communication, not shared tables.

### Example:

* **User Service** → manages users (UserDbContext)
* **Order Service** → manages orders (OrderDbContext)
* **Billing Service** → manages payments (BillingDbContext)

If the order service needs user data → it subscribes to a “UserCreated” event, not the user DB.

✅ Keeps EF Core contexts lightweight, decoupled, and scalable.

---

## 🧩 20.7 Multi-Region Deployment

When your application runs in multiple regions (e.g., Asia, Europe, US), you can:

* Deploy regional replicas of databases (read-only)
* Use EF Core read contexts pointing to nearest replica
* Sync changes asynchronously across regions

### Example Strategy:

* Writes → Global primary (Azure SQL, AWS RDS Primary)
* Reads → Regional replicas (low latency)
* Conflict handling → Last-write-wins or version-based merge

---

## 🧠 Exercises

1. **Read/Write Split**

   * Implement separate contexts for read and write operations.
   * Confirm writes only affect the primary DB.

2. **Outbox Pattern**

   * Extend your DbContext to store outbox events.
   * Simulate event publishing and mark processed events.

3. **CQRS Implementation**

   * Create `CommandDbContext` and `QueryDbContext` for a blog system.
   * Separate models for write (Blog entity) and read (BlogSummary DTO).

4. **Replica Simulation**

   * Set up two SQLite databases (primary + replica).
   * Manually replicate changes after writes and query from replica.

5. **Distributed Consistency**

   * Simulate message-based synchronization between write DB and read DB using background tasks.

---

## ✅ Summary

| Concept                     | Description                           | Use Case                    |
| --------------------------- | ------------------------------------- | --------------------------- |
| **Read Replica**            | Offload read queries to secondary DB  | Scaling reads               |
| **Write DB**                | Handles all inserts/updates           | Strong consistency          |
| **CQRS**                    | Separate read & write models          | Scalable, decoupled systems |
| **Outbox Pattern**          | Reliable event publishing             | Async consistency           |
| **Distributed Contexts**    | Isolated EF Core models per service   | Microservice design         |
| **Multi-Region Deployment** | Global replicas for latency reduction | Cloud-scale systems         |

---

**Key Takeaways:**

* Split reads/writes for instant scalability.
* Implement CQRS to decouple business logic and improve performance.
* Use the Outbox pattern to ensure atomic consistency across services.
* Read replicas reduce latency and offload primary DB load.
* In distributed EF systems — *design for eventual consistency, not synchronous coupling.*
