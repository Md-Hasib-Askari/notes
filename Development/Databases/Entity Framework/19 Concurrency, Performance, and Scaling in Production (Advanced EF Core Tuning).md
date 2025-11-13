## 🧩 Step 19: Concurrency, Performance, and Scaling in Production (Advanced EF Core Tuning)

### 📘 Overview

Performance and scaling in EF Core involve optimizing:

* **Query execution & caching**
* **Context lifetime management**
* **Connection pooling**
* **Concurrency handling (async, parallelism, locking)**
* **Data loading and projection**
* **Profiling and metrics**

The goal: **high throughput with minimal latency**, even under multi-user, multi-threaded workloads.

---

## ⚙️ 19.1 DbContext Lifetime Management

### 💡 Key Rule:

> “DbContext is **not thread-safe**. Use one instance per unit of work.”

| Lifetime      | Scope               | When to Use                 |
| ------------- | ------------------- | --------------------------- |
| **Scoped**    | One per web request | ✅ Default for web APIs      |
| **Transient** | New per operation   | For background tasks        |
| **Singleton** | ❌ Never             | Causes thread-safety issues |

Example setup in ASP.NET Core:

```csharp
builder.Services.AddDbContext<AppDbContext>(options =>
    options.UseSqlServer(connString),
    ServiceLifetime.Scoped);
```

---

### ⚡ Mistake to Avoid:

Sharing one `DbContext` instance across threads (e.g., in background tasks or static classes).
Use **`IDbContextFactory`** for multi-threaded scenarios.

```csharp
public class MyWorker
{
    private readonly IDbContextFactory<AppDbContext> _factory;
    public MyWorker(IDbContextFactory<AppDbContext> factory) => _factory = factory;

    public async Task RunAsync()
    {
        using var db = _factory.CreateDbContext();
        var data = await db.Users.AsNoTracking().ToListAsync();
    }
}
```

✅ Each task gets its own isolated context — safe and efficient.

---

## 🧩 19.2 Connection Pooling

Connection pooling is handled by ADO.NET automatically.
You can **tune it** with connection string parameters:

```plaintext
Server=.;Database=ProdDB;Trusted_Connection=True;Max Pool Size=200;Min Pool Size=5;
```

**Best Practices:**

* Keep `DbContext` lifetime short — open/close connections quickly.
* Don’t call `db.Database.OpenConnection()` manually unless necessary.
* Use **async I/O** to free threads under load.

---

## 🧩 19.3 Query Performance Optimization

### 🔹 Project Only Needed Columns

Avoid fetching full entities if you only need a few fields.

❌ Bad:

```csharp
var users = db.Users.ToList(); // Loads entire entity
```

✅ Good:

```csharp
var users = db.Users.Select(u => new { u.Id, u.Name }).ToList();
```

---

### 🔹 Use AsNoTracking for Read Queries

```csharp
var blogs = await db.Blogs.AsNoTracking().ToListAsync();
```

> Cuts out tracking overhead — ideal for reporting or read-only APIs.

---

### 🔹 Precompile Frequent Queries

```csharp
static readonly Func<AppDbContext, string, Task<User>> GetUserByEmail =
    EF.CompileAsyncQuery((AppDbContext db, string email) =>
        db.Users.FirstOrDefault(u => u.Email == email));
```

✅ Removes translation overhead for high-frequency queries.

---

### 🔹 Avoid N+1 Query Problem

When using navigation properties, EF may fire multiple queries.

❌ Bad:

```csharp
var blogs = db.Blogs.ToList();
foreach (var blog in blogs)
    Console.WriteLine(blog.Posts.Count);
```

✅ Good (eager load):

```csharp
var blogs = db.Blogs.Include(b => b.Posts).ToList();
```

Or use **projection**:

```csharp
var blogs = db.Blogs
    .Select(b => new { b.Url, PostCount = b.Posts.Count })
    .ToList();
```

---

## 🧩 19.4 Caching for EF Core

EF Core doesn’t have built-in second-level caching (by design).
You can use libraries like:

* 🧩 **EFCoreSecondLevelCacheInterceptor**
* 🧩 **EasyCaching**
* 🧩 **CacheManager.Core**

### Example — EFCoreSecondLevelCacheInterceptor

```bash
dotnet add package EFCoreSecondLevelCacheInterceptor
```

Configure:

```csharp
builder.Services.AddEFSecondLevelCache(options =>
    options.UseMemoryCacheProvider().DisableLogging(true));

builder.Services.AddDbContext<AppDbContext>(
    (serviceProvider, options) =>
    options
        .UseSqlServer(connString)
        .AddInterceptors(serviceProvider.GetRequiredService<SecondLevelCacheInterceptor>()));
```

Use:

```csharp
var users = db.Users
    .Cacheable() // Caches results in memory
    .ToList();
```

✅ The cache auto-invalidates when the entity changes (tracked via interceptor).

---

## 🧩 19.5 Parallelism and Async Concurrency

Always use async queries in high-traffic apps.

### ✅ Example

```csharp
var users = await db.Users.ToListAsync();
```

**Avoid:**

```csharp
var users = db.Users.ToList(); // Blocks threads
```

---

### ⚠️ Important:

Don’t share a single `DbContext` instance across parallel async tasks:

❌ Bad:

```csharp
var task1 = db.Users.ToListAsync();
var task2 = db.Posts.ToListAsync();
await Task.WhenAll(task1, task2); // Crash risk
```

✅ Correct:

```csharp
await Task.WhenAll(
    Task.Run(async () => await using var db1 = factory.CreateDbContext(); await db1.Users.ToListAsync()),
    Task.Run(async () => await using var db2 = factory.CreateDbContext(); await db2.Posts.ToListAsync())
);
```

Each parallel task uses its own context.

---

## 🧩 19.6 Compiled Models (EF Core 6+)

Compiled models **pre-generate** EF’s internal metadata and mapping logic for large models.

```bash
dotnet ef dbcontext optimize
```

Generates a precompiled model class (e.g., `CompiledModels.cs`)
✅ Reduces startup time by 50–70% for large schemas.

---

## 🧩 19.7 Indexing and Database Tuning

* Always index columns used in `WHERE`, `JOIN`, or `ORDER BY`.
* EF Fluent API example:

```csharp
modelBuilder.Entity<User>()
    .HasIndex(u => u.Email)
    .IsUnique();
```

* Use database profiling tools:

  * **SQL Server Profiler**
  * **Azure Data Studio**
  * **pgAdmin (PostgreSQL)**
  * **EXPLAIN / Query Analyzer**

---

## 🧩 19.8 Connection Resiliency and Retries

Enable retry policies for transient errors (network, timeouts):

```csharp
options.UseSqlServer(connString, sql =>
{
    sql.EnableRetryOnFailure(
        maxRetryCount: 5,
        maxRetryDelay: TimeSpan.FromSeconds(10),
        errorNumbersToAdd: null);
});
```

✅ Essential for cloud-hosted databases (Azure, AWS RDS).

---

## 🧩 19.9 Pooled DbContext Factory

For high-throughput apps, use **DbContext pooling** to reuse context instances.

```csharp
builder.Services.AddPooledDbContextFactory<AppDbContext>(options =>
    options.UseSqlServer(connString));
```

✅ Reduces allocation cost for frequently created DbContexts.

---

## 🧠 Exercises

1. **Compiled Query Benchmark**

   * Compare `EF.CompileQuery()` vs regular LINQ query in a loop of 10,000 iterations.

2. **N+1 Problem Simulation**

   * Load 100 blogs and posts without `Include()` vs with it. Measure SQL calls.

3. **Cache Interceptor**

   * Implement EFCoreSecondLevelCacheInterceptor and confirm that repeated queries don’t hit the DB.

4. **DbContext Pooling**

   * Enable pooling and log instantiations to verify reuse.

5. **Retry Policy**

   * Add `EnableRetryOnFailure()` and simulate transient DB exceptions (e.g., manual connection drop).

---

## ✅ Summary

| Feature                | Purpose                              | Result                  |
| ---------------------- | ------------------------------------ | ----------------------- |
| **DbContext Factory**  | Safe multi-threaded context creation | Avoids conflicts        |
| **Pooling**            | Reuse DbContexts                     | Faster context creation |
| **Compiled Queries**   | Cache query translation              | Lower CPU cost          |
| **AsNoTracking**       | Disable tracking for reads           | Faster read operations  |
| **Second-Level Cache** | Cache query results                  | Reduce DB load          |
| **Connection Pooling** | Reuse DB connections                 | Faster DB round-trips   |
| **Retry Policy**       | Handle transient failures            | More resilient app      |

---

**Key Takeaways:**

* Manage DbContext lifetime carefully — never share across threads.
* Use caching, compiled queries, and pooling to eliminate overhead.
* Always profile and index based on query patterns.
* Tune connection limits and retry logic for cloud reliability.
* Scale horizontally (multiple app instances) + optimize vertically (query and connection performance).
