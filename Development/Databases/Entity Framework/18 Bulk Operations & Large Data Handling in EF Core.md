## 🧩 Step 18: Bulk Operations & Large Data Handling in EF Core

### 📘 Overview

By default, EF Core handles inserts, updates, and deletes **one entity at a time**, which becomes slow for large datasets.

To optimize:

* Use **batch operations** and **bulk libraries**.
* Minimize **Change Tracker overhead**.
* Stream or chunk large operations.
* Tune **transaction settings** and **context lifetime**.

---

## ⚙️ 18.1 The Problem — Default EF Core Behavior

Example (inefficient):

```csharp
using var db = new AppDbContext();

var blogs = Enumerable.Range(1, 10000)
    .Select(i => new Blog { Url = $"https://blog{i}.com" });

db.Blogs.AddRange(blogs);
db.SaveChanges(); // ❌ Inserts one by one
```

EF Core executes **10,000 separate INSERT statements** — slow and memory heavy.

---

## 🧩 18.2 Optimize with Batching (Native EF Behavior)

Since EF Core 2.1+, `SaveChanges()` batches multiple statements together automatically.
You can configure batch size in SQL Server:

```csharp
options.UseSqlServer(connectionString, o => o.MaxBatchSize(500));
```

✅ EF Core executes **batches of 500 statements per round-trip**.

---

## 🧩 18.3 Bulk Extensions (Fastest Solution)

Use the open-source package **EFCore.BulkExtensions** for true bulk operations.

### 📦 Install

```bash
dotnet add package EFCore.BulkExtensions
```

### 💡 Example — Bulk Insert

```csharp
using EFCore.BulkExtensions;

var users = Enumerable.Range(1, 100000)
    .Select(i => new User { Name = $"User {i}" }).ToList();

await db.BulkInsertAsync(users);
```

✅ Inserts 100,000 rows in a few seconds — *hundreds of times faster* than normal EF.

---

### 💡 Bulk Update

```csharp
await db.BulkUpdateAsync(users);
```

### 💡 Bulk Delete

```csharp
await db.BulkDeleteAsync(users);
```

### 💡 Bulk Read (with filter)

```csharp
var results = await db.BulkReadAsync(new List<User> { new() { Id = 5 } });
```

---

### ⚡ Bulk Extensions Advantages

| Feature      | Description               |
| ------------ | ------------------------- |
| Bulk Insert  | Fast, minimal tracking    |
| Bulk Update  | Update many rows at once  |
| Bulk Delete  | Efficient large deletions |
| Bulk Merge   | Upsert (insert or update) |
| Preserve IDs | Keeps identity values     |
| SQL Logging  | Shows generated SQL       |

---

### 🧠 Example — Bulk Merge (Upsert)

```csharp
await db.BulkMergeAsync(users);
```

If an entity exists → it’s updated.
If not → it’s inserted.

---

## 🧩 18.4 Disable Change Tracking for Speed

For large reads or imports, turn off EF’s Change Tracker.

```csharp
db.ChangeTracker.AutoDetectChangesEnabled = false;
db.Blogs.AddRange(largeList);
db.SaveChanges();
db.ChangeTracker.AutoDetectChangesEnabled = true;
```

✅ Avoids re-evaluating thousands of tracked entities.

---

## 🧩 18.5 Use No-Tracking Queries for Read-Heavy Workloads

```csharp
var blogs = db.Blogs.AsNoTracking().ToList();
```

* Saves memory and CPU time.
* Recommended when reading large datasets without updating them.

---

## 🧩 18.6 Stream Large Data Sets (Avoid Loading All in Memory)

When processing millions of rows:

* Stream results using **`AsAsyncEnumerable()`**
* Or use **pagination** (`Skip` + `Take`)

### Example — Stream Processing

```csharp
await foreach (var user in db.Users.AsNoTracking().AsAsyncEnumerable())
{
    Console.WriteLine(user.Name);
}
```

✅ Keeps memory usage low — processes one record at a time.

---

## 🧩 18.7 Bulk Insert Using Raw SQL (Manual Option)

If you can’t use EFCore.BulkExtensions, use raw SQL for mass inserts.

```csharp
var sql = "INSERT INTO Users (Name) VALUES " +
          string.Join(",", Enumerable.Range(1, 1000).Select(i => $"('User{i}')"));
await db.Database.ExecuteSqlRawAsync(sql);
```

⚠️ Be careful — validate and sanitize all inputs to prevent injection.

---

## 🧩 18.8 Transactions in Bulk Operations

Wrap large bulk operations in a transaction for safety:

```csharp
using var transaction = db.Database.BeginTransaction();

await db.BulkInsertAsync(users);
await db.BulkInsertAsync(blogs);

await transaction.CommitAsync();
```

✅ Ensures all data is inserted or rolled back together.

---

## 🧩 18.9 Handling Memory Pressure

For extremely large datasets (e.g., imports), process in **chunks**:

```csharp
int batchSize = 1000;
for (int i = 0; i < total; i += batchSize)
{
    var chunk = users.Skip(i).Take(batchSize).ToList();
    await db.BulkInsertAsync(chunk);
}
```

✅ Prevents out-of-memory crashes and keeps operations stable.

---

## 🧩 18.10 Measuring Performance

Log SQL timings with interceptors or EF’s logging:

```csharp
options.LogTo(Console.WriteLine, LogLevel.Information);
```

Or manually time it:

```csharp
var stopwatch = Stopwatch.StartNew();
await db.BulkInsertAsync(users);
stopwatch.Stop();

Console.WriteLine($"Inserted in {stopwatch.ElapsedMilliseconds} ms");
```

---

## ⚡ Benchmarks (Typical Results)

| Operation | 10,000 Records | Default EF  | BulkExtensions |
| --------- | -------------- | ----------- | -------------- |
| Insert    | 45 sec         | **0.8 sec** |                |
| Update    | 50 sec         | **1.1 sec** |                |
| Delete    | 35 sec         | **0.7 sec** |                |

Bulk libraries are **50–100× faster** for large operations.

---

## 🧠 Exercises

1. **Bulk Insert**

   * Generate 50,000 fake users with `Bogus` library.
   * Bulk insert them using EFCore.BulkExtensions.
   * Compare execution time with normal EF.

2. **Bulk Update**

   * Update a field for 100,000 users using both EF loop and BulkUpdateAsync.
   * Measure time difference.

3. **Chunked Import**

   * Read a CSV file of 1M records in batches of 5,000.
   * Insert in chunks and monitor memory usage.

4. **Streaming**

   * Stream 100k rows using `AsAsyncEnumerable()` and process one by one.

5. **Transaction Safety**

   * Wrap multiple bulk operations in a transaction.
   * Intentionally throw an exception to confirm rollback works.

---

## ✅ Summary

| Technique                         | Description                   | Best For                 |
| --------------------------------- | ----------------------------- | ------------------------ |
| **Batching**                      | EF groups SQL commands        | Moderate datasets        |
| **Bulk Extensions**               | High-performance operations   | Massive imports/updates  |
| **Disable Change Tracking**       | Skip tracking overhead        | Bulk writes              |
| **AsNoTracking()**                | Read-only queries             | Reports, exports         |
| **Streaming (AsAsyncEnumerable)** | Row-by-row reads              | Huge datasets            |
| **Chunk Processing**              | Divide data into smaller sets | Memory safety            |
| **Transactions**                  | Atomic multi-step bulk ops    | Reliable bulk operations |

---

**Key Takeaways:**

* Default EF Core is great for CRUD, not bulk operations.
* Use **EFCore.BulkExtensions** for serious performance boosts.
* Disable change tracking and stream for large data pipelines.
* Always chunk large imports to prevent crashes.
* Log and measure performance continuously.
