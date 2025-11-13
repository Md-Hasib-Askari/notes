## ⚙️ Step 8: Change Tracking & Performance Optimization in EF Core

### 📘 Notes

EF Core has an internal **Change Tracker** that keeps track of all entities you load, modify, add, or delete.
It determines **what SQL statements** to execute when you call `SaveChanges()`.

---

### 🧩 8.1 Entity States

Every entity tracked by EF Core is in one of these states:

| State       | Description             | Example                        |
| ----------- | ----------------------- | ------------------------------ |
| `Added`     | Entity will be inserted | `context.Add(entity)`          |
| `Modified`  | Entity will be updated  | After editing a tracked entity |
| `Deleted`   | Entity will be deleted  | `context.Remove(entity)`       |
| `Unchanged` | Entity matches the DB   | After being loaded from DB     |
| `Detached`  | Not tracked             | Manually created, not attached |

**Check state example:**

```csharp
using var db = new AppDbContext();
var blog = db.Blogs.First();
Console.WriteLine(db.Entry(blog).State); // Unchanged

blog.Url = "https://updatedurl.com";
Console.WriteLine(db.Entry(blog).State); // Modified
```

When you call:

```csharp
db.SaveChanges();
```

EF Core automatically generates the correct `UPDATE` statement for all modified entities.

---

### 🧩 8.2 Change Tracker in Action

```csharp
using var db = new AppDbContext();
var blog = db.Blogs.First();

blog.Url = "https://newsite.com";
db.SaveChanges();
```

**What EF Core does internally:**

* Detects that `Url` changed.
* Marks entity as `Modified`.
* Executes:

  ```sql
  UPDATE Blogs SET Url = 'https://newsite.com' WHERE BlogId = 1;
  ```

---

### 🧩 8.3 Detaching & Attaching Entities

Detached entities aren’t tracked. You often see this in web apps where data comes from an API.

```csharp
var updatedBlog = new Blog { BlogId = 1, Url = "https://reconnected.com" };
db.Blogs.Attach(updatedBlog);
db.Entry(updatedBlog).State = EntityState.Modified;
db.SaveChanges();
```

---

### 🧩 8.4 AsNoTracking — Boost Read Performance

For **read-only queries**, change tracking is unnecessary and slows things down.
Disable it with `.AsNoTracking()`:

```csharp
var blogs = db.Blogs.AsNoTracking().ToList();
```

* **Tracked Query:** Uses memory to track every object → slower for large reads.
* **No Tracking:** Faster, less memory overhead, cannot update those entities.

**Benchmark Example:**

```csharp
var tracked = db.Blogs.ToList();
var notTracked = db.Blogs.AsNoTracking().ToList();
```

---

### 🧩 8.5 Explicit State Management

You can manually set an entity’s state:

```csharp
db.Entry(blog).State = EntityState.Modified;
db.SaveChanges();
```

Or apply state changes to multiple entities:

```csharp
foreach (var post in posts)
    db.Entry(post).State = EntityState.Modified;
```

---

### 🧩 8.6 AutoDetectChanges Optimization

EF Core automatically detects changes before `SaveChanges()`.
For bulk operations, disable this temporarily for better performance:

```csharp
db.ChangeTracker.AutoDetectChangesEnabled = false;
foreach (var item in largeList)
    db.Blogs.Add(item);
db.SaveChanges();
db.ChangeTracker.AutoDetectChangesEnabled = true;
```

---

### 🧩 8.7 Logging and Tracking Queries

Enable query logging to inspect what EF Core executes:

```csharp
protected override void OnConfiguring(DbContextOptionsBuilder options)
{
    options
        .UseSqlite("Data Source=EfPerfDemo.db")
        .LogTo(Console.WriteLine, LogLevel.Information);
}
```

Or check individual query SQL:

```csharp
var query = db.Blogs.Where(b => b.BlogId > 0);
Console.WriteLine(query.ToQueryString());
```

---

### 🧩 8.8 Batch Operations & Large Data Sets

* EF Core processes `SaveChanges()` in **batches of 1000 rows by default**.
* For **large inserts/updates**, use:

  * [EFCore.BulkExtensions](https://github.com/borisdj/EFCore.BulkExtensions)
  * [Z.EntityFramework.Extensions](https://entityframework-extensions.net/)

Example with EFCore.BulkExtensions:

```csharp
using EFCore.BulkExtensions;
await db.BulkInsertAsync(largeList);
```

---

### ⚡ Performance Tips Summary

| Technique                          | Description                            |
| ---------------------------------- | -------------------------------------- |
| `AsNoTracking()`                   | Disable tracking for read-only queries |
| `AutoDetectChangesEnabled = false` | Boosts speed in bulk operations        |
| `Select()`                         | Fetch only needed columns              |
| `Include()`                        | Load related data efficiently          |
| `ToQueryString()`                  | Debug generated SQL                    |
| Use Async (`await`)                | Prevent thread blocking                |
| Use Batching                       | For large inserts/updates              |

---

### 🧠 Exercises

1. **State Inspection**

   * Load a blog, print its state before and after modification.
   * Print all tracked entity states using:

     ```csharp
     db.ChangeTracker.Entries().ToList()
     ```

2. **Detached Update**

   * Create a new `Blog` object (not from DB).
   * Attach it, mark it as modified, and update it.

3. **Performance Comparison**

   * Fetch 10,000 blogs using:

     1. Regular query
     2. `.AsNoTracking()`
   * Measure the time difference.

4. **AutoDetectChanges**

   * Insert 1000 entities with and without `AutoDetectChangesEnabled`.
   * Compare runtime.

5. **Query Logging**

   * Enable logging and run a few queries.
   * Inspect generated SQL in console output.

---

### ✅ Summary

* EF Core tracks entity state changes automatically.
* You can manually control state and optimize performance.
* `.AsNoTracking()` is essential for large read-only operations.
* Use logging to understand query behavior and tune performance.
