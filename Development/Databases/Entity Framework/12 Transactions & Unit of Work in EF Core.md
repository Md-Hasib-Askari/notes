## ⚙️ Step 12: Transactions & Unit of Work in EF Core

### 📘 Notes

A **transaction** ensures that a group of database operations either **all succeed or all fail** — preventing partial updates that can corrupt your data.

Entity Framework Core automatically uses transactions for **each `SaveChanges()`** call, but you can also **manually manage** transactions across multiple operations.

---

## 🧩 12.1 Automatic Transactions (Default Behavior)

EF Core wraps every `SaveChanges()` in its own transaction.

**Example:**

```csharp
using var db = new AppDbContext();

var blog = new Blog { Url = "https://efcore.io" };
db.Blogs.Add(blog);
db.SaveChanges(); // ✅ Automatically wrapped in a transaction
```

If anything fails inside `SaveChanges()`, EF automatically rolls back the transaction.

---

## 🧩 12.2 Manual Transactions

Use this when you need to perform **multiple `SaveChanges()` calls atomically**, or when combining EF operations with raw SQL.

### 💡 Example — Manual Transaction

```csharp
using var db = new AppDbContext();
using var transaction = db.Database.BeginTransaction();

try
{
    db.Blogs.Add(new Blog { Url = "https://dotnet.microsoft.com" });
    db.SaveChanges();

    db.Posts.Add(new Post { Title = "EF Core Transactions", BlogId = 1 });
    db.SaveChanges();

    transaction.Commit(); // ✅ Commit only if both succeed
}
catch (Exception)
{
    transaction.Rollback(); // ❌ Rollback if any step fails
}
```

✅ This ensures both operations either **succeed together** or **fail together**.

---

## 🧩 12.3 Async Transactions

EF Core also supports async versions:

```csharp
await using var transaction = await db.Database.BeginTransactionAsync();

try
{
    db.Users.Add(new User { Name = "Alice" });
    await db.SaveChangesAsync();

    db.Roles.Add(new Role { Name = "Admin" });
    await db.SaveChangesAsync();

    await transaction.CommitAsync();
}
catch
{
    await transaction.RollbackAsync();
}
```

---

## 🧩 12.4 Savepoints (Nested Transactions)

You can create **savepoints** within a transaction — useful for partial rollbacks.

```csharp
using var db = new AppDbContext();
using var tx = db.Database.BeginTransaction();

try
{
    db.Users.Add(new User { Name = "John" });
    db.SaveChanges();

    tx.CreateSavepoint("AfterFirstInsert");

    db.Users.Add(new User { Name = "InvalidUser" });
    db.SaveChanges();

    tx.Commit();
}
catch
{
    tx.RollbackToSavepoint("AfterFirstInsert"); // rollback only last insert
    tx.Commit(); // keep the first insert
}
```

---

## 🧩 12.5 Using Transactions with Raw SQL

Transactions can also include **raw SQL** operations:

```csharp
using var db = new AppDbContext();
using var tx = db.Database.BeginTransaction();

try
{
    db.Database.ExecuteSqlRaw("INSERT INTO Logs (Message) VALUES ('Started')");
    db.Blogs.Add(new Blog { Url = "https://hasibdev.io" });
    db.SaveChanges();

    tx.Commit();
}
catch
{
    tx.Rollback();
}
```

✅ This is very useful when mixing EF Core and manual SQL commands.

---

## 🧩 12.6 Unit of Work Pattern

The **Unit of Work** pattern coordinates the work of multiple repositories, ensuring they share the same `DbContext` (and thus, the same transaction).

### Example — Repository + Unit of Work

**UserRepository.cs**

```csharp
public class UserRepository
{
    private readonly AppDbContext _db;
    public UserRepository(AppDbContext db) => _db = db;

    public void Add(User user) => _db.Users.Add(user);
}
```

**BlogRepository.cs**

```csharp
public class BlogRepository
{
    private readonly AppDbContext _db;
    public BlogRepository(AppDbContext db) => _db = db;

    public void Add(Blog blog) => _db.Blogs.Add(blog);
}
```

**UnitOfWork.cs**

```csharp
public class UnitOfWork : IDisposable
{
    private readonly AppDbContext _db;
    public UserRepository Users { get; }
    public BlogRepository Blogs { get; }

    public UnitOfWork(AppDbContext db)
    {
        _db = db;
        Users = new UserRepository(db);
        Blogs = new BlogRepository(db);
    }

    public int Save() => _db.SaveChanges();
    public async Task<int> SaveAsync() => await _db.SaveChangesAsync();

    public void Dispose() => _db.Dispose();
}
```

**Usage:**

```csharp
using var uow = new UnitOfWork(new AppDbContext());
uow.Users.Add(new User { Name = "Hasib" });
uow.Blogs.Add(new Blog { Url = "https://hasib.ai" });
uow.Save(); // ✅ Both are committed together
```

---

### 💡 Why Unit of Work Matters

| Benefit                   | Description                            |
| ------------------------- | -------------------------------------- |
| ✅ Transaction consistency | All repositories share one `DbContext` |
| ✅ Reduced coupling        | Keeps repositories independent         |
| ✅ Centralized commits     | All changes committed once             |
| ✅ Easier testing          | You can mock or swap `DbContext`       |

---

### 🧩 12.7 Ambient Transactions (Advanced)

For distributed or external operations, EF Core supports **System.Transactions**:

```csharp
using (var scope = new TransactionScope())
{
    db.Users.Add(new User { Name = "Alice" });
    db.SaveChanges();

    // Other DB or external service call here

    scope.Complete(); // Commits all participants
}
```

> Useful when coordinating multiple databases or services — but be cautious with distributed transaction overhead.

---

### ⚡ Common Pitfalls

| Problem                                      | Cause                                        | Solution                                     |
| -------------------------------------------- | -------------------------------------------- | -------------------------------------------- |
| “The connection is already in a transaction” | Multiple nested transactions                 | Use one shared transaction or savepoints     |
| Rollback didn’t undo changes                 | Forget to call `Rollback()`                  | Always wrap in `try/catch`                   |
| Lost changes in multi-threaded env           | Shared DbContext across threads              | Use separate DbContext instances per request |
| Partial commits                              | Multiple `SaveChanges()` without transaction | Wrap all in a manual transaction             |

---

### 🧠 Exercises

1. **Simple Transaction**

   * Insert a blog and post in a single manual transaction.
   * Throw an exception in between and confirm rollback.

2. **Savepoint Practice**

   * Create two inserts, add a savepoint, then trigger a rollback to it.
   * Confirm that the first insert remains.

3. **Raw SQL Transaction**

   * Mix `ExecuteSqlRaw()` with EF entities inside one transaction.

4. **Unit of Work**

   * Build a simple Unit of Work class that manages two repositories.
   * Perform inserts into both repositories and call `Save()` once.

5. **Async Transaction**

   * Create an async transaction that adds multiple entities and rolls back if any validation fails.

---

### ✅ Summary

| Feature                    | Description                                    | Example                   |
| -------------------------- | ---------------------------------------------- | ------------------------- |
| **Automatic Transactions** | Each `SaveChanges()` is atomic                 | Built-in                  |
| **Manual Transactions**    | Explicit control for multi-step operations     | `BeginTransaction()`      |
| **Async Transactions**     | Async version for web apps                     | `BeginTransactionAsync()` |
| **Savepoints**             | Partial rollback within a transaction          | `RollbackToSavepoint()`   |
| **Unit of Work**           | Aggregate pattern for multi-repository commits | Shared DbContext          |
| **ExecuteSqlRaw**          | Combine raw SQL in a transaction               | Hybrid approach           |

---

**Key Takeaway:**

* EF Core automatically protects your data with transactions,
* but for **multi-step or multi-repo operations**, use manual transactions or the **Unit of Work pattern** to maintain integrity and consistency.
