## ⚙️ Step 11: Compiled Queries & Raw SQL in EF Core

### 📘 Notes

When EF Core executes a LINQ query, it must **translate** the expression tree into SQL every time.
For frequently executed queries (like in high-traffic APIs), this translation overhead can become noticeable.

Two advanced techniques help:

1. **Compiled Queries** → Precompile the query once and reuse it.
2. **Raw SQL Queries** → Write SQL manually for maximum control and performance.

---

## 🧩 11.1 Compiled Queries

A **compiled query** caches the translation of a LINQ expression so EF doesn’t have to re-parse it every time.

This is most useful for:

* Read-heavy operations
* Repetitive queries in loops or services

---

### 💡 Example — Regular Query

```csharp
var activeUsers = db.Users
    .Where(u => u.IsActive)
    .ToList();
```

Every time this executes, EF must:

1. Build an expression tree
2. Translate it to SQL
3. Execute it

---

### ⚡ Compiled Query Version

```csharp
using Microsoft.EntityFrameworkCore;

static readonly Func<AppDbContext, bool, List<User>> GetUsersByStatus =
    EF.CompileQuery((AppDbContext db, bool isActive) =>
        db.Users.Where(u => u.IsActive == isActive).ToList());

using var db = new AppDbContext();

var active = GetUsersByStatus(db, true);
var inactive = GetUsersByStatus(db, false);
```

✅ **Benefit:**
The SQL generation and expression parsing are done **once**, so subsequent calls are significantly faster.

---

### 💡 Async Compiled Query

```csharp
static readonly Func<AppDbContext, string, Task<User>> GetUserByEmail =
    EF.CompileAsyncQuery((AppDbContext db, string email) =>
        db.Users.FirstOrDefault(u => u.Email == email));

var user = await GetUserByEmail(db, "test@domain.com");
```

---

### ⚙️ When to Use Compiled Queries

| Situation                                          | Use Compiled Query?                      |
| -------------------------------------------------- | ---------------------------------------- |
| Query runs once                                    | ❌ No                                     |
| Query runs hundreds of times (e.g., API endpoints) | ✅ Yes                                    |
| Query structure changes dynamically                | ❌ No (parameters allowed, structure not) |

---

## 🧩 11.2 Raw SQL Queries

Sometimes EF Core’s LINQ translation isn’t expressive enough — for example, when using database-specific functions or performance tuning.

EF Core lets you run **raw SQL safely**, while still mapping results to entities.

---

### 💡 Example — Raw SQL for Entities

```csharp
var users = db.Users
    .FromSqlRaw("SELECT * FROM Users WHERE IsActive = 1")
    .ToList();
```

This query:

* Maps directly to the `Users` DbSet
* Returns tracked entities
* Can use parameters safely

---

### 🧩 Parameterized SQL (Safe)

Always use parameters to prevent SQL injection:

```csharp
var email = "john@example.com";

var user = db.Users
    .FromSqlInterpolated($"SELECT * FROM Users WHERE Email = {email}")
    .FirstOrDefault();
```

✅ EF Core automatically parameterizes the query behind the scenes.

---

### 💡 Example — Raw SQL for Non-Entity Types (DTOs)

You can project results into **custom DTOs** using `Database.SqlQuery<T>` (EF Core 8+).

```csharp
var report = db.Database
    .SqlQuery<ReportDto>("SELECT COUNT(*) AS UserCount FROM Users")
    .ToList();
```

---

### 🧩 11.3 Executing Non-Query SQL Commands

Use this for manual updates, inserts, or stored procedure calls.

```csharp
var affectedRows = db.Database.ExecuteSqlRaw(
    "UPDATE Users SET IsActive = 0 WHERE LastLogin < '2025-01-01'");
```

Or with interpolation:

```csharp
db.Database.ExecuteSqlInterpolated(
    $"DELETE FROM Users WHERE Email = {email}");
```

---

### 💡 Example — Stored Procedure Call

```csharp
var result = db.Users
    .FromSqlRaw("EXEC GetActiveUsers")
    .ToList();
```

> Works only if the stored procedure returns columns that match your entity.

---

### 🧩 11.4 Combining Raw SQL with LINQ

You can still append LINQ operators to raw SQL queries:

```csharp
var users = db.Users
    .FromSqlRaw("SELECT * FROM Users")
    .Where(u => u.IsActive)
    .ToList();
```

✅ EF Core will append your `WHERE` clause to the SQL query automatically.

---

### ⚡ Common Pitfalls

| Problem                   | Cause                      | Solution                                           |
| ------------------------- | -------------------------- | -------------------------------------------------- |
| SQL Injection             | String concatenation       | Use `FromSqlInterpolated()`                        |
| Missing key mapping       | Raw SQL doesn’t return PK  | Include primary key columns                        |
| Changes not tracked       | DTO queries or projections | Use entities for tracked results                   |
| Query caching not working | Dynamic SQL strings        | Use parameters instead of building SQL dynamically |

---

### 🧠 Exercises

1. **Compiled Query Practice**

   * Create a compiled query that returns all users by role (parameterized).
   * Measure execution time vs regular LINQ query in a loop of 1000 iterations.

2. **Async Compiled Query**

   * Create an async compiled query to fetch posts by title keyword.

3. **Raw SQL Practice**

   * Run a `FromSqlRaw()` query to fetch blogs created after a specific date.
   * Add a `.Where()` LINQ filter afterward.

4. **Execute Non-Query**

   * Use `ExecuteSqlInterpolated()` to deactivate users who haven’t logged in for over a year.

5. **Challenge**

   * Combine compiled query + raw SQL by fetching users via compiled LINQ and joining with a raw SQL table (advanced integration test).

---

### ✅ Summary

| Feature                  | Use Case                                    | Benefits                     |
| ------------------------ | ------------------------------------------- | ---------------------------- |
| **Compiled Query**       | High-frequency, repeated queries            | Faster execution, cached SQL |
| **Async Compiled Query** | Same as above, async context                | Non-blocking, efficient      |
| **Raw SQL (FromSqlRaw)** | DB-specific optimization, stored procedures | Full SQL control             |
| **ExecuteSqlRaw**        | Non-query SQL (UPDATE, DELETE)              | Manual control when needed   |
| **FromSqlInterpolated**  | Safe SQL parameterization                   | Prevents SQL injection       |

---

**Key Takeaways:**

* Compiled queries remove LINQ translation overhead for repeated calls.
* Use raw SQL for complex DB logic or legacy stored procedures.
* Always parameterize to keep queries secure.
* Never mix untrusted user input directly into SQL strings.
