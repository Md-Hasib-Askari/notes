## 🧩 Step 17: Testing with In-Memory & SQLite Databases in EF Core

### 📘 Overview

When testing EF Core apps, the goal is to:

* Run **unit and integration tests quickly**
* Avoid touching real production data
* Validate your queries, CRUD, and business logic safely

EF Core provides **two main test database providers**:

| Provider                  | Description                             | Use Case                        |
| ------------------------- | --------------------------------------- | ------------------------------- |
| **In-Memory Provider**    | Stores data in memory, doesn’t use SQL  | Unit tests (no schema behavior) |
| **SQLite In-Memory Mode** | Real relational engine stored in memory | Integration tests (real SQL)    |

---

## ⚙️ 17.1 Testing with In-Memory Provider

The simplest way to test — zero setup.

### 💡 Example — Using InMemoryDatabase

```csharp
using Microsoft.EntityFrameworkCore;
using Xunit;

public class BlogTests
{
    private AppDbContext GetContext()
    {
        var options = new DbContextOptionsBuilder<AppDbContext>()
            .UseInMemoryDatabase("TestDb")
            .Options;
        return new AppDbContext(options);
    }

    [Fact]
    public void CanAddBlog()
    {
        using var db = GetContext();

        db.Blogs.Add(new Blog { Url = "https://hasib.dev" });
        db.SaveChanges();

        Assert.Single(db.Blogs);
    }
}
```

✅ Quick, no configuration, perfect for small tests.

---

### ⚠️ Limitations of InMemory Provider

| Limitation                         | Description                                    |
| ---------------------------------- | ---------------------------------------------- |
| No relational behavior             | No FK enforcement or transactions              |
| No SQL translation                 | LINQ queries don’t test SQL correctness        |
| Different from production provider | You might pass a test that fails on SQL Server |

Hence, for real validation — use SQLite.

---

## 🧩 17.2 Testing with SQLite (In-Memory Relational Mode)

SQLite supports full **relational behavior** (constraints, indexes, etc.), but entirely in memory.

### 💡 Example — SQLite In-Memory

```csharp
using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;
using Xunit;

public class BlogSqliteTests
{
    private AppDbContext GetContext(SqliteConnection connection)
    {
        var options = new DbContextOptionsBuilder<AppDbContext>()
            .UseSqlite(connection)
            .Options;

        return new AppDbContext(options);
    }

    [Fact]
    public void CanInsertAndRetrieveBlog()
    {
        var connection = new SqliteConnection("DataSource=:memory:");
        connection.Open(); // keep alive for test lifetime

        using (var db = GetContext(connection))
        {
            db.Database.EnsureCreated(); // creates schema
            db.Blogs.Add(new Blog { Url = "https://hasib.ai" });
            db.SaveChanges();
        }

        using (var db = GetContext(connection))
        {
            var blog = db.Blogs.First();
            Assert.Equal("https://hasib.ai", blog.Url);
        }

        connection.Close();
    }
}
```

✅ This behaves almost exactly like a real relational database — including constraints, transactions, and SQL generation.

---

## 🧩 17.3 Reusing Test Database Context

You can use a **shared in-memory database** for integration tests.

```csharp
public class TestDatabaseFixture : IDisposable
{
    public SqliteConnection Connection { get; }

    public TestDatabaseFixture()
    {
        Connection = new SqliteConnection("DataSource=:memory:");
        Connection.Open();
        using var db = new AppDbContext(new DbContextOptionsBuilder<AppDbContext>()
            .UseSqlite(Connection).Options);
        db.Database.EnsureCreated();
    }

    public void Dispose() => Connection.Close();
}
```

**Use it across tests:**

```csharp
public class BlogServiceTests : IClassFixture<TestDatabaseFixture>
{
    private readonly TestDatabaseFixture _fixture;

    public BlogServiceTests(TestDatabaseFixture fixture) => _fixture = fixture;

    [Fact]
    public void CanAddMultipleBlogs()
    {
        using var db = new AppDbContext(new DbContextOptionsBuilder<AppDbContext>()
            .UseSqlite(_fixture.Connection).Options);

        db.Blogs.AddRange(
            new Blog { Url = "https://a.com" },
            new Blog { Url = "https://b.com" });
        db.SaveChanges();

        Assert.Equal(2, db.Blogs.Count());
    }
}
```

✅ The schema persists between tests, while data can be reset.

---

## 🧩 17.4 Testing Repositories and Services

Example of testing a repository pattern:

```csharp
public class BlogRepository
{
    private readonly AppDbContext _db;
    public BlogRepository(AppDbContext db) => _db = db;

    public void Add(Blog blog) => _db.Blogs.Add(blog);
    public Blog Get(int id) => _db.Blogs.Find(id);
    public void Save() => _db.SaveChanges();
}

public class BlogRepositoryTests
{
    [Fact]
    public void AddAndRetrieveBlog()
    {
        var options = new DbContextOptionsBuilder<AppDbContext>()
            .UseInMemoryDatabase("RepoTest").Options;

        using var db = new AppDbContext(options);
        var repo = new BlogRepository(db);

        repo.Add(new Blog { Url = "https://testrepo.com" });
        repo.Save();

        var result = repo.Get(1);
        Assert.Equal("https://testrepo.com", result.Url);
    }
}
```

✅ Use InMemory for quick logic tests; use SQLite for relational testing.

---

## 🧩 17.5 Seeding Test Data

You can seed your test DB before tests run:

```csharp
public static void Seed(AppDbContext db)
{
    db.Blogs.AddRange(
        new Blog { Url = "https://seeded1.dev" },
        new Blog { Url = "https://seeded2.dev" });
    db.SaveChanges();
}
```

Call this inside your test setup (`EnsureCreated` first).

---

## 🧩 17.6 Testing Transactions and Constraints (SQLite Only)

SQLite supports transactions and foreign key enforcement.

```csharp
[Fact]
public void EnforcesForeignKeyConstraint()
{
    var connection = new SqliteConnection("DataSource=:memory:");
    connection.Open();

    var options = new DbContextOptionsBuilder<AppDbContext>()
        .UseSqlite(connection)
        .Options;

    using var db = new AppDbContext(options);
    db.Database.EnsureCreated();

    var post = new Post { BlogId = 999 }; // Invalid FK
    db.Posts.Add(post);

    Assert.Throws<DbUpdateException>(() => db.SaveChanges());
}
```

✅ Great for verifying schema correctness.

---

## 🧩 17.7 Comparing Both Providers

| Feature                | InMemory   | SQLite In-Memory           |
| ---------------------- | ---------- | -------------------------- |
| Speed                  | ⚡ Fastest  | ⚡ Fast                     |
| Relational Constraints | ❌ No       | ✅ Yes                      |
| SQL Translation        | ❌ Skipped  | ✅ Real SQL                 |
| Best For               | Unit Tests | Integration Tests          |
| Persistent Schema      | ✅ Yes      | ⚠️ Only if connection open |

---

## 🧠 Exercises

1. **Basic CRUD Test (InMemory)**

   * Create a `UserRepository` and test all CRUD methods using `UseInMemoryDatabase`.

2. **Relational Test (SQLite)**

   * Create `Blog` → `Post` relation.
   * Write a test that confirms a post cannot exist without a blog (foreign key constraint).

3. **Data Seeding**

   * Seed 3 test users and verify `Count()` and query results.

4. **Transaction Test**

   * Test rollback behavior by throwing an exception mid-transaction.

5. **Fixture Test**

   * Use `IClassFixture<TestDatabaseFixture>` to share the same SQLite in-memory DB across multiple tests.

---

### ✅ Summary

| Concept               | Use                           | Key Method                          |
| --------------------- | ----------------------------- | ----------------------------------- |
| **InMemory Provider** | Fast unit testing             | `.UseInMemoryDatabase()`            |
| **SQLite In-Memory**  | Relational testing            | `.UseSqlite("DataSource=:memory:")` |
| **EnsureCreated()**   | Creates schema manually       | `db.Database.EnsureCreated()`       |
| **Shared Connection** | Persist schema between tests  | Keep connection open                |
| **Seeding**           | Setup data for tests          | Manual or OnModelCreating           |
| **Foreign Key Tests** | Enforce referential integrity | SQLite only                         |

---

**Key Takeaways:**

* Use **InMemory** for speed; **SQLite** for realistic relational behavior.
* Always open the SQLite connection manually in tests.
* You can simulate transactions, foreign keys, and migrations reliably with SQLite.
* Keep test databases small, isolated, and disposable.
