## ⚙️ Step 4: CRUD Operations in EF Core

### 📘 Notes

CRUD = **Create, Read, Update, Delete** — the core database operations.
EF Core simplifies all these through the `DbSet<T>` and `DbContext`.

Let’s assume we already have this setup:

```csharp
public class Blog
{
    public int BlogId { get; set; }
    public string Url { get; set; }
}

public class AppDbContext : DbContext
{
    public DbSet<Blog> Blogs { get; set; }

    protected override void OnConfiguring(DbContextOptionsBuilder options)
        => options.UseSqlite("Data Source=EfCrudDemo.db");
}
```

---

### 🧩 4.1 CREATE — Insert New Records

Add new entities using `Add()` or `AddRange()`, then persist using `SaveChanges()`.

```csharp
using var db = new AppDbContext();

var blog1 = new Blog { Url = "https://learn.microsoft.com" };
var blog2 = new Blog { Url = "https://hasibdev.io" };

db.Blogs.Add(blog1);
db.Blogs.Add(blog2);
db.SaveChanges();

Console.WriteLine("Blogs inserted successfully!");
```

**EF Core Behavior:**

* Tracks added entities in memory.
* When you call `SaveChanges()`, EF generates and executes the necessary SQL INSERT statements.

---

### 🧩 4.2 READ — Fetch Data

Use **LINQ queries** to read data.
EF Core translates LINQ → SQL automatically.

```csharp
using var db = new AppDbContext();

var allBlogs = db.Blogs.ToList(); // SELECT * FROM Blogs

foreach (var blog in allBlogs)
    Console.WriteLine($"[{blog.BlogId}] {blog.Url}");
```

You can filter, sort, and project:

```csharp
var filtered = db.Blogs
    .Where(b => b.Url.Contains("microsoft"))
    .OrderBy(b => b.BlogId)
    .Select(b => new { b.Url })
    .ToList();

foreach (var b in filtered)
    Console.WriteLine(b.Url);
```

---

### 🧩 4.3 UPDATE — Modify Existing Data

EF Core **tracks entities automatically** when you query them, so updating is simple:

```csharp
using var db = new AppDbContext();

var blog = db.Blogs.First(); // tracked
blog.Url = "https://updatedblog.com";
db.SaveChanges(); // EF issues UPDATE statement
```

If you have a detached entity (e.g., from API), attach it first:

```csharp
var detachedBlog = new Blog { BlogId = 1, Url = "https://detachedupdate.com" };
db.Blogs.Update(detachedBlog);
db.SaveChanges();
```

---

### 🧩 4.4 DELETE — Remove Data

Use `Remove()` or `RemoveRange()`:

```csharp
using var db = new AppDbContext();

var blog = db.Blogs.First();
db.Blogs.Remove(blog);
db.SaveChanges();
```

You can also delete by ID (without fetching first):

```csharp
var b = new Blog { BlogId = 2 };
db.Entry(b).State = EntityState.Deleted;
db.SaveChanges();
```

---

### 🧩 4.5 ASYNC CRUD (for web apps)

EF Core supports async operations — always use them in ASP.NET Core.

```csharp
await db.Blogs.AddAsync(new Blog { Url = "https://asyncinsert.com" });
await db.SaveChangesAsync();

var blogs = await db.Blogs.ToListAsync();
```

---

### 🧠 Exercises

1. Create a console app that performs **all 4 CRUD operations** through a simple text menu.
   Example menu:

   ```
   1. Add Blog
   2. View Blogs
   3. Update Blog
   4. Delete Blog
   ```

2. Add a new entity `Post` with fields:

   * `PostId`, `Title`, `Content`, `BlogId` (FK).
     Create its table via migration and perform CRUD for posts as well.

3. Practice writing **LINQ queries**:

   * Find all posts where title contains “AI”.
   * Sort posts by `Title`.
   * Count total posts for each blog.

4. (Optional) Implement CRUD using **async** methods.

---

### 💡 Quick Summary

| Operation | EF Core Method                      | SQL Equivalent |
| --------- | ----------------------------------- | -------------- |
| Create    | `Add()`, `AddRange()`               | `INSERT`       |
| Read      | LINQ (`ToList`, `Where`, etc.)      | `SELECT`       |
| Update    | Change properties + `SaveChanges()` | `UPDATE`       |
| Delete    | `Remove()`                          | `DELETE`       |
