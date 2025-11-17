## 🧩 Step 6: Querying with LINQ in EF Core

### 📘 Notes

EF Core integrates tightly with **LINQ (Language Integrated Query)**, which lets you query the database using **C# syntax** instead of SQL.
EF Core translates LINQ expressions into optimized SQL queries behind the scenes.

---

### ⚙️ 6.1 Basic Querying

Let’s say you have:

```csharp
public class Blog
{
    public int BlogId { get; set; }
    public string Url { get; set; }
}

public class Post
{
    public int PostId { get; set; }
    public string Title { get; set; }
    public string Content { get; set; }
    public int BlogId { get; set; }
    public Blog Blog { get; set; }
}
```

---

#### 🔍 Example — Basic Query

```csharp
using var db = new AppDbContext();

// Fetch all blogs
var blogs = db.Blogs.ToList();

foreach (var blog in blogs)
    Console.WriteLine($"{blog.BlogId} - {blog.Url}");
```

> `.ToList()` executes the query and loads data into memory.
> Without it, the query remains **deferred** (not yet sent to the DB).

---

### 🧩 6.2 Filtering Data

```csharp
var microsoftBlogs = db.Blogs
    .Where(b => b.Url.Contains("microsoft"))
    .ToList();
```

**Equivalent SQL:**

```sql
SELECT * FROM Blogs WHERE Url LIKE '%microsoft%'
```

---

### 🧩 6.3 Sorting and Projection

**Sorting:**

```csharp
var orderedBlogs = db.Blogs
    .OrderBy(b => b.BlogId)
    .ToList();
```

**Projection (select specific columns):**

```csharp
var blogUrls = db.Blogs
    .Select(b => new { b.Url })
    .ToList();
```

This generates SQL that selects only the `Url` column, making queries more efficient.

---

### 🧩 6.4 Joining Related Entities

If you have a relationship between `Blog` and `Post`:

**Eager Loading with Include:**

```csharp
var blogs = db.Blogs
    .Include(b => b.Posts)
    .ToList();

foreach (var b in blogs)
{
    Console.WriteLine($"Blog: {b.Url}");
    foreach (var post in b.Posts)
        Console.WriteLine($"  - {post.Title}");
}
```

> EF Core automatically performs an INNER JOIN under the hood.

---

### 🧩 6.5 Aggregation & Counting

```csharp
int totalBlogs = db.Blogs.Count();
int totalPosts = db.Posts.Count(p => p.Title.Contains("AI"));
```

**Group and Aggregate:**

```csharp
var postCounts = db.Posts
    .GroupBy(p => p.BlogId)
    .Select(g => new
    {
        BlogId = g.Key,
        Count = g.Count()
    })
    .ToList();
```

---

### 🧩 6.6 Paging (Skip & Take)

Useful for pagination in web APIs.

```csharp
var page = 2;
var pageSize = 5;

var pagedPosts = db.Posts
    .OrderBy(p => p.PostId)
    .Skip((page - 1) * pageSize)
    .Take(pageSize)
    .ToList();
```

---

### 🧩 6.7 Raw SQL Queries (when needed)

When you must run custom SQL:

```csharp
var posts = db.Posts
    .FromSqlRaw("SELECT * FROM Posts WHERE Title LIKE '%EF%'")
    .ToList();
```

> Be cautious — use parameterized SQL to avoid injection:

```csharp
var keyword = "AI";
var posts = db.Posts
    .FromSqlInterpolated($"SELECT * FROM Posts WHERE Title LIKE '%{keyword}%'")
    .ToList();
```

---

### 🧠 Exercises

1. **Filtering Practice**

   * Fetch all posts containing the keyword `"EF Core"`.
   * Fetch all blogs with URLs starting with `"https"`.

2. **Projection Practice**

   * Select only `Title` and `BlogId` from `Posts`.
   * Display `Blog.Url` with the count of its posts.

3. **Grouping and Aggregation**

   * Group posts by `BlogId` and display the total number of posts per blog.
   * Find the blog that has the maximum number of posts.

4. **Pagination Challenge**

   * Implement pagination logic for displaying posts, 5 per page.

5. **Join Challenge**

   * Fetch posts along with their blog URL using either `Include()` or `Join()`.

---

### ⚡ Performance Tips

| Tip                              | Description                                         |
| -------------------------------- | --------------------------------------------------- |
| Use `AsNoTracking()`             | For read-only queries (faster, no change tracking). |
| Use projections (`Select`)       | Fetch only needed columns.                          |
| Use async methods                | Always in web APIs or high-load apps.               |
| Avoid multiple `.ToList()` calls | Execute once; reuse the results.                    |

---

### 🧩 Async Example

```csharp
var recentPosts = await db.Posts
    .Where(p => p.PostId > 10)
    .OrderByDescending(p => p.PostId)
    .ToListAsync();
```

---

✅ **In summary:**

* LINQ = query language in C# for EF.
* EF translates LINQ to SQL automatically.
* Learn to read generated SQL using:

  ```csharp
  db.Posts.Where(p => p.PostId > 10).ToQueryString();
  ```