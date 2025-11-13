## 🧩 Step 7: Relationships and Navigation Properties in EF Core

### 📘 Notes

In relational databases, data is connected using **foreign keys**.
In EF Core, those connections are represented through **navigation properties** and **relationship mappings**.

There are three primary types of relationships:

1. **One-to-Many (most common)**
2. **One-to-One**
3. **Many-to-Many**

EF Core can automatically infer relationships through naming conventions and navigation properties, or you can define them explicitly using the **Fluent API**.

---

### 🧩 7.1 One-to-Many Relationship

**Scenario:**
A `Blog` has many `Posts`.
A `Post` belongs to one `Blog`.

**Example:**

```csharp
public class Blog
{
    public int BlogId { get; set; }
    public string Url { get; set; }

    // Navigation Property
    public List<Post> Posts { get; set; } = new();
}

public class Post
{
    public int PostId { get; set; }
    public string Title { get; set; }

    // Foreign Key
    public int BlogId { get; set; }

    // Navigation Property
    public Blog Blog { get; set; }
}
```

**EF Core automatically detects:**

* `Blog.Posts` = one-to-many navigation
* `Post.BlogId` = foreign key

---

**Add migration and update DB:**

```bash
dotnet ef migrations add AddBlogPostRelation
dotnet ef database update
```

**Insert Data Example:**

```csharp
using var db = new AppDbContext();

var blog = new Blog
{
    Url = "https://efcore.dev",
    Posts = new List<Post>
    {
        new Post { Title = "Intro to EF Core" },
        new Post { Title = "Understanding LINQ" }
    }
};

db.Blogs.Add(blog);
db.SaveChanges();
```

**Query with Include:**

```csharp
var blogs = db.Blogs.Include(b => b.Posts).ToList();
```

---

### 🧩 7.2 One-to-One Relationship

**Scenario:**
Each `User` has one `Profile`.

**Example:**

```csharp
public class User
{
    public int UserId { get; set; }
    public string Username { get; set; }
    public UserProfile Profile { get; set; }
}

public class UserProfile
{
    public int UserProfileId { get; set; }
    public string Bio { get; set; }

    // Foreign Key
    public int UserId { get; set; }

    public User User { get; set; }
}
```

**Fluent API (optional for clarity):**

```csharp
protected override void OnModelCreating(ModelBuilder modelBuilder)
{
    modelBuilder.Entity<User>()
        .HasOne(u => u.Profile)
        .WithOne(p => p.User)
        .HasForeignKey<UserProfile>(p => p.UserId);
}
```

---

### 🧩 7.3 Many-to-Many Relationship (EF Core 5+)

**Scenario:**
`Post` can have many `Tags`, and each `Tag` can belong to many `Posts`.

**Example:**

```csharp
public class Tag
{
    public int TagId { get; set; }
    public string Name { get; set; }

    public List<Post> Posts { get; set; } = new();
}

public class Post
{
    public int PostId { get; set; }
    public string Title { get; set; }

    public List<Tag> Tags { get; set; } = new();
}
```

EF Core automatically creates a **join table** (`PostTag`) without needing an explicit model.

**Insert Data Example:**

```csharp
using var db = new AppDbContext();

var post = new Post { Title = "EF Relationships" };
var tag = new Tag { Name = "EFCore" };

post.Tags.Add(tag);
db.Posts.Add(post);
db.SaveChanges();
```

**Query Example:**

```csharp
var postsWithTags = db.Posts
    .Include(p => p.Tags)
    .ToList();
```

---

### 🧩 7.4 Relationship Loading Options

| Loading Type | Description                                             | Example                                           |
| ------------ | ------------------------------------------------------- | ------------------------------------------------- |
| **Eager**    | Load related data immediately                           | `.Include(b => b.Posts)`                          |
| **Explicit** | Load on demand via `Entry()`                            | `db.Entry(blog).Collection(b => b.Posts).Load();` |
| **Lazy**     | Load automatically when accessed (requires proxy setup) | Enable lazy loading proxies                       |

---

### 🧩 7.5 Enable Lazy Loading (Optional)

Install:

```bash
dotnet add package Microsoft.EntityFrameworkCore.Proxies
```

Configure:

```csharp
protected override void OnConfiguring(DbContextOptionsBuilder options)
{
    options
        .UseSqlite("Data Source=EfRelations.db")
        .UseLazyLoadingProxies();
}
```

Make navigation properties **virtual**:

```csharp
public virtual List<Post> Posts { get; set; }
```

---

### ⚡ Common Mistakes

| Problem                                 | Cause                                          | Solution                                                          |
| --------------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------- |
| Infinite loop during JSON serialization | Circular navigation (Blog → Post → Blog)       | Use DTOs or `.Select()` projection                                |
| Duplicate FK creation                   | Both Data Annotations + Fluent API define keys | Use one consistent method                                         |
| Lazy loading not working                | Missing proxy package or `virtual` keyword     | Add `Microsoft.EntityFrameworkCore.Proxies` and mark as `virtual` |

---

### 🧠 Exercises

1. **One-to-Many Practice**

   * Create `Blog` ↔ `Post` relationship.
   * Add 2 blogs, each with multiple posts.
   * Use `Include()` to list blogs and their posts.

2. **One-to-One Practice**

   * Create `User` ↔ `UserProfile` relationship.
   * Insert a few users with profiles.
   * Fetch a user and display their bio.

3. **Many-to-Many Practice**

   * Create `Post` ↔ `Tag` relationship.
   * Add multiple tags to posts.
   * Query all tags and show how many posts each tag has.

4. **Loading Practice**

   * Compare eager vs explicit vs lazy loading for `Blog` → `Post`.
   * Measure SQL queries (enable logging via `options.LogTo(Console.WriteLine)`).

---

### ✅ Summary

| Relationship | Example        | Foreign Key      | Best Use                |
| ------------ | -------------- | ---------------- | ----------------------- |
| One-to-Many  | Blog → Posts   | `Post.BlogId`    | Most business models    |
| One-to-One   | User → Profile | `Profile.UserId` | User metadata, configs  |
| Many-to-Many | Post ↔ Tag     | auto join table  | Tags, categories, roles |
