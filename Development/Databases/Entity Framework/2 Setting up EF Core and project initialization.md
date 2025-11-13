## ⚙️ Step 2: Setting up EF Core & Basic Project Initialization

### 📘 Notes

#### 2.1 Project Setup

1. **Create a new .NET project**

   ```bash
   dotnet new console -o EfDemo
   cd EfDemo
   ```

2. **Add EF Core packages**
   (example using SQL Server)

   ```bash
   dotnet add package Microsoft.EntityFrameworkCore
   dotnet add package Microsoft.EntityFrameworkCore.SqlServer
   dotnet add package Microsoft.EntityFrameworkCore.Design
   ```

3. **Add the EF CLI tools** (globally)

   ```bash
   dotnet tool install --global dotnet-ef
   ```

---

#### 2.2 Create the Model and Context

Your first entity and DbContext:

```csharp
// Models/Blog.cs
public class Blog
{
    public int BlogId { get; set; }
    public string Url { get; set; }
}

// Data/AppDbContext.cs
using Microsoft.EntityFrameworkCore;

public class AppDbContext : DbContext
{
    public DbSet<Blog> Blogs { get; set; }

    protected override void OnConfiguring(DbContextOptionsBuilder options)
        => options.UseSqlServer("Server=(localdb)\\mssqllocaldb;Database=EfDemo;Trusted_Connection=True;");
}
```

> 💡 `DbSet<T>` maps your C# class to a database table.

---

#### 2.3 Applying Migrations

Migrations track schema changes from your C# models to the actual database.

**Initialize the first migration:**

```bash
dotnet ef migrations add InitialCreate
dotnet ef database update
```

**What happens:**

* A `Migrations` folder is created.
* SQL commands are generated and executed automatically.
* Database + table `Blogs` is created.

---

### 💻 Example — Insert and Read Data

```csharp
class Program
{
    static void Main()
    {
        using var db = new AppDbContext();

        // Insert data
        var blog = new Blog { Url = "https://dotnet.microsoft.com" };
        db.Blogs.Add(blog);
        db.SaveChanges();

        // Read data
        var blogs = db.Blogs.ToList();
        foreach (var b in blogs)
            Console.WriteLine($"BlogId: {b.BlogId}, Url: {b.Url}");
    }
}
```

---

### 🧠 Exercises

1. Create a new entity `User` with properties: `UserId`, `Name`, `Email`.

   * Add it to your `AppDbContext`.
   * Run a migration to update the database.

2. Insert at least **3 users** in the database and print all users from the table.

3. Change a user’s name and call `db.SaveChanges()` again.

   * Observe how EF tracks changes automatically.

4. Delete one user and verify the remaining users in the database.

5. (Optional Challenge) Switch your provider from SQL Server to **SQLite**:
   Replace:

   ```csharp
   options.UseSqlServer(...)
   ```

   with:

   ```csharp
   options.UseSqlite("Data Source=EfDemo.db");
   ```
