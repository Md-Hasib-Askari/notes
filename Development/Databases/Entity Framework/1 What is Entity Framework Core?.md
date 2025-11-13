## 🧩 Step 1: What is Entity Framework Core?

### 📘 Notes

* **Entity Framework Core (EF Core)** is an **Object-Relational Mapper (ORM)** for .NET.
* It lets you work with databases using **C# objects instead of SQL**.
* Converts LINQ queries to SQL automatically.
* Supports many databases: SQL Server, SQLite, PostgreSQL, MySQL, etc.
* Handles **change tracking**, **schema migrations**, and **relationships** between tables.

**Advantages:**

* Less boilerplate (no manual SQL).
* Strongly typed queries (compile-time checking).
* Works cross-platform (.NET Core).

**Trade-offs:**

* Slower than raw SQL for bulk operations.
* Requires understanding of how EF translates LINQ → SQL to avoid inefficiency.

---

### 💻 Example

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
    {
        options.UseSqlServer("Server=(localdb)\\mssqllocaldb;Database=EfDemo;Trusted_Connection=True;");
    }
}

class Program
{
    static void Main()
    {
        using var db = new AppDbContext();
        Console.WriteLine("EF Core connected successfully!");
    }
}
```

Run this using:

```bash
dotnet new console -o EfDemo
cd EfDemo
dotnet add package Microsoft.EntityFrameworkCore
dotnet add package Microsoft.EntityFrameworkCore.SqlServer
dotnet add package Microsoft.EntityFrameworkCore.Design
dotnet run
```

---

### 🧠 Exercises

1. In your own words, explain what an ORM does.
2. List **two advantages** and **two trade-offs** of using EF Core.
3. Create a simple project with an entity `User` (`Id`, `Name`, `Email`) and confirm EF Core runs without error.
4. Change the database provider to **SQLite** instead of SQL Server and rerun your project.
