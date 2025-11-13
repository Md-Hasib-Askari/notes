## 🧩 Step 3: Code-First vs Database-First Approaches

### 📘 Notes

Entity Framework supports **two main workflows** for handling your database and models:

---

### 🧱 3.1 Code-First Approach

**Definition:**
You write your **C# entity classes first**, and EF Core **creates and manages the database schema** from your models.

**Workflow:**

1. Define your entity classes and `DbContext`.
2. Run `dotnet ef migrations add <MigrationName>`.
3. Apply migrations to create/update the database.

**Advantages:**

* Great for new projects.
* Version control for database schema (via migrations).
* Changes to models are easy to track and evolve.

**Disadvantages:**

* Risk of accidental schema drift if migrations aren’t used properly.
* Not ideal for legacy or shared databases.

---

### 💻 Example — Code-First

```csharp
public class Author
{
    public int AuthorId { get; set; }
    public string Name { get; set; }
    public List<Book> Books { get; set; } = new();
}

public class Book
{
    public int BookId { get; set; }
    public string Title { get; set; }
    public int AuthorId { get; set; }
    public Author Author { get; set; }
}

public class LibraryContext : DbContext
{
    public DbSet<Author> Authors { get; set; }
    public DbSet<Book> Books { get; set; }

    protected override void OnConfiguring(DbContextOptionsBuilder options)
        => options.UseSqlite("Data Source=Library.db");
}
```

**Migrate:**

```bash
dotnet ef migrations add CreateLibrarySchema
dotnet ef database update
```

EF Core creates the SQLite DB and tables automatically.

---

### 🗄️ 3.2 Database-First Approach

**Definition:**
You start with an **existing database**, and EF Core **generates entity classes and DbContext** based on that schema.

**Workflow:**

1. Point EF Core to the database.
2. Scaffold models using the EF CLI.
3. Use the generated classes in your app.

**Command:**

```bash
dotnet ef dbcontext scaffold "Your_Connection_String" Microsoft.EntityFrameworkCore.SqlServer --output-dir Models
```

You’ll get:

* A `DbContext` class (with `DbSet<T>` properties).
* Entity classes for each table.

**Advantages:**

* Perfect for existing databases.
* Automatically maps complex schemas.

**Disadvantages:**

* Harder to customize or rename entities.
* Regeneration may overwrite changes.

---

### 💡 Example — Database-First Scaffold (SQL Server)

```bash
dotnet ef dbcontext scaffold "Server=(localdb)\\mssqllocaldb;Database=MyExistingDB;Trusted_Connection=True;" Microsoft.EntityFrameworkCore.SqlServer --output-dir Models
```

Generated files include:

* `MyExistingDbContext.cs`
* `Customer.cs`
* `Order.cs`, etc.

Then you can immediately use them:

```csharp
using var db = new MyExistingDbContext();
var customers = db.Customers.ToList();
foreach (var c in customers)
    Console.WriteLine($"{c.CustomerId} - {c.Name}");
```

---

### ⚖️ 3.3 Comparison Summary

| Feature        | Code-First             | Database-First                 |
| -------------- | ---------------------- | ------------------------------ |
| Starting Point | C# Classes             | Existing Database              |
| Schema Control | Via Migrations         | Via Database                   |
| Best For       | New Projects           | Legacy Databases               |
| Flexibility    | High                   | Moderate                       |
| Tooling        | `dotnet ef migrations` | `dotnet ef dbcontext scaffold` |

---

### 🧠 Exercises

1. **Code-First Task:**
   Create a new `Product` and `Category` model.

   * Each Product belongs to a Category.
   * Add a migration and apply it.
   * Insert sample data and query all products with category names.

2. **Database-First Task:**

   * Create a small SQL Server DB manually with tables `Students` and `Courses`.
   * Scaffold it using EF CLI.
   * List all students and their courses using LINQ.

3. **Reflection Task:**
   Write down when you’d use Code-First vs Database-First in a real-world project (e.g., startup project vs legacy enterprise app).
