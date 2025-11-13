## 🧩 Step 5: Migrations and Schema Evolution

### 📘 Notes

A **migration** in Entity Framework Core is a **record of changes made to your data model (C# classes)** that EF uses to **synchronize the database schema** with your code.

Think of it as **version control for your database**.

---

### ⚙️ 5.1 Why Migrations Are Important

* Keeps **code and database in sync**
* Supports **incremental schema updates** (no need to recreate DB each time)
* Can **rollback** to a previous schema version
* Allows **team collaboration** — everyone can apply the same DB changes safely

---

### 💻 Example Setup

Assume your context and model:

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
        => options.UseSqlite("Data Source=EfMigrationsDemo.db");
}
```

---

### 🧩 5.2 Creating the First Migration

```bash
dotnet ef migrations add InitialCreate
```

EF Core will:

* Create a new folder `/Migrations`
* Add two files:

  * `YYYYMMDDHHMMSS_InitialCreate.cs` → actual migration code
  * `AppDbContextModelSnapshot.cs` → current schema snapshot

Now update the DB:

```bash
dotnet ef database update
```

✅ This will create the SQLite database file and `Blogs` table.

---

### 🧩 5.3 Adding a New Entity and Evolving Schema

Add a new model:

```csharp
public class Post
{
    public int PostId { get; set; }
    public string Title { get; set; }
    public string Content { get; set; }

    public int BlogId { get; set; }
    public Blog Blog { get; set; }
}
```

Update your context:

```csharp
public DbSet<Post> Posts { get; set; }
```

Now, create a new migration:

```bash
dotnet ef migrations add AddPostEntity
dotnet ef database update
```

EF Core adds the new `Posts` table automatically, preserving existing data.

---

### 🧩 5.4 Viewing and Managing Migrations

**List migrations:**

```bash
dotnet ef migrations list
```

**Revert to a previous migration:**

```bash
dotnet ef database update InitialCreate
```

**Generate raw SQL for deployment (CI/CD use):**

```bash
dotnet ef migrations script
```

You can open the generated `.sql` file to inspect the exact SQL EF would execute.

---

### 🧩 5.5 Editing or Removing Migrations

If you made a mistake:

* **Remove the last migration** (before applying it):

  ```bash
  dotnet ef migrations remove
  ```

* **Edit the migration file manually** (only before applying it) — EF will recompile it when running `database update`.

---

### ⚡ Common Issues

| Problem                           | Cause                            | Solution                                           |
| --------------------------------- | -------------------------------- | -------------------------------------------------- |
| “No migrations were found.”       | Missing DbContext reference      | Add `DbContext` or specify with `--context`        |
| “Pending model changes detected.” | Model differs from last snapshot | Add a new migration                                |
| DB mismatch errors                | You manually changed DB schema   | Revert manual changes or re-sync with EF migration |

---

### 🧠 Exercises

1. **Schema Evolution Practice**

   * Add a new property `CreatedAt` (DateTime) to `Blog`.
   * Add a new entity `Author` (with `AuthorId`, `Name`).
   * Run `dotnet ef migrations add AddAuthorAndTimestamp` and `dotnet ef database update`.

2. **Rollback Practice**

   * Roll back the DB to the previous migration.
   * Verify that the `Author` table disappears.

3. **SQL Inspection**

   * Generate a migration script (`dotnet ef migrations script`) and open it.
   * Identify the SQL `CREATE TABLE` and `ALTER TABLE` statements.

4. **Challenge:**

   * Add a **foreign key** between `Post` and `Author`.
   * Generate migration and confirm foreign key constraints in the database.

---

### 💡 Summary

| Command                           | Description                         |
| --------------------------------- | ----------------------------------- |
| `dotnet ef migrations add <Name>` | Create a new migration              |
| `dotnet ef database update`       | Apply migrations to DB              |
| `dotnet ef migrations list`       | List all migrations                 |
| `dotnet ef migrations script`     | Generate SQL migration script       |
| `dotnet ef migrations remove`     | Delete last migration (not applied) |
