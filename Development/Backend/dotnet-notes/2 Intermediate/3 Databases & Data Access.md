# Databases & Data Access (Intermediate Notes)

## 1. ADO.NET Basics

* **ADO.NET**: Core library for database connectivity.
* Uses `SqlConnection`, `SqlCommand`, `SqlDataReader`, `SqlDataAdapter`.

  ```csharp
  using (SqlConnection conn = new SqlConnection(connectionString))
  {
      conn.Open();
      SqlCommand cmd = new SqlCommand("SELECT * FROM Users", conn);
      SqlDataReader reader = cmd.ExecuteReader();
      while (reader.Read())
          Console.WriteLine(reader["Name"]);
  }
  ```
* Low-level, efficient but requires more boilerplate code.

---

## 2. Entity Framework Core (ORM)

* **ORM (Object-Relational Mapper)** for .NET.
* Maps database tables to C# classes.
* Simplifies queries, inserts, updates, and deletes.

  ```csharp
  public class User
  {
      public int Id { get; set; }
      public string Name { get; set; }
  }

  public class AppDbContext : DbContext
  {
      public DbSet<User> Users { get; set; }
  }
  ```

---

## 3. CRUD Operations

* **Create**:

  ```csharp
  db.Users.Add(new User { Name = "Hasib" });
  db.SaveChanges();
  ```
* **Read**:

  ```csharp
  var users = db.Users.ToList();
  ```
* **Update**:

  ```csharp
  var user = db.Users.First();
  user.Name = "Updated";
  db.SaveChanges();
  ```
* **Delete**:

  ```csharp
  db.Users.Remove(user);
  db.SaveChanges();
  ```

---

## 4. LINQ-to-SQL

* Write SQL-like queries directly in C# using LINQ.

  ```csharp
  var results = db.Users.Where(u => u.Name.StartsWith("H"));
  ```
* Translates LINQ expressions into SQL queries.

---

## 5. Migrations & Code First Approach

* **Code First**: Define models in C#, EF generates database.
* **Migrations**: Track and apply schema changes.

  ```bash
  dotnet ef migrations add InitialCreate
  dotnet ef database update
  ```
* Keeps database schema in sync with code.
