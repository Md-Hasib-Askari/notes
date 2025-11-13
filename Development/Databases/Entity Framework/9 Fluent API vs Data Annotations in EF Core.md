## 🧩 Step 9: Fluent API vs Data Annotations in EF Core

### 📘 Notes

EF Core uses **conventions** to automatically infer table names, column names, keys, and relationships.
But when conventions aren’t enough, you can use:

1. **Data Annotations** → attributes inside your model classes
2. **Fluent API** → code-based configuration inside `OnModelCreating()` in your `DbContext`

You can use both — but the **Fluent API takes precedence** when there’s a conflict.

---

### 🧩 9.1 Data Annotations (Attribute-Based)

Attributes let you define constraints and mappings directly in your entity classes.

**Example:**

```csharp
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

public class User
{
    [Key]
    public int Id { get; set; }

    [Required, MaxLength(100)]
    public string Name { get; set; }

    [Column("EmailAddress")]
    public string Email { get; set; }

    [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    [ForeignKey("Profile")]
    public int ProfileId { get; set; }
    public UserProfile Profile { get; set; }
}
```

**Common Data Annotations:**

| Attribute                                               | Description          |
| ------------------------------------------------------- | -------------------- |
| `[Key]`                                                 | Primary key          |
| `[Required]`                                            | Not nullable         |
| `[MaxLength(n)]`, `[StringLength(n)]`                   | Field length limit   |
| `[Column("Name")]`                                      | Custom column name   |
| `[Table("Users")]`                                      | Custom table name    |
| `[ForeignKey("...")]`                                   | Define foreign key   |
| `[DatabaseGenerated(DatabaseGeneratedOption.Identity)]` | Auto-generated value |

---

### 🧩 9.2 Fluent API (Code-Based)

Fluent API gives more **control and flexibility**, especially for complex models or composite keys.

It’s defined in the `OnModelCreating()` method inside your `DbContext`.

**Example:**

```csharp
protected override void OnModelCreating(ModelBuilder modelBuilder)
{
    // Table name
    modelBuilder.Entity<User>().ToTable("AppUsers");

    // Primary key
    modelBuilder.Entity<User>().HasKey(u => u.Id);

    // Property configurations
    modelBuilder.Entity<User>()
        .Property(u => u.Name)
        .IsRequired()
        .HasMaxLength(100);

    // Column rename
    modelBuilder.Entity<User>()
        .Property(u => u.Email)
        .HasColumnName("EmailAddress");

    // Composite key
    modelBuilder.Entity<LoginHistory>()
        .HasKey(l => new { l.UserId, l.LoginTime });
}
```

---

### 🧩 9.3 Fluent API for Relationships

**One-to-Many:**

```csharp
modelBuilder.Entity<Blog>()
    .HasMany(b => b.Posts)
    .WithOne(p => p.Blog)
    .HasForeignKey(p => p.BlogId);
```

**One-to-One:**

```csharp
modelBuilder.Entity<User>()
    .HasOne(u => u.Profile)
    .WithOne(p => p.User)
    .HasForeignKey<UserProfile>(p => p.UserId);
```

**Many-to-Many:**

```csharp
modelBuilder.Entity<Post>()
    .HasMany(p => p.Tags)
    .WithMany(t => t.Posts);
```

---

### 🧩 9.4 Indexes and Constraints

You can define **indexes, unique keys, and check constraints** in Fluent API:

```csharp
modelBuilder.Entity<User>()
    .HasIndex(u => u.Email)
    .IsUnique();

modelBuilder.Entity<User>()
    .HasCheckConstraint("CK_User_Name", "length(Name) >= 3");
```

---

### 🧩 9.5 Default Values and Value Conversions

**Default value:**

```csharp
modelBuilder.Entity<User>()
    .Property(u => u.CreatedAt)
    .HasDefaultValueSql("CURRENT_TIMESTAMP");
```

**Enum conversion example:**

```csharp
modelBuilder.Entity<User>()
    .Property(u => u.Status)
    .HasConversion<string>();
```

---

### 🧩 9.6 Combining Both Approaches

You can use **Data Annotations for simple models** and **Fluent API for complex rules**.

Example mix:

```csharp
public class Product
{
    [Key] public int ProductId { get; set; }
    [Required] public string Name { get; set; }
    public decimal Price { get; set; }
}
```

```csharp
protected override void OnModelCreating(ModelBuilder modelBuilder)
{
    modelBuilder.Entity<Product>()
        .Property(p => p.Price)
        .HasPrecision(10, 2);
}
```

---

### ⚡ Comparison Table

| Feature        | Data Annotations       | Fluent API                   |
| -------------- | ---------------------- | ---------------------------- |
| Location       | Inside Model Class     | Inside `DbContext`           |
| Simplicity     | Easy for simple models | Better for complex logic     |
| Control        | Limited                | Full control                 |
| Composite Keys | ❌ Not supported        | ✅ Supported                  |
| Readability    | Inline, clean          | Centralized in configuration |

---

### 🧠 Exercises

1. **Annotation Practice**

   * Add `[Required]`, `[MaxLength]`, and `[Column]` attributes to a `Customer` model.
   * Add a migration and inspect the generated schema.

2. **Fluent API Practice**

   * Configure the same `Customer` model using Fluent API instead.
   * Add a unique index on `Email` and a default timestamp for `CreatedAt`.

3. **Composite Key Practice**

   * Create a model `OrderItem` with `OrderId` and `ProductId` as a **composite key**.
   * Define it using Fluent API.

4. **Conversion Practice**

   * Create an `enum OrderStatus { Pending, Shipped, Cancelled }`.
   * Store it as a string in the DB using a value converter.

5. **Constraint Practice**

   * Add a check constraint to ensure `Price > 0` in `Product`.

---

### ✅ Summary

| Topic                   | Fluent API | Data Annotations |
| ----------------------- | ---------- | ---------------- |
| Table name, column name | ✅          | ✅                |
| Required fields         | ✅          | ✅                |
| Composite keys          | ✅          | ❌                |
| Indexes, constraints    | ✅          | ❌                |
| Default values          | ✅          | Limited          |
| Enum conversions        | ✅          | ❌                |

---

**Key takeaway:**

* Use **Data Annotations** for simplicity.
* Use **Fluent API** when you need precision, complex relationships, or constraints.
* In large projects → prefer Fluent API to keep model classes clean and flexible.
