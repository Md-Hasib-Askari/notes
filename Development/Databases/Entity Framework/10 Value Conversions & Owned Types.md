## 🧩 Step 10: Value Conversions & Owned Types

### 📘 Notes

EF Core allows you to:

1. **Convert property values** from C# type → database type (`ValueConverter`), and
2. **Embed complex objects** inside an entity (Owned Types).

These help make your code more expressive while keeping the database schema normalized and simple.

---

## 🎯 10.1 Value Conversions

A **Value Converter** defines how a .NET property is stored and retrieved from the database.

---

### 💡 Example 1 — Enum Conversion

Let’s say you have an enum:

```csharp
public enum OrderStatus
{
    Pending,
    Processing,
    Completed,
    Cancelled
}
```

By default, EF stores enums as integers.
To store them as strings:

```csharp
public class Order
{
    public int OrderId { get; set; }
    public OrderStatus Status { get; set; }
}
```

```csharp
protected override void OnModelCreating(ModelBuilder modelBuilder)
{
    modelBuilder.Entity<Order>()
        .Property(o => o.Status)
        .HasConversion<string>();
}
```

✅ Now your database will store `"Pending"`, `"Processing"`, etc., instead of `0`, `1`, `2`.

---

### 💡 Example 2 — Custom Conversion (Encrypt/Decrypt, Serialize/Deserialize)

You can store complex or sensitive data in special formats.

Example: Convert a `List<string>` to a comma-separated string.

```csharp
public class User
{
    public int Id { get; set; }
    public List<string> Roles { get; set; } = new();
}
```

```csharp
protected override void OnModelCreating(ModelBuilder modelBuilder)
{
    var converter = new ValueConverter<List<string>, string>(
        v => string.Join(",", v),
        v => v.Split(',', StringSplitOptions.RemoveEmptyEntries).ToList()
    );

    modelBuilder.Entity<User>()
        .Property(u => u.Roles)
        .HasConversion(converter);
}
```

✅ This way:

* C#: `["Admin", "Editor"]`
* DB: `"Admin,Editor"`

---

### 💡 Example 3 — Boolean Conversion

Store a boolean as `"Y"`/`"N"` in the DB:

```csharp
modelBuilder.Entity<Product>()
    .Property(p => p.IsActive)
    .HasConversion(
        v => v ? "Y" : "N",
        v => v == "Y"
    );
```

---

### ⚙️ Built-in Converters

EF Core provides many ready-made converters, such as:

* `EnumToStringConverter`
* `BoolToZeroOneConverter`
* `DateTimeToBinaryConverter`
* `ValueConverter<Guid, string>`

You can even chain converters for complex types.

---

## 🧩 10.2 Owned Types (Value Objects)

An **Owned Type** represents a value object that has **no identity of its own** and **belongs to another entity**.

### Example — Address embedded in Customer

```csharp
public class Customer
{
    public int Id { get; set; }
    public string Name { get; set; }

    public Address Address { get; set; }
}

[Owned]
public class Address
{
    public string Street { get; set; }
    public string City { get; set; }
    public string ZipCode { get; set; }
}
```

Or configure ownership via Fluent API:

```csharp
protected override void OnModelCreating(ModelBuilder modelBuilder)
{
    modelBuilder.Entity<Customer>()
        .OwnsOne(c => c.Address);
}
```

---

### 🗄️ Database Output

EF Core will create **a single table** `Customers` like this:

| Id | Name | Address_Street | Address_City | Address_ZipCode |
| -- | ---- | -------------- | ------------ | --------------- |

✅ The owned type’s properties are stored as **columns in the owner’s table**, not a separate table.

---

### 🧩 10.3 Nested Owned Types

Owned types can contain other owned types:

```csharp
public class Order
{
    public int Id { get; set; }
    public ShippingInfo Shipping { get; set; }
}

[Owned]
public class ShippingInfo
{
    public Address Address { get; set; }
    public DateTime ShippedOn { get; set; }
}
```

Configured via Fluent API:

```csharp
modelBuilder.Entity<Order>()
    .OwnsOne(o => o.Shipping)
    .OwnsOne(s => s.Address);
```

---

### 🧩 10.4 Collection of Owned Types

You can also have collections of owned types:

```csharp
public class User
{
    public int Id { get; set; }
    public List<Phone> Phones { get; set; } = new();
}

[Owned]
public class Phone
{
    public string Number { get; set; }
    public string Type { get; set; }
}
```

EF Core will create a **separate table** for `Phones`, linked by a foreign key to `User`.

---

### ⚡ Why Use Owned Types?

| Benefit               | Description                                  |
| --------------------- | -------------------------------------------- |
| Clean domain modeling | Keeps complex data grouped logically         |
| No redundant tables   | Avoids unnecessary joins                     |
| Easy maintenance      | Owned type changes propagate automatically   |
| Works well with DDD   | Ideal for Domain-Driven Design Value Objects |

---

### ⚠️ Limitations

* Owned types **cannot exist independently**.
* EF cannot share owned types between two owners.
* Owned collections require their own table.

---

### 🧠 Exercises

1. **Enum Conversion Practice**

   * Create `OrderStatus` enum and store it as a string.
   * Insert multiple orders and confirm that statuses are stored as text.

2. **Custom Converter**

   * Create a converter that stores `List<int>` as a comma-separated string.
   * Verify that reading/writing works properly.

3. **Owned Type**

   * Create `Customer` with an `Address` owned type.
   * Insert and read back customers; inspect database columns.

4. **Nested Owned Type**

   * Extend `Order` → `ShippingInfo` → `Address`.
   * Run migration and examine how EF Core flattens columns.

5. **Challenge:**

   * Add a `List<Phone>` owned collection to `User`.
   * Verify that EF creates a second table with a foreign key.

---

### ✅ Summary

| Concept              | Description                    | DB Behavior               |
| -------------------- | ------------------------------ | ------------------------- |
| **Value Converter**  | Converts property value type   | Maps property differently |
| **Owned Type**       | Embedded object (value object) | Stored in same table      |
| **Owned Collection** | List of owned types            | Stored in separate table  |
| **Enum Conversion**  | Enum stored as string          | Simplifies readability    |

---

**Key takeaway:**
Use **Value Converters** for custom serialization or transformations, and **Owned Types** to represent domain value objects cleanly inside your entities.
