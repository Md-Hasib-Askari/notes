## 🧩 Step 13: Concurrency Control in EF Core

### 📘 Notes

**Concurrency control** ensures that two or more users updating the same record **don’t overwrite each other’s changes** accidentally.

EF Core primarily supports **Optimistic Concurrency**, which assumes conflicts are rare — and only checks for them before committing changes.

If a conflict is detected, EF Core throws a `DbUpdateConcurrencyException`.

---

## 🧩 13.1 The Problem — Lost Updates

Imagine this scenario:

1. User A loads a product (`Price = 100`).
2. User B loads the same product (`Price = 100`).
3. User A changes it to `90` and saves.
4. User B changes it to `80` and saves.

Result: **User B’s save overwrote A’s update** — A’s change is lost.

EF Core prevents this using concurrency tokens.

---

## ⚙️ 13.2 Concurrency Tokens

A **concurrency token** is a column EF uses to detect if data has changed since it was read.

Common patterns:

* **Timestamp / RowVersion** (recommended)
* **LastUpdated** datetime
* **Manual token (e.g., Guid)**

---

### 💡 Example — RowVersion (Timestamp) Concurrency

```csharp
public class Product
{
    public int ProductId { get; set; }
    public string Name { get; set; }
    public decimal Price { get; set; }

    [Timestamp]
    public byte[] RowVersion { get; set; }
}
```

EF Core automatically:

* Reads the current `RowVersion` when loading the entity.
* Adds it to the `WHERE` clause during updates/deletes.
* Throws an exception if another update modified it first.

**Generated SQL Example:**

```sql
UPDATE Products
SET Price = 90.00
WHERE ProductId = 1 AND RowVersion = 0x00000000000003E8;
```

If no rows are affected → EF detects that the row was changed by someone else → concurrency exception.

---

### 💡 Example — Fluent API Equivalent

```csharp
protected override void OnModelCreating(ModelBuilder modelBuilder)
{
    modelBuilder.Entity<Product>()
        .Property(p => p.RowVersion)
        .IsRowVersion();
}
```

---

## 🧩 13.3 Handling Concurrency Exceptions

When EF detects a conflict, it throws `DbUpdateConcurrencyException`.
You can **catch and resolve** it manually.

**Example:**

```csharp
using var db = new AppDbContext();
var product = db.Products.First();

product.Price += 10;

try
{
    db.SaveChanges();
}
catch (DbUpdateConcurrencyException ex)
{
    Console.WriteLine("⚠️ Concurrency conflict detected!");

    foreach (var entry in ex.Entries)
    {
        if (entry.Entity is Product)
        {
            var databaseValues = entry.GetDatabaseValues();
            var databasePrice = (decimal)databaseValues["Price"];

            Console.WriteLine($"Database Price: {databasePrice}, Your Price: {product.Price}");

            // Option 1: Refresh entity
            entry.OriginalValues.SetValues(databaseValues);

            // Option 2: Resolve conflict manually
            product.Price = (product.Price + databasePrice) / 2; // average
        }
    }

    db.SaveChanges(); // retry after resolution
}
```

---

## 🧩 13.4 Custom Concurrency Token

If you prefer a simpler setup, use a custom token like `UpdatedAt`.

```csharp
public class Post
{
    public int PostId { get; set; }
    public string Title { get; set; }
    public DateTime UpdatedAt { get; set; }
}
```

Fluent API:

```csharp
modelBuilder.Entity<Post>()
    .Property(p => p.UpdatedAt)
    .IsConcurrencyToken();
```

Now EF will compare the `UpdatedAt` timestamp before updating the record.

---

## 🧩 13.5 Multiple Concurrency Tokens

You can use **multiple fields** for concurrency checks — e.g., both `RowVersion` and `UpdatedAt`:

```csharp
modelBuilder.Entity<Product>()
    .Property(p => p.RowVersion).IsRowVersion();

modelBuilder.Entity<Product>()
    .Property(p => p.UpdatedAt).IsConcurrencyToken();
```

---

## 🧩 13.6 Concurrency in Web Applications

In web APIs, concurrency conflicts often occur when users edit the same record from different clients.
Common strategies:

| Strategy                          | Description                                  |
| --------------------------------- | -------------------------------------------- |
| **Client Wins (Last Write Wins)** | Overwrite DB with client’s data              |
| **Store Wins (Database Wins)**    | Refresh client data from DB                  |
| **Merge Changes**                 | Combine client and DB data intelligently     |
| **Ask User**                      | Prompt user to resolve manually (UI pattern) |

Example using “Store Wins”:

```csharp
entry.OriginalValues.SetValues(entry.GetDatabaseValues());
await db.SaveChangesAsync();
```

---

## 🧩 13.7 Detecting Conflicts in APIs

When exposing APIs, include the concurrency token in responses (e.g., `RowVersion` as base64 string).
Then, require the same token in update requests:

**Response:**

```json
{
  "productId": 1,
  "price": 100.0,
  "rowVersion": "AAAAAAAAB9E="
}
```

**Update Request:**

```json
{
  "productId": 1,
  "price": 90.0,
  "rowVersion": "AAAAAAAAB9E="
}
```

On mismatch → return `409 Conflict` HTTP status code.

---

### ⚡ Common Pitfalls

| Problem                                          | Cause                                | Solution                                               |
| ------------------------------------------------ | ------------------------------------ | ------------------------------------------------------ |
| “Concurrency conflict detected” always triggered | Missing `RowVersion` in WHERE clause | Ensure `[Timestamp]` or `.IsRowVersion()`              |
| Updates overwrite silently                       | No concurrency token                 | Add `RowVersion` or `.IsConcurrencyToken()`            |
| Hard-to-resolve merge conflicts                  | Multiple users edit same record      | Decide merge policy (Client Wins / Store Wins / Merge) |

---

### 🧠 Exercises

1. **Basic RowVersion Practice**

   * Add `[Timestamp]` to your `Product` entity.
   * Simulate two updates on the same product in two contexts.
   * Handle the `DbUpdateConcurrencyException`.

2. **Custom Concurrency Token**

   * Add an `UpdatedAt` field as a concurrency token.
   * Verify that EF adds it to `WHERE` clauses.

3. **API Simulation**

   * Create a small Web API endpoint for editing products.
   * Include the RowVersion in responses and require it in updates.
   * Return `409 Conflict` on mismatched RowVersion.

4. **Conflict Resolution Strategy**

   * Implement “Client Wins” and “Store Wins” strategies.
   * Compare results and note trade-offs.

---

### ✅ Summary

| Concept                          | Description                                   | Example                              |
| -------------------------------- | --------------------------------------------- | ------------------------------------ |
| **Optimistic Concurrency**       | Detects conflicts only during `SaveChanges()` | Default EF behavior                  |
| **RowVersion**                   | Binary timestamp auto-managed by DB           | `[Timestamp]`                        |
| **Concurrency Token**            | Field checked in `WHERE` clause               | `.IsConcurrencyToken()`              |
| **DbUpdateConcurrencyException** | Exception thrown when conflicts occur         | Use `GetDatabaseValues()` to resolve |
| **Resolution Strategies**        | Decide how to handle conflicts                | Client Wins / Store Wins / Merge     |

---

**Key Takeaway:**
EF Core’s concurrency control ensures safe multi-user updates by detecting conflicting changes.
Use `[Timestamp]` for production — it’s reliable, automatic, and efficient.
