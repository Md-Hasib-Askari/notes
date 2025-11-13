## 🧩 Step 16: Multi-Tenant and Sharding Strategies in EF Core

### 📘 Overview

**Multi-tenancy** means one application serves multiple clients (tenants), but each tenant’s data is isolated.

Common approaches:

1. **Database-per-tenant**
2. **Schema-per-tenant**
3. **Shared-database with tenant filters**

Each has trade-offs in cost, complexity, and scalability.

---

## ⚙️ 16.1 Models of Multi-Tenancy

| Model                                 | Description                                         | Isolation | Complexity |
| ------------------------------------- | --------------------------------------------------- | --------- | ---------- |
| **Database-per-tenant**               | Each tenant has its own physical database           | ✅ Highest | ❌ High     |
| **Schema-per-tenant**                 | Each tenant gets a dedicated schema in one database | ✅ Medium  | ⚙️ Medium  |
| **Shared-database (TenantID column)** | All tenants share tables with tenant discriminator  | ⚠️ Low    | ✅ Easiest  |

---

## 🧩 16.2 Database-per-Tenant

Each tenant’s connection string points to its own database.

**Pros:**

* Full isolation
* Easy backup/restore per tenant
* Separate scaling

**Cons:**

* Harder to manage at scale (migrations, config)
* Expensive connection handling

---

### 💡 Implementation Example

```csharp
public interface ITenantProvider
{
    string GetTenantConnectionString();
}

public class TenantProvider : ITenantProvider
{
    private readonly IHttpContextAccessor _context;

    public TenantProvider(IHttpContextAccessor context)
    {
        _context = context;
    }

    public string GetTenantConnectionString()
    {
        var tenant = _context.HttpContext?.Request.Headers["X-Tenant-ID"];
        return tenant switch
        {
            "tenantA" => "Server=.;Database=TenantA_DB;Trusted_Connection=True;",
            "tenantB" => "Server=.;Database=TenantB_DB;Trusted_Connection=True;",
            _ => throw new Exception("Unknown tenant")
        };
    }
}
```

**Configure the DbContext dynamically:**

```csharp
public class MultiTenantDbContext : DbContext
{
    private readonly ITenantProvider _tenantProvider;

    public MultiTenantDbContext(ITenantProvider tenantProvider)
    {
        _tenantProvider = tenantProvider;
    }

    protected override void OnConfiguring(DbContextOptionsBuilder options)
        => options.UseSqlServer(_tenantProvider.GetTenantConnectionString());
}
```

✅ This creates a **separate database connection per tenant** at runtime.

---

## 🧩 16.3 Shared Database with Tenant Discriminator

All tenants share the same database and tables, but each record includes a `TenantId`.

**Example Entity:**

```csharp
public class Blog
{
    public int BlogId { get; set; }
    public string Url { get; set; }
    public string TenantId { get; set; }
}
```

---

### 💡 Global Query Filters for Tenant Isolation

You can automatically filter all queries by tenant using EF Core’s **global filters**.

```csharp
public class AppDbContext : DbContext
{
    private readonly string _tenantId;

    public AppDbContext(string tenantId)
    {
        _tenantId = tenantId;
    }

    public DbSet<Blog> Blogs { get; set; }

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<Blog>().HasQueryFilter(b => b.TenantId == _tenantId);
    }
}
```

Now EF Core automatically applies `WHERE TenantId = 'xyz'` to **every query** — no need to add it manually.

**Example generated SQL:**

```sql
SELECT * FROM Blogs WHERE TenantId = 'tenant_123';
```

✅ Ensures data isolation at the EF Core level.

---

### 🧩 16.4 Inserting Tenant Data Automatically

Override `SaveChanges` to populate `TenantId` automatically:

```csharp
public override int SaveChanges()
{
    foreach (var entry in ChangeTracker.Entries()
                 .Where(e => e.State == EntityState.Added))
    {
        if (entry.Entity is Blog blog)
            blog.TenantId = _tenantId;
    }

    return base.SaveChanges();
}
```

---

### 💡 Tenant Context Example (Web API)

Inject a **TenantContext** that identifies the current tenant (from header, JWT claim, or subdomain):

```csharp
public class TenantContext
{
    public string TenantId { get; set; } = string.Empty;
}
```

Register it in `Program.cs`:

```csharp
builder.Services.AddScoped<TenantContext>();
builder.Services.AddScoped<AppDbContext>(sp =>
{
    var tenant = sp.GetRequiredService<TenantContext>();
    return new AppDbContext(tenant.TenantId);
});
```

Then in your middleware or controller, assign the tenant dynamically from the request.

---

## 🧩 16.5 Schema-per-Tenant Approach

Each tenant gets its own schema (e.g., `tenant1.Blogs`, `tenant2.Blogs`) within the same database.

EF Core doesn’t directly support dynamic schemas per context, but you can modify it at runtime:

```csharp
protected override void OnModelCreating(ModelBuilder modelBuilder)
{
    var schema = _tenantId ?? "dbo";
    modelBuilder.Entity<Blog>().ToTable("Blogs", schema);
}
```

✅ Works well if tenants are moderate in number and you need strong isolation without full separate DBs.

---

## 🧩 16.6 Sharding Strategy (for Scale-Out)

**Sharding** = partitioning data across multiple databases or servers based on key (like `TenantId`, region, or date).

Example:

* Tenant A → Shard 1
* Tenant B → Shard 2
* Tenant C → Shard 3

Sharding improves scalability by reducing DB load per shard.

---

### 💡 Sharding Example Pattern

```csharp
public class ShardResolver
{
    public string ResolveShard(string tenantId)
    {
        return tenantId switch
        {
            "tenantA" => "Server=.;Database=Shard1;Trusted_Connection=True;",
            "tenantB" => "Server=.;Database=Shard2;Trusted_Connection=True;",
            _ => throw new Exception("No shard found")
        };
    }
}
```

Use this in your context factory:

```csharp
public class ShardedDbContextFactory
{
    private readonly ShardResolver _resolver;

    public ShardedDbContextFactory(ShardResolver resolver)
    {
        _resolver = resolver;
    }

    public AppDbContext Create(string tenantId)
    {
        var connString = _resolver.ResolveShard(tenantId);
        return new AppDbContext(connString);
    }
}
```

---

## 🧩 16.7 Multi-Tenant Migrations

For multi-tenant systems, migrations can be tricky. You can:

1. Run migrations **per-tenant** dynamically (for DB-per-tenant).
2. Apply migrations **once globally** (for shared DBs).

Example:

```bash
dotnet ef database update --connection "Server=.;Database=TenantA_DB"
dotnet ef database update --connection "Server=.;Database=TenantB_DB"
```

Or write a script:

```bash
foreach ($tenant in Get-Content tenants.txt) {
  dotnet ef database update --connection "Server=.;Database=$tenant"
}
```

---

## ⚡ Choosing the Right Model

| Strategy                | Isolation      | Scale     | Complexity | Ideal For                         |
| ----------------------- | -------------- | --------- | ---------- | --------------------------------- |
| **Database-per-tenant** | High           | High      | High       | Enterprise SaaS, large clients    |
| **Schema-per-tenant**   | Medium         | Medium    | Medium     | Regional data, moderate tenants   |
| **Shared DB**           | Low            | Very High | Low        | Small tenants, cost-sensitive     |
| **Sharding**            | High (logical) | Very High | High       | Large-scale systems (>1M tenants) |

---

## 🧠 Exercises

1. **Tenant Filter Practice**

   * Add a `TenantId` to all entities.
   * Configure a global filter to auto-filter data per tenant.

2. **Tenant Injection**

   * Create a middleware that reads `X-Tenant-ID` from request headers and sets it in a `TenantContext`.

3. **Auto Tenant Assignment**

   * Override `SaveChanges()` to assign the tenant ID automatically on inserts.

4. **Database-per-Tenant**

   * Simulate multiple tenants each with their own connection string.
   * Run migrations and CRUD operations per tenant.

5. **Sharding Simulation**

   * Write a `ShardResolver` that distributes tenants across multiple SQLite databases.
   * Query data across shards dynamically.

---

### ✅ Summary

| Concept                     | Description                             | Key Mechanism           |
| --------------------------- | --------------------------------------- | ----------------------- |
| **Multi-Tenancy**           | One app, multiple isolated clients      | Separate DBs or filters |
| **Database-per-Tenant**     | Each tenant = its own DB                | Best isolation          |
| **Schema-per-Tenant**       | Same DB, multiple schemas               | Balanced                |
| **Shared-DB Tenant Filter** | One DB, filtered data                   | Simplest                |
| **Sharding**                | Data distributed by key                 | Horizontal scalability  |
| **Global Query Filter**     | Enforces tenant isolation automatically | `.HasQueryFilter()`     |

---

**Key Takeaways:**

* Multi-tenancy is about *data isolation strategy*, not just code reuse.
* Use **global query filters** for safe tenant-level filtering.
* Database-per-tenant ensures isolation but needs automation.
* Sharding is critical for scaling beyond a few thousand tenants.
* Always validate tenant boundaries — never trust client input blindly.
