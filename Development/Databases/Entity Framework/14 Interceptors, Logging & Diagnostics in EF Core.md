## 🧩 Step 14: Interceptors, Logging & Diagnostics in EF Core

### 📘 Overview

EF Core gives you full visibility and control over what happens **under the hood**.
You can:

1. Log database queries and performance data.
2. Intercept EF Core operations (commands, connections, saves).
3. Analyze metrics and diagnose performance bottlenecks.

These tools are essential in production apps — especially for debugging, monitoring, and enforcing custom policies.

---

## ⚙️ 14.1 Logging SQL Commands

EF Core integrates seamlessly with `ILoggerFactory` (from Microsoft.Extensions.Logging).
You can use this to log **executed SQL**, **execution time**, and **warnings**.

### 💡 Example — Console Logging

```csharp
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;

public class AppDbContext : DbContext
{
    public static readonly ILoggerFactory MyLoggerFactory =
        LoggerFactory.Create(builder => { builder.AddConsole(); });

    protected override void OnConfiguring(DbContextOptionsBuilder options)
    {
        options
            .UseSqlite("Data Source=EfLoggingDemo.db")
            .UseLoggerFactory(MyLoggerFactory)
            .EnableSensitiveDataLogging(); // ⚠️ Use only in dev!
    }

    public DbSet<User> Users { get; set; }
}
```

**What you’ll see:**

```
info: Microsoft.EntityFrameworkCore.Database.Command[20101]
      Executed DbCommand (2ms) [Parameters=[], CommandType='Text', CommandTimeout='30']
      SELECT "u"."Id", "u"."Name"
      FROM "Users" AS "u"
```

✅ Tip:

* `EnableSensitiveDataLogging()` includes parameter values — great for debugging, **never use it in production**.

---

## 🧩 14.2 Filtering Log Categories

You can log only specific EF Core events to avoid clutter:

```csharp
builder.AddFilter((category, level) =>
    category == DbLoggerCategory.Database.Command.Name &&
    level == LogLevel.Information);
```

This filters out all but SQL command logs.

---

## 🧩 14.3 ToQueryString() — See Generated SQL Instantly

When debugging queries, view EF’s translated SQL without executing:

```csharp
var query = db.Users.Where(u => u.Name.Contains("Hasib"));
Console.WriteLine(query.ToQueryString());
```

Output:

```sql
SELECT "u"."Id", "u"."Name"
FROM "Users" AS "u"
WHERE "u"."Name" LIKE '%' || @__p_0 || '%'
```

✅ Super useful for debugging LINQ-to-SQL translation issues.

---

## 🧩 14.4 Interceptors — Hook into EF Core’s Internals

**Interceptors** allow you to execute logic *before* or *after* EF Core actions.
You can intercept:

* Command execution (SQL)
* Connection open/close
* SaveChanges events
* Transaction lifecycle

They are like **middleware** for EF Core.

---

### 💡 Example — Command Interceptor

```csharp
using Microsoft.EntityFrameworkCore.Diagnostics;
using System.Diagnostics;

public class QueryInterceptor : DbCommandInterceptor
{
    public override InterceptionResult<DbDataReader> ReaderExecuting(
        DbCommand command,
        CommandEventData eventData,
        InterceptionResult<DbDataReader> result)
    {
        Console.WriteLine($"🔍 SQL: {command.CommandText}");
        return base.ReaderExecuting(command, eventData, result);
    }

    public override async ValueTask<InterceptionResult<DbDataReader>> ReaderExecutingAsync(
        DbCommand command,
        CommandEventData eventData,
        InterceptionResult<DbDataReader> result,
        CancellationToken cancellationToken = default)
    {
        var sw = Stopwatch.StartNew();
        var response = await base.ReaderExecutingAsync(command, eventData, result, cancellationToken);
        sw.Stop();

        Console.WriteLine($"⚡ Query took {sw.ElapsedMilliseconds} ms");
        return response;
    }
}
```

**Register it in your context:**

```csharp
protected override void OnConfiguring(DbContextOptionsBuilder options)
{
    options
        .AddInterceptors(new QueryInterceptor())
        .UseSqlite("Data Source=EfInterceptors.db");
}
```

✅ This logs every executed SQL and how long it took — excellent for profiling slow queries.

---

## 🧩 14.5 SaveChanges Interceptor (Audit Trail)

Intercept save events to track changes — e.g., auto-fill audit columns.

```csharp
public class AuditInterceptor : SaveChangesInterceptor
{
    public override InterceptionResult<int> SavingChanges(
        DbContextEventData eventData,
        InterceptionResult<int> result)
    {
        var context = eventData.Context;

        foreach (var entry in context.ChangeTracker.Entries()
                     .Where(e => e.State == EntityState.Added))
        {
            if (entry.Entity is IAuditable entity)
            {
                entity.CreatedAt = DateTime.UtcNow;
                entity.CreatedBy = "System";
            }
        }

        return base.SavingChanges(eventData, result);
    }
}

public interface IAuditable
{
    DateTime CreatedAt { get; set; }
    string CreatedBy { get; set; }
}
```

**Register:**

```csharp
options.AddInterceptors(new AuditInterceptor());
```

Now all new entities that implement `IAuditable` will automatically get timestamps.

---

## 🧩 14.6 Diagnostic Listeners (Low-Level Monitoring)

For deep monitoring and integration with tools like Application Insights, EF Core supports **DiagnosticSource**.

```csharp
DiagnosticListener.AllListeners.Subscribe(new EfDiagnosticListener());
```

You can monitor events such as:

* Command executed
* Connection opened/closed
* Transaction committed
* SaveChanges started/completed

These hooks are often used for **telemetry dashboards** and **APM integrations**.

---

## ⚡ Common Logging Levels

| Level         | Meaning                           | When to Use           |
| ------------- | --------------------------------- | --------------------- |
| `Trace`       | Very detailed EF internals        | Debugging performance |
| `Debug`       | Query and change tracking details | Local dev             |
| `Information` | Executed commands                 | Production            |
| `Warning`     | Potential issues                  | Always on             |
| `Error`       | Operation failed                  | Always on             |

---

## 🧠 Exercises

1. **SQL Logging**

   * Enable console logging for EF queries.
   * Run a few LINQ queries and observe SQL output.

2. **Command Interceptor**

   * Create an interceptor that logs queries slower than 200 ms.

3. **Audit Interceptor**

   * Add a SaveChangesInterceptor that auto-fills `CreatedAt` and `UpdatedAt` fields.

4. **Performance Monitoring**

   * Add timing logic in your interceptor and log average query duration.

5. **ToQueryString Practice**

   * Take 3 LINQ queries and print their generated SQL using `.ToQueryString()`.

---

### ✅ Summary

| Feature                          | Description             | Use Case                     |
| -------------------------------- | ----------------------- | ---------------------------- |
| **ILoggerFactory**               | Built-in logging system | Console or file logs         |
| **EnableSensitiveDataLogging()** | Show parameter values   | Debugging only               |
| **Interceptors**                 | Hook into EF events     | Custom logic, audit, metrics |
| **SaveChangesInterceptor**       | Capture DB save events  | Audit trail, timestamps      |
| **DiagnosticSource**             | Deep event stream       | App Insights, telemetry      |
| **ToQueryString()**              | View generated SQL      | Debug queries                |

---

**Key Takeaways:**

* Logging gives visibility; interceptors give control.
* Use interceptors for auditing, validation, and performance metrics.
* Use logging filters to target only what matters.
* Never log sensitive data in production.
