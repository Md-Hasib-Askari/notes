## 🧩 Step 15: Migrations in CI/CD & Deployment Pipelines

### 📘 Overview

EF Core **migrations** are great for local schema updates, but real teams need to:

* Automatically apply migrations during deployment
* Version-control and test migrations
* Generate SQL scripts for approval or manual deployment

This step teaches how to integrate migrations into **CI/CD pipelines** (e.g., GitHub Actions, Azure DevOps, GitLab CI).

---

## ⚙️ 15.1 Goals of Migration Automation

| Goal             | Description                         |
| ---------------- | ----------------------------------- |
| **Consistency**  | Same schema across all environments |
| **Safety**       | Controlled and reversible updates   |
| **Automation**   | No manual DB updates                |
| **Traceability** | Versioned, logged schema changes    |

---

## 🧩 15.2 Generating SQL Scripts for Deployment

Instead of applying migrations directly, generate a SQL script you can review or run manually.

```bash
dotnet ef migrations script -o ./migrations/ReleaseScript.sql
```

✅ This script contains **all migrations** from the initial one up to the latest.

For safer deployments, generate **idempotent scripts** (safe to re-run):

```bash
dotnet ef migrations script --idempotent -o ./migrations/DeploySafe.sql
```

---

### 💡 Why Idempotent?

Idempotent scripts check whether each migration has already been applied before running.
This makes them **safe to run repeatedly** in CI/CD without breaking production.

---

## 🧩 15.3 Applying Migrations Automatically in Code

You can apply migrations automatically on app startup (useful for small projects or dev environments):

```csharp
public static void Main(string[] args)
{
    using var db = new AppDbContext();
    db.Database.Migrate(); // Automatically applies pending migrations
}
```

> ⚠️ Only do this in **development or test environments**.
> In production, it’s safer to run migrations explicitly from the pipeline.

---

## 🧩 15.4 Running EF Core Migrations in CI/CD

You can integrate EF migration commands in your **build and deploy pipelines**.

---

### ✅ Example — GitHub Actions Workflow

```yaml
name: EF Core CI/CD
on:
  push:
    branches: [ main ]

jobs:
  build-and-migrate:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repo
        uses: actions/checkout@v4

      - name: Setup .NET
        uses: actions/setup-dotnet@v4
        with:
          dotnet-version: '8.0.x'

      - name: Restore dependencies
        run: dotnet restore

      - name: Build
        run: dotnet build --configuration Release --no-restore

      - name: Run EF Core migrations
        env:
          ConnectionStrings__DefaultConnection: ${{ secrets.DB_CONNECTION }}
        run: |
          dotnet tool install --global dotnet-ef
          dotnet ef database update --project ./YourProject.csproj
```

✅ This will:

* Build your app
* Connect to your production or staging DB
* Run `dotnet ef database update` automatically during deployment

---

### 💡 Example — Azure DevOps YAML Pipeline

```yaml
- task: DotNetCoreCLI@2
  displayName: 'Apply EF Core Migrations'
  inputs:
    command: 'custom'
    custom: 'ef'
    arguments: 'database update --project YourProject/YourProject.csproj'
```

---

## 🧩 15.5 Testing Migrations Before Deployment

To prevent failures during release:

1. **Run migrations in CI on a test database** (e.g., SQLite or local SQL Server).
2. **Verify SQL script execution** before applying to production.

Example (local test DB in CI):

```bash
dotnet ef database update --connection "Data Source=test.db"
```

---

## 🧩 15.6 Rolling Back Migrations

If something breaks after a deployment, you can rollback:

```bash
dotnet ef database update PreviousMigrationName
```

To see your migration history:

```bash
dotnet ef migrations list
```

Or generate a rollback SQL script:

```bash
dotnet ef migrations script LastGoodMigrationName CurrentMigrationName --idempotent -o rollback.sql
```

---

## 🧩 15.7 Managing Environment-Specific Databases

You can configure EF Core to connect to different databases for each environment.

**appsettings.Development.json**

```json
{
  "ConnectionStrings": {
    "DefaultConnection": "Server=localhost;Database=DevDB;Trusted_Connection=True;"
  }
}
```

**appsettings.Production.json**

```json
{
  "ConnectionStrings": {
    "DefaultConnection": "Server=sql.prod.internal;Database=ProdDB;User Id=sa;Password=secret;"
  }
}
```

**Program.cs**

```csharp
builder.Services.AddDbContext<AppDbContext>(options =>
    options.UseSqlServer(builder.Configuration.GetConnectionString("DefaultConnection")));
```

✅ In your CI/CD, set environment variables:

```yaml
env:
  ASPNETCORE_ENVIRONMENT: Production
```

---

## 🧩 15.8 Best Practices for Production Migrations

| Practice                       | Why                             |
| ------------------------------ | ------------------------------- |
| **Review SQL before applying** | Catch destructive changes early |
| **Use idempotent scripts**     | Safe re-runs in CI/CD           |
| **Back up DB before updates**  | Always keep a rollback path     |
| **Run migrations off-peak**    | Avoid downtime                  |
| **Automate test migrations**   | Detect schema issues early      |
| **Use transactions**           | Ensure migration atomicity      |

---

## 🧩 15.9 Advanced: Pre-Deployment Validation

You can test migration scripts **without applying them**:

```bash
dotnet ef migrations script --no-build
```

Or check which migrations are pending:

```bash
dotnet ef migrations list
```

---

### 🧠 Exercises

1. **Script Generation**

   * Generate a SQL migration script using `--idempotent` and inspect it.

2. **Auto Apply**

   * Add `db.Database.Migrate()` in `Program.cs`.
   * Run the app and confirm DB schema is created automatically.

3. **GitHub Actions**

   * Build a small workflow that runs `dotnet ef database update` against a test DB.

4. **Rollback Simulation**

   * Create a faulty migration and rollback to a previous one.

5. **Environment Config**

   * Configure different connection strings for dev, staging, and prod using `appsettings.{env}.json`.

---

### ✅ Summary

| Concept             | Command/Method                          | Purpose                     |
| ------------------- | --------------------------------------- | --------------------------- |
| Generate SQL script | `dotnet ef migrations script`           | Manual or CI/CD deployment  |
| Idempotent scripts  | `--idempotent`                          | Safe for multiple runs      |
| Apply in code       | `db.Database.Migrate()`                 | Auto apply migrations       |
| Rollback            | `dotnet ef database update <Migration>` | Undo last schema change     |
| CI/CD Integration   | YAML steps                              | Automate schema deployment  |
| Review SQL          | Manual review step                      | Prevent destructive changes |

---

**Key Takeaways:**

* Always version your schema through EF migrations.
* Use idempotent scripts for safe CI/CD.
* Never run raw migrations blindly in production — automate and validate them.
* Treat schema as part of your source-controlled application code.
