# ✅ **Module 6 — Entity Framework Integration (DB, Migrations, CRUD)**

This is a core module — once you understand EF + MVC together, you can build any real-world application.

---

# 🔥 1. What Is Entity Framework (EF)?

Entity Framework is Microsoft’s ORM that lets you work with the database using **C# classes instead of SQL**.

EF handles:

* Database creation
* Migrations
* CRUD
* Querying
* Relationships

EF makes development faster + safer.

---

# 🧠 2. EF Workflow in MVC

1. Create a **DbContext**
2. Create **Models** (Entities)
3. Define **DbSet<T>** for each table
4. Run **migrations**
5. Do CRUD through EF
6. Use ViewModels to send data to views

---

# 🏗 3. Step-by-Step Example — Create a Database

### Step 1 — Create Entity Model

```csharp
public class Student
{
    public int Id { get; set; }
    public string Name { get; set; }
    public string Department { get; set; }
}
```

---

### Step 2 — Create DbContext

```csharp
public class AppDbContext : DbContext
{
    public DbSet<Student> Students { get; set; }
}
```

EF will create a **Students** table automatically.

---

### Step 3 — Configure Connection String

Inside `web.config`:

```xml
<connectionStrings>
  <add name="AppDbContext"
       connectionString="Data Source=(localdb)\MSSQLLocalDB;Initial Catalog=MVCApp;Integrated Security=True;"
       providerName="System.Data.SqlClient" />
</connectionStrings>
```

---

### Step 4 — Enable Migrations

Open **Package Manager Console**:

```
Enable-Migrations
Add-Migration Initial
Update-Database
```

This builds the database.

---

# 🧰 4. Basic CRUD Operations (Real MVC Stuff)

---

## 🟩 A. Create (Insert)

```csharp
[HttpPost]
public ActionResult Create(Student model)
{
    _db.Students.Add(model);
    _db.SaveChanges();

    return RedirectToAction("Index");
}
```

---

## 🟦 B. Read (List + Details)

### List:

```csharp
public ActionResult Index()
{
    var students = _db.Students.ToList();
    return View(students);
}
```

### Details:

```csharp
public ActionResult Details(int id)
{
    var student = _db.Students.Find(id);
    return View(student);
}
```

---

## 🟨 C. Update (Edit)

```csharp
[HttpPost]
public ActionResult Edit(Student model)
{
    _db.Entry(model).State = EntityState.Modified;
    _db.SaveChanges();
    return RedirectToAction("Index");
}
```

---

## 🟥 D. Delete

```csharp
public ActionResult Delete(int id)
{
    var student = _db.Students.Find(id);
    _db.Students.Remove(student);
    _db.SaveChanges();

    return RedirectToAction("Index");
}
```

---

# 🔗 5. Include Related Data (Relationships)

### Example: Student has a Department

```csharp
public class Student
{
    public int Id { get; set; }
    public string Name { get; set; }

    public int DepartmentId { get; set; }
    public Department Department { get; set; }
}
```

Query:

```csharp
var students = _db.Students
                  .Include(s => s.Department)
                  .ToList();
```

---

# 🏆 6. Best Practices (Industry-Level)

🔥 Use **ViewModels** for forms and views
Never expose EF models directly.

🔥 Use **async EF methods**

```csharp
var data = await _db.Students.ToListAsync();
```

🔥 Avoid business logic inside controllers
Use a service layer.

🔥 Always use **migrations** to manage schema
Never modify DB manually.

🔥 Index frequently queried columns
EF won’t do this automatically.

🔥 Use `Include()` carefully
Too many includes = slow queries.

---

# 🔥 7. Professional MVC + EF Pattern (What Companies Use)

```
Controllers → Services → Repositories → DbContext
```

### Example:

* Controller calls StudentService
* StudentService calls StudentRepository
* Repository interacts with DbContext

This gives:

* Clean architecture
* Reusable logic
* Testable classes

We will reach this in advanced modules.

---

# 🧪 Mini Task — Full CRUD With EF

1. Create Entity: `Employee` (Id, Name, Salary, Age)
2. Set up DbContext & migrations
3. Build CRUD:

   * List
   * Create
   * Edit
   * Delete
4. Use ViewModels for forms
5. Add validation
6. Show results in HTML table

This will make you ready for real projects.

---