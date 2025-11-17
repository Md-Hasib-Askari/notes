# ✅ **Module 14 — Advanced Architecture (Layered Architecture, Clean Architecture, SOLID)**

You'll learn how to structure your MVC project like a professional system:

✔ Layered Architecture
✔ Service + Repository Pattern
✔ Clean Architecture
✔ SOLID principles
✔ Separation of concerns
✔ True enterprise folder structure
✔ How big teams organize code

Let’s get into it.

---

# 🔥 1. The Problem With Typical MVC Apps

Beginners often put everything in:

* Controllers
* Models
* Views

This becomes impossible to maintain once the app grows.

Common issues:

* Fat controllers
* EF logic everywhere
* Impossible to test
* Hard to refactor
* Business logic mixed with UI logic

This is where **architecture** comes in.

---

# 🧱 2. Layered Architecture (Industry Standard)

The most commonly used structure:

```
Presentation Layer        → MVC Controllers + Views
Business Logic Layer      → Services (application logic)
Data Access Layer         → Repositories + DbContext
Domain Layer              → Entities + Interfaces
```

### MVC should only handle:

* Routing
* Input/Output
* Calling services

Nothing else.

---

# 🏗 3. Folder Structure (Professional-Grade)

A real enterprise MVC solution will look like:

```
/YourApp.Web
   /Controllers
   /Views
   /ViewModels
   /Filters
   /Middlewares
   /Assets

/YourApp.Core (Domain Layer)
   /Entities
   /Interfaces
   /DTOs

/YourApp.Services (Business Layer)
   /Services
   /ServiceInterfaces
   /Helpers

/YourApp.Data (Data Layer)
   /DbContext
   /Repositories
   /Migrations

/YourApp.Infrastructure
   /Logging
   /Email
   /Cache
   /ExternalAPIs
```

This is how large teams maintain huge systems.

---

# 🔄 4. Service Layer (Business Logic)

Every controller should call a **service**, not the database.

Example:

```csharp
public interface IStudentService
{
    Student GetById(int id);
    void Create(Student student);
    List<Student> GetAll();
}
```

Implementation:

```csharp
public class StudentService : IStudentService
{
    private readonly IStudentRepository _repo;

    public StudentService(IStudentRepository repo)
    {
        _repo = repo;
    }

    public Student GetById(int id) => _repo.Get(id);

    public void Create(Student student)
    {
        // Business rules
        _repo.Add(student);
    }

    public List<Student> GetAll() => _repo.GetAll();
}
```

---

# 🧲 5. Repository Layer (Clean DB Access)

```csharp
public interface IStudentRepository
{
    Student Get(int id);
    List<Student> GetAll();
    void Add(Student student);
}
```

Implementation:

```csharp
public class StudentRepository : IStudentRepository
{
    private readonly AppDbContext _db;

    public StudentRepository(AppDbContext db)
    {
        _db = db;
    }

    public Student Get(int id) => _db.Students.Find(id);

    public List<Student> GetAll() => _db.Students.ToList();

    public void Add(Student student)
    {
        _db.Students.Add(student);
        _db.SaveChanges();
    }
}
```

---

# 🎯 6. Clean Architecture (Advanced)

A step above layered architecture.

Rules:

1. **Core layer** (Entities, abstractions) depends on nothing.
2. Business logic depends only on Core.
3. Data layer depends on Core.
4. Web/MVC depends on everything — but nothing depends on it.

```
Entities → Use Cases → Interface Adapters → Frameworks & UI
```

MVC sits at the **outermost** layer.

Business logic is at the center.

This makes the system:

* Extremely maintainable
* Testable
* Flexible
* Independent of UI / DB tools

---

# ⚙️ 7. SOLID Principles (Foundation of All Good Architecture)

## 🟩 **S — Single Responsibility**

Each class does one thing.

Bad:

```csharp
StudentController handles DB logic + business rules.
```

Good:

* Controller handles HTTP
* Service handles business rules
* Repository handles DB

---

## 🟦 **O — Open/Closed**

Class is open for extension, closed for modification.

Use interfaces to replace parts easily.

---

## 🟨 **L — Liskov Substitution**

If a class uses an interface, it should work with any implementation.

---

## 🟪 **I — Interface Segregation**

Break large interfaces into smaller ones.

---

## 🟥 **D — Dependency Inversion**

High-level modules depend on abstractions, not concrete classes.

Controller depends on:

```csharp
IStudentService
```

—not—

```csharp
StudentService
```

This is the heart of clean architecture.

---

# 🧱 8. Clean Controller (Using DI + Services)

```csharp
public class StudentController : Controller
{
    private readonly IStudentService _service;

    public StudentController(IStudentService service)
    {
        _service = service;
    }

    public ActionResult Index()
    {
        var students = _service.GetAll();
        return View(students);
    }
}
```

Controller is now:

* Thin
* Clean
* Testable
* Maintainable

---

# 🏆 9. Industry Best Practices

🔥 Controllers must be thin
🔥 Business logic in services
🔥 Database logic in repositories
🔥 Views use ViewModels
🔥 Domain models stay clean
🔥 Use dependency injection everywhere
🔥 Avoid static classes
🔥 Avoid God classes with too many responsibilities
🔥 Validate at the service layer
🔥 Use async DB operations
🔥 Keep services stateless
🔥 Divide your project into multiple layers

---

# 🧪 Mini Task — Build a Full Clean Architecture Module

Create:

### Domain Layer

* `Product` entity
* `IProductRepository`

### Data Layer

* `ProductRepository`
* EF DbContext

### Service Layer

* `IProductService`
* `ProductService`

### Presentation Layer (MVC)

* `ProductController` using DI

This will give you a real enterprise experience.

---