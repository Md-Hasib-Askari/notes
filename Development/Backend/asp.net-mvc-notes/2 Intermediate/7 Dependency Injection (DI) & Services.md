# ✅ **Module 7 — Dependency Injection (DI) & Services (Intermediate Level)**

DI is how professional apps avoid messy code.
It keeps controllers clean, removes duplication, and enables testability.

---

# 🔥 1. What Is Dependency Injection?

Normally, you might write:

```csharp
var service = new StudentService();
```

This is **tight coupling** — bad for maintainability.

With DI:

```csharp
private readonly IStudentService _service;

public StudentController(IStudentService service)
{
    _service = service;
}
```

You don’t create the object — the framework does.

---

# 🧩 2. Where DI Is Configured?

In ASP.NET MVC 5, DI is set up manually using containers like:

* Autofac
* Unity
* Ninject
* SimpleInjector
* StructureMap

**Industry Favorite:** *Autofac* or *Unity*

---

# ⚙️ 3. Why DI Matters in Real Projects?

✔ Keep controllers thin
✔ No “new Service()” scattered everywhere
✔ Swap implementations easily
✔ Easier unit testing
✔ Better architecture

---

# 🏗 4. Creating a Service Layer (Industry Standard)

### Step 1 — Create an Interface

```csharp
public interface IStudentService
{
    List<Student> GetAll();
    Student GetById(int id);
    void Create(Student s);
}
```

---

### Step 2 — Implement the Interface

```csharp
public class StudentService : IStudentService
{
    private readonly AppDbContext _db;

    public StudentService(AppDbContext db)
    {
        _db = db;
    }

    public List<Student> GetAll() => _db.Students.ToList();

    public Student GetById(int id) => _db.Students.Find(id);

    public void Create(Student s)
    {
        _db.Students.Add(s);
        _db.SaveChanges();
    }
}
```

---

### Step 3 — Register Service With DI Container

**Autofac Example:**

```csharp
builder.RegisterType<StudentService>()
       .As<IStudentService>()
       .InstancePerLifetimeScope();

builder.RegisterType<AppDbContext>().InstancePerLifetimeScope();
```

---

# 🧠 5. Using Services in Controllers (Clean Controller)

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

Your controller is now:

* Clean
* Testable
* Maintainable

---

# 🧿 6. Repository Pattern (Optional but Common)

Many companies combine:

```
Controller → Service → Repository → DbContext
```

Repository example:

```csharp
public interface IStudentRepository
{
    List<Student> All();
    Student Find(int id);
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

    public List<Student> All() => _db.Students.ToList();
    public Student Find(int id) => _db.Students.Find(id);
}
```

Service calls repository.
Controller calls service.

Clean architecture.

---

# 🧩 7. Best Practices (Industry-Level)

🔥 **Never inject DbContext into controllers**
Always through services.

🔥 **Use interfaces everywhere**
It makes swapping implementations easy.

🔥 **Keep services stateless**
Avoid storing state inside service objects.

🔥 **Avoid service doing too much**
Follow Single Responsibility Principle.

🔥 **Use DI containers**
Do not manually instantiate dependencies.

🔥 **Follow naming conventions**

* `IStudentService`, `StudentService`
* `IStudentRepository`, `StudentRepository`

🔥 **Group services by domain**
Not by CRUD functions.

---

# 🧪 Mini Example — Clean Architecture Setup

### Domain Layer:

`Student.cs`

### Service Layer:

`IStudentService`
`StudentService`

### Controller Layer:

`StudentController`

### View Layer:

`Views/Student/Index.cshtml`

Everything is separated… very clean.

---

# 🧩 **Exercise 7 — Build a Service Layer**

Create a clean CRUD using DI:

### Domain Model:

`Employee` (Id, Name, Position, Salary)

### Required layers:

1. **IEmployeeService**
2. **EmployeeService**
3. **AppDbContext** registration
4. **EmployeeController** using DI
5. Views for Index / Create / Edit / Delete

This will level up your app architecture dramatically.

---