# ✅ **Module 19 — Testing (Unit Tests, Integration Tests, Mocking Services)**

You’ll learn:

✔ Unit testing controllers
✔ Testing services (the real business logic)
✔ Mocking repositories & dependencies
✔ Integration tests with real DB/API
✔ Test project structure
✔ Tools used in industry
✔ Best practices

Let’s get into it.

---

# 🔥 1. Why Testing Matters

Testing helps you:

* Catch bugs early
* Avoid regressions
* Improve code structure
* Make refactoring safe
* Build confidence before deploying
* Keep large systems stable

Companies *require* automated tests for scalable products.

---

# 🧱 2. Test Project Setup (Professional Pattern)

A real solution structure:

```
/YourApp.Web
/YourApp.Services
/YourApp.Data
/YourApp.Core
/YourApp.Tests  ← Test Project
```

Your test project should test:

* Services
* Repositories
* Controllers
* Utilities
* Filters

---

# 🧪 3. Testing Tools You Will Use

### Unit Testing Frameworks:

* MSTest (built-in)
* NUnit
* xUnit (most popular)

### Mocking Framework:

* **Moq** (industry standard)

Install Moq:

```
Install-Package Moq
```

---

# 🧩 4. Testing Services (Most Important)

This is where your business logic lives.

### Example Service:

```csharp
public class StudentService : IStudentService
{
    private readonly IStudentRepository _repo;

    public StudentService(IStudentRepository repo)
    {
        _repo = repo;
    }

    public Student GetById(int id)
    {
        return _repo.Get(id);
    }
}
```

### Unit Test (using xUnit + Moq):

```csharp
public class StudentServiceTests
{
    [Fact]
    public void GetById_ReturnsStudent()
    {
        // ARRANGE
        var mockRepo = new Mock<IStudentRepository>();
        mockRepo.Setup(r => r.Get(1))
                .Returns(new Student { Id = 1, Name = "Hasib" });

        var service = new StudentService(mockRepo.Object);

        // ACT
        var result = service.GetById(1);

        // ASSERT
        Assert.Equal("Hasib", result.Name);
    }
}
```

This tests **business logic**, not the database.

---

# 🧪 5. Testing Controllers (With Mocked Services)

Example controller:

```csharp
public class StudentController : Controller
{
    private readonly IStudentService _service;

    public StudentController(IStudentService service)
    {
        _service = service;
    }

    public ActionResult Details(int id)
    {
        var student = _service.GetById(id);
        return View(student);
    }
}
```

### Controller Test:

```csharp
[Fact]
public void Details_ReturnsView_WithStudent()
{
    var mockService = new Mock<IStudentService>();
    mockService.Setup(s => s.GetById(1))
               .Returns(new Student { Id = 1, Name = "Hasib" });

    var controller = new StudentController(mockService.Object);

    var result = controller.Details(1) as ViewResult;

    Assert.NotNull(result);
    Assert.Equal("Hasib", (result.Model as Student).Name);
}
```

This confirms your controller:

* Calls service
* Returns correct view
* Passes correct data

---

# 🧵 6. Integration Tests (Advanced)

Integration tests check:

* Controller + Routing + EF + DB
* Everything together

### Using In-Memory Database

Install:

```
Install-Package Effort
```

or use EF InMemory provider.

### Example:

```csharp
using (var context = new AppDbContext(CreateInMemoryOptions()))
{
    context.Students.Add(new Student { Id = 1, Name = "Hasib" });
    context.SaveChanges();

    var repo = new StudentRepository(context);
    var student = repo.Get(1);

    Assert.Equal("Hasib", student.Name);
}
```

This tests real repository behavior.

---

# 🧠 7. Mocking Database (Best Practice)

You should never use real database for unit tests.

Use Moq:

```csharp
mockRepo.Setup(r => r.GetAll())
        .Returns(new List<Student> 
        { 
           new Student { Id = 1, Name = "Hasib" } 
        });
```

Mock repositories → test services easily.

---

# 🔗 8. Testing Security (Important)

Test authorization logic:

```csharp
[Fact]
public void AdminPage_RequiresAdminRole()
{
    var controller = new AdminController();
    var attr = controller.GetType().GetCustomAttributes(true)
                         .OfType<AuthorizeAttribute>()
                         .FirstOrDefault();

    Assert.Equal("Admin", attr.Roles);
}
```

---

# 🧲 9. Testing Filters (Logging, Authorization, Validation)

Testing custom filters:

```csharp
[Fact]
public void LogFilter_LogsBeforeAction()
{
    var filter = new LogFilter();
    var context = new ActionExecutingContext(...);

    filter.OnActionExecuting(context);

    // Assert logs written...
}
```

---

# 🏆 10. Best Practices (Enterprise-Level)

🔥 Test service layer thoroughly
🔥 Mock repositories & external services
🔥 Don’t test Entity Framework itself
🔥 Keep test names descriptive
🔥 One assertion per test when possible
🔥 Use Arrange → Act → Assert structure
🔥 Test edge cases (null, empty, invalid)
🔥 Test authorization & permissions
🔥 Use integration tests for complex flows
🔥 Add tests to CI/CD pipeline

---

# 🧪 Mini Example — Full Testing Scenario

1. Create `IProductService`
2. Mock `IProductRepository`
3. Test: `GetAll()` returns list
4. Test: `GetById()` returns correct product
5. Test controller actions
6. Test error paths (product not found)
7. Add integration test with in-memory DB

This gives you real production testing experience.

---

# 🧩 **Exercise 19 — Build a Complete Testing Suite**

You must write tests for:

✔ Services
✔ Controllers
✔ Repositories
✔ Filters
✔ Authorization
✔ Async operations
✔ Error handling

Once done, your MVC system becomes **battle-tested** and ready for deployment.

---