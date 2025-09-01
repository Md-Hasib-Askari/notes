# Web Development (Intermediate Notes)

## 1. ASP.NET Core Basics

* **ASP.NET Core**: Cross-platform, open-source framework for building web apps and APIs.
* Built-in dependency injection, middleware pipeline, and high performance.
* Project files:

  * **Program.cs**: Entry point, sets up host and services.
  * **Startup.cs**: Configures request pipeline and app services.

---

## 2. MVC Pattern

* **Model-View-Controller**:

  * **Model**: Represents data and business logic.
  * **View**: UI layer (Razor views).
  * **Controller**: Handles requests, interacts with model, returns views or data.
* Example:

  ```csharp
  public class HomeController : Controller
  {
      public IActionResult Index() => View();
  }
  ```

---

## 3. Razor Pages

* Simplified way to build page-focused web apps.
* Each page has a `.cshtml` file (view) and a `.cshtml.cs` file (page model).
* Example:

  ```csharp
  public class IndexModel : PageModel
  {
      public string Message { get; set; }
      public void OnGet() => Message = "Hello Razor Pages!";
  }
  ```

---

## 4. REST APIs with ASP.NET Core Web API

* Build RESTful services using controllers.
* Return JSON by default.

  ```csharp
  [ApiController]
  [Route("api/[controller]")]
  public class UsersController : ControllerBase
  {
      [HttpGet]
      public IEnumerable<string> Get() => new string[] { "Alice", "Bob" };

      [HttpPost]
      public IActionResult Create(User user)
      {
          return Ok(user);
      }
  }
  ```
* Supports routing, model binding, validation, authentication, and versioning.
