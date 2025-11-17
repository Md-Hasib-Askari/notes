# ✅ **Module 2 — Controllers & Routing (Beginner Level)**

Controllers are the “brain” of an MVC app. If you master routing + actions, you can control every request precisely.

---

# 🔥 1. What Is a Controller?

A **controller** is just a C# class that:

* Receives a request
* Processes logic (or calls services)
* Returns a View, JSON, or Redirect

Every controller:

* Must end with **Controller**
* Lives in `/Controllers` directory

Example:

```csharp
public class ProductController : Controller
{
}
```

---

# 🎯 2. Action Methods (The Heart of MVC)

These are the functions inside your controller that handle specific URLs.

Example:

```csharp
public ActionResult List()
{
    return View();
}
```

### Actions can return:

* `View()` → HTML Page
* `Json()` → API response
* `RedirectToAction()` → navigate
* `Content()` → plain text
* `File()` → downloads

---

# 🔗 3. Default Routing Behavior (Important!)

MVC uses the default pattern:

```
/{controller}/{action}/{id}
```

So:

* `/home/index` calls `HomeController.Index()`
* `/product/details/7` calls `ProductController.Details(7)`

If you write:

```csharp
public ActionResult Details(int id)
{
    return Content("Product ID: " + id);
}
```

Then visiting:

```
/product/details/10
```

shows:

```
Product ID: 10
```

---

# ⚡ 4. Attribute Routing (Modern + Industry Standard)

This is cleaner and more readable.

Enable in `RouteConfig`:

```csharp
routes.MapMvcAttributeRoutes();
```

Then in controller:

```csharp
[Route("products/{id}")]
public ActionResult Details(int id)
{
    return Content("ID = " + id);
}
```

Now URL becomes:

```
/products/5
```

Much more REST-friendly.

---

# 🛠 5. Passing Data to Views

Three common ways:

### 1️⃣ ViewBag (Dynamic)

```csharp
ViewBag.Name = "Hasib";
```

### 2️⃣ ViewData (Dictionary)

```csharp
ViewData["Age"] = 22;
```

### 3️⃣ Model (Strongly Typed) → **Preferred**

```csharp
return View(productModel);
```

---

# 📌 6. Controller Best Practices (Industry-Level)

🔥 **Always use services**
Controllers should **never** contain business logic.

🔥 **Keep controllers thin**
Only:

* Validate inputs
* Call service
* Return result

🔥 **Use attribute routing**
It’s cleaner and API-friendly.

🔥 **Return the smallest needed response**
Especially for mobile/web API endpoints.

🔥 **Name actions clearly**
`Edit(int id)` → good
`Process()` → bad

---

# 🧪 Mini Example — Your First REST-Style Controller

```csharp
public class BookController : Controller
{
    [Route("books")]
    public ActionResult All()
    {
        return Content("All Books");
    }

    [Route("books/{id}")]
    public ActionResult Details(int id)
    {
        return Content("Book ID: " + id);
    }

    [Route("books/new")]
    public ActionResult Create()
    {
        return View();
    }
}
```

---

# 🧩 **Exercise 2 — Build a Custom Routing Controller**

Create `StudentController` with actions:

* `/students` → List all students
* `/students/{id}` → Show student details
* `/students/create` → HTML form (empty form is fine)

Inside the list action, pass a list of student names using ViewBag.

Example:

```csharp
ViewBag.Students = new List<string> { "Arafat", "Hasib", "Sami" };
```

In the view, display them in a table.

---