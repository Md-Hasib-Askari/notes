# ✅ **Module 10 — Routing (Advanced Routing, Attribute Routing, Custom Constraints)**

By now you've used simple routes like:

```
/students/details/5
```

But in real-world apps, URLs must be:

* Clean
* Semantic
* SEO-friendly
* API-friendly
* Easy to maintain

This module teaches you exactly that.

---

# 🔥 1. Review — Default MVC Route

Located in `RouteConfig.cs`:

```csharp
routes.MapRoute(
    name: "Default",
    url: "{controller}/{action}/{id}",
    defaults: new { controller = "Home", action = "Index", id = UrlParameter.Optional }
);
```

This means:

```
/home/index
/product/details/5
```

---

# 🟦 2. Attribute Routing (Modern, Recommended)

First, enable it:

```csharp
routes.MapMvcAttributeRoutes();
```

Now you can decorate controllers:

```csharp
[Route("products/{id}")]
public ActionResult Details(int id)
{
    ...
}
```

This gives you **full control**.

---

# 🏷 3. Attribute Routing — Clean REST URLs

### List all:

```csharp
[Route("products")]
public ActionResult Index()
{
    ...
}
```

### Get details:

```csharp
[Route("products/{id:int}")]
public ActionResult Details(int id)
{
    ...
}
```

### Create:

```csharp
[Route("products/new")]
public ActionResult Create()
{
    ...
}
```

These URLs are what **real companies** use.

---

# 🧱 4. Route Constraints (Important!)

You can force a route to accept only certain values.

### Numeric:

```csharp
[Route("students/{id:int}")]
```

### GUID:

```csharp
[Route("files/{id:guid}")]
```

### Min/max length:

```csharp
[Route("blog/{slug:maxlength(80)}")]
```

### Regex:

```csharp
[Route("user/{username:regex(^[a-zA-Z0-9_]+$)}")]
```

---

# 🔐 5. HTTP Method Constraints

```csharp
[HttpGet]
[Route("api/products")]
public ActionResult GetProducts()
```

```csharp
[HttpPost]
[Route("api/products")]
public ActionResult AddProduct()
```

This is perfect for API-style development.

---

# 🔄 6. Optional Parameters & Defaults

```csharp
[Route("products/{category?}")]
public ActionResult List(string category = "all")
{
    ...
}
```

If category not provided → defaults to "all".

---

# 🔀 7. Multiple Routes for One Action

```csharp
[Route("students/{id}")]
[Route("pupils/{id}")]
public ActionResult Details(int id)
{
    ...
}
```

Useful during migrations or legacy support.

---

# 🧩 8. Route Prefix (Group URLs)

Applied at controller level:

```csharp
[RoutePrefix("admin/products")]
public class AdminProductController : Controller
{
    [Route("")]
    public ActionResult Index() {}

    [Route("create")]
    public ActionResult Create() {}
}
```

Generates:

```
/admin/products
/admin/products/create
```

Super clean.

---

# 🧲 9. Custom Route Constraints (Advanced Skill)

You can enforce custom rules on a route.

Example: Must start with letter 'P'

### Step 1: Create the constraint

```csharp
public class StartsWithPConstraint : IRouteConstraint
{
    public bool Match(HttpContextBase httpContext,
                      Route route,
                      string parameterName,
                      RouteValueDictionary values,
                      RouteDirection routeDirection)
    {
        var value = values[parameterName]?.ToString();
        return !string.IsNullOrEmpty(value) && value.StartsWith("P");
    }
}
```

### Step 2: Register constraint

```csharp
var constraints = routes.ConstraintMap;
constraints.Add("StartsWithP", typeof(StartsWithPConstraint));
```

### Step 3: Use it

```csharp
[Route("codes/{name:StartsWithP}")]
```

---

# 📌 10. Custom Routes in RouteConfig

Sometimes you want classic routing for special cases.

Example: Friendly blog URLs

```csharp
routes.MapRoute(
    "BlogRoute",
    "blog/{year}/{month}/{title}",
    new { controller = "Blog", action = "Post" },
    new { year = @"\d{4}", month = @"\d{2}" }
);
```

---

# 🏆 11. Best Practices (Industry-Level)

🔥 Prefer **attribute routing** — it’s cleaner.
🔥 Always use route **constraints** to prevent bad routing.
🔥 Avoid long URLs — keep them semantic.
🔥 Use **RoutePrefix** for admin panels.
🔥 Use **hyphens** in URLs (SEO friendly):

```
/product-details/new-phone
```

🔥 Avoid exposing database IDs where security matters.
🔥 Never overload routes that look too similar.
🔥 Validate route parameters even with constraints — users can hack URLs.

---

# 🧪 Mini Example — Build a Clean Route Structure

Create:

1. `/students` → list
2. `/students/create` → create
3. `/students/5` → details
4. `/students/5/edit` → edit
5. `/students/5/delete` → delete

All done via **attribute routing**.

This is the exact structure used in modern CRUD apps.

---

# 🧩 **Exercise 10 — Build a SEO-Friendly Blog Route System**

Create:

✔ `/blog` → all posts
✔ `/blog/{year}/{month}` → posts by month
✔ `/blog/{slug}` → single blog post

Constraints:

* year = 4 digits
* month = 2 digits
* slug = lowercase letters + hyphens

Use attribute routing for the entire module.

---