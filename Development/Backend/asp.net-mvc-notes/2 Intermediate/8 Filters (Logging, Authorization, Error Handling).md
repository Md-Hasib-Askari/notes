# ✅ **Module 8 — Filters (Logging, Authorization, Error Handling)**

Filters let you inject behavior **before or after** actions run.

This is how large companies keep controllers clean while still enforcing rules.

---

# 🔥 1. What Are Filters?

Filters run **automatically** at specific stages of the MVC pipeline.

Types of filters:

| Filter Type              | Purpose                        |
| ------------------------ | ------------------------------ |
| **Authorization Filter** | Security checks, role checks   |
| **Action Filter**        | Pre/post-processing logic      |
| **Result Filter**        | Modify view result or response |
| **Exception Filter**     | Global error handling          |
| **Resource Filter**      | Low-level resource management  |

You can apply them:

* Globally
* On controller
* On action

---

# 🧠 2. Why Filters Matter (Industry Use)

You use filters for:

✔ Logging requests
✔ Performance measuring
✔ Authentication/authorization
✔ Audit logs
✔ Input sanitization
✔ Exception handling
✔ Caching

They make your controller code **simple and clean**.

---

# 🧱 3. Action Filters (Most Common)

These run **before and after action methods**.

### Example: Custom Logging Filter

```csharp
public class LogActionFilter : ActionFilterAttribute
{
    public override void OnActionExecuting(ActionExecutingContext filterContext)
    {
        // before action
        Debug.WriteLine("Action Started: " + filterContext.ActionDescriptor.ActionName);
    }

    public override void OnActionExecuted(ActionExecutedContext filterContext)
    {
        // after action
        Debug.WriteLine("Action Finished");
    }
}
```

Apply to controller:

```csharp
[LogActionFilter]
public class StudentController : Controller
{
    public ActionResult Index()
    {
        return View();
    }
}
```

---

# 🔐 4. Authorization Filters (Role & Permission Checks)

### Built-in:

```csharp
[Authorize]
public ActionResult Dashboard()
{
    return View();
}
```

### With Roles:

```csharp
[Authorize(Roles = "Admin")]
public ActionResult Manage()
{
    return View();
}
```

---

### Custom Authorization Example

```csharp
public class MustBeCSEAttribute : AuthorizeAttribute
{
    protected override bool AuthorizeCore(HttpContextBase context)
    {
        return context.User.IsInRole("CSE");
    }
}
```

Use:

```csharp
[MustBeCSE]
public ActionResult Index()
{
    return View();
}
```

---

# ⚠️ 5. Exception Filters (Global Error Handling)

This is how you prevent app crashes.

### Custom Error Filter:

```csharp
public class ErrorLoggerFilter : HandleErrorAttribute
{
    public override void OnException(ExceptionContext filterContext)
    {
        // log error
        Debug.WriteLine("ERROR: " + filterContext.Exception.Message);
        
        filterContext.ExceptionHandled = true;
        filterContext.Result = new RedirectResult("/Error/500");
    }
}
```

Register globally:

```csharp
GlobalFilters.Filters.Add(new ErrorLoggerFilter());
```

Now your app never throws unhandled exceptions to the user.

---

# 🧲 6. Result Filters

Modify or inspect the result returned by controllers.

Example: Measure render time.

```csharp
public class TimerFilter : ActionFilterAttribute
{
    private Stopwatch watch;

    public override void OnResultExecuting(ResultExecutingContext filterContext)
    {
        watch = Stopwatch.StartNew();
    }

    public override void OnResultExecuted(ResultExecutedContext filterContext)
    {
        watch.Stop();
        Debug.WriteLine("View Render Time: " + watch.ElapsedMilliseconds + " ms");
    }
}
```

---

# 🧵 7. Global Filters (Apply Everywhere)

In `FilterConfig.cs`:

```csharp
public static void RegisterGlobalFilters(GlobalFilterCollection filters)
{
    filters.Add(new HandleErrorAttribute());  // built-in
    filters.Add(new LogActionFilter());       // custom
}
```

This applies logging or error handling to **all controllers**.

Companies use this for:

* Global logging
* Global authorization
* Global auditing

---

# 🏆 8. Best Practices (Industry-Level)

🔥 Always use filters for:

* Logging
* Security
* Exception handling
* Cross-cutting concerns

🔥 Never put logging or auth logic in controllers
Filters or middleware only.

🔥 Add filters at appropriate levels:

* **Global** → Logging, errors
* **Controller** → Access control
* **Action** → Unique cases

🔥 Keep filters small and focused
Single Responsibility Principle.

🔥 Use DI inside filters if needed
Register them with your container.

---

# 🧪 Mini Example — Apply Global Error Filter

1. Create `ErrorLoggerFilter`
2. Add it in `FilterConfig.cs`
3. Add `~/Error/500` view
4. Throw an exception in any controller
5. Check if user is redirected to custom error page

This replicates real enterprise-level error handling.

---

# 🧩 **Exercise 8 — Build a Logging + Authorization Filter**

### Requirements:

1. Create `ActivityLoggerFilter`:

   * Log user, timestamp, controller, action

2. Apply globally

3. Create `AdminOnlyFilter`:

   * Reject if user isn’t in role “Admin”
   * Return 403 page

4. Apply `AdminOnlyFilter` to an admin controller

This is exactly what companies expect in interviews/projects.

---