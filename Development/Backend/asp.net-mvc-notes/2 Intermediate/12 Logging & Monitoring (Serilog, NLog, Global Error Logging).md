# ✅ **Module 12 — Logging & Monitoring (Serilog, NLog, Global Error Logging)**

Goals:
✔ Add real logging
✔ Centralize error tracking
✔ Create audit logs
✔ Log user activities
✔ Use Serilog/NLog like industry projects
✔ Build a global exception handling pipeline

---

# 🔥 1. Why Logging Matters

You must log:

* User actions
* Errors and exceptions
* System failures
* Login attempts
* Data changes
* API calls
* Performance timings

Logging is essential for:

* Debugging
* Security monitoring
* Analytics
* Compliance
* Auditing

---

# 🧱 2. Built-in Logging (OK but Limited)

You can use:

```csharp
System.Diagnostics.Debug.WriteLine("Something happened");
```

Or:

```csharp
Trace.WriteLine("Trace log");
```

**But this is not scalable or production-grade.**

Professional apps use **Serilog** or **NLog**.

---

# 🟦 3. Using Serilog (Most Popular Today)

Install NuGet packages:

```
Install-Package Serilog
Install-Package Serilog.Sinks.File
Install-Package Serilog.Sinks.Console
Install-Package Serilog.Sinks.MongoDB
Install-Package Serilog.Sinks.Seq
```

---

## ⭐ Basic Configuration (Global.asax)

```csharp
Log.Logger = new LoggerConfiguration()
    .WriteTo.File("logs/log.txt", rollingInterval: RollingInterval.Day)
    .CreateLogger();
```

Now you can log:

```csharp
Log.Information("User {Name} visited homepage", User.Identity.Name);
Log.Error(ex, "Failed to load dashboard");
```

---

## ⭐ Output Example:

```
2025-11-13 14:33:01 INF User Hasib visited homepage
2025-11-13 14:34:10 ERR Failed to load dashboard
System.NullReferenceException...
```

---

# 🟨 4. Using NLog (Enterprises Love It)

Install:

```
Install-Package NLog
Install-Package NLog.Config
Install-Package NLog.Web
```

Add `nlog.config`:

```xml
<targets>
   <target xsi:type="File" name="logfile" fileName="logs/log.txt" />
</targets>
<rules>
   <logger name="*" minlevel="Info" writeTo="logfile" />
</rules>
```

Log in controller:

```csharp
private static Logger logger = LogManager.GetCurrentClassLogger();

logger.Info("User logged in");
logger.Error(ex, "Error in StudentController");
```

---

# 🧲 5. Global Error Logging (HandleErrorAttribute)

You can capture all unhandled exceptions globally.

### Create Custom Filter

```csharp
public class GlobalExceptionLogger : HandleErrorAttribute
{
    public override void OnException(ExceptionContext filterContext)
    {
        Log.Error(filterContext.Exception, "Unhandled exception occurred");

        filterContext.ExceptionHandled = true;
        filterContext.Result = new RedirectResult("/Error/500");
    }
}
```

### Register Globally

`FilterConfig.cs`:

```csharp
filters.Add(new GlobalExceptionLogger());
```

Now any crash is logged automatically.

---

# 🧾 6. Logging User Activity (Audit Logging)

Example: Log every action call.

```csharp
public class ActivityLogFilter : ActionFilterAttribute
{
    public override void OnActionExecuting(ActionExecutingContext context)
    {
        var user = context.HttpContext.User.Identity.Name ?? "Anonymous";
        var action = context.ActionDescriptor.ActionName;
        var controller = context.ActionDescriptor.ControllerDescriptor.ControllerName;

        Log.Information("{User} visited {Controller}/{Action}", user, controller, action);
    }
}
```

Apply globally:

```csharp
filters.Add(new ActivityLogFilter());
```

Audit logs help you:

* Detect suspicious users
* Track admin actions
* Analyze user behavior

---

# 📈 7. Performance Monitoring (Stopwatch Logs)

Example:

```csharp
public class PerformanceFilter : ActionFilterAttribute
{
    private Stopwatch watch;

    public override void OnActionExecuting(ActionExecutingContext context)
    {
        watch = Stopwatch.StartNew();
    }

    public override void OnActionExecuted(ActionExecutedContext context)
    {
        watch.Stop();
        Log.Information("Action {Name} took {Time} ms",
            context.ActionDescriptor.ActionName,
            watch.ElapsedMilliseconds);
    }
}
```

This identifies slow actions immediately.

---

# 🌐 8. External Monitoring Tools (Industry Tools)

Large teams use tools like:

* **Seq** (Serilog dashboard)
* **Elastic Stack (ELK)**
* **Azure Application Insights**
* **Datadog**
* **New Relic**
* **Sentry**
* **Raygun**

These tools:

* Collect logs
* Show trends
* Notify about issues
* Provide dashboards

---

# 🏆 9. Best Practices (Enterprise Level)

🔥 Log only what’s necessary
Too much logging = performance drop.

🔥 Never log sensitive data
(passwords, tokens, personal info)

🔥 Use structured logging
Example:

```csharp
Log.Information("Order {OrderId} created by {User}", orderId, username);
```

🔥 Log at correct levels:

* `Verbose`
* `Debug`
* `Information`
* `Warning`
* `Error`
* `Fatal`

🔥 Always use global error handling
No unhandled exceptions allowed.

🔥 Store logs outside the app folder
security purposes.

---

# 🧩 **Exercise 12 — Build a Logging System**

Implement:

1. Serilog with rolling logs
2. Global exception handler using your custom filter
3. Activity log filter
4. Performance monitoring filter
5. Log:

   * Logins
   * CRUD operations
   * Exceptions
   * Admin page visits

This will give you a complete enterprise-grade logging setup.

---