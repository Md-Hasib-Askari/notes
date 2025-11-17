# ✅ **Module 18 — Areas & Modular Architecture (Admin Panel, API Modules, Multi-Tenant Apps)**

You’ll learn:

✔ How to use Areas
✔ How to create an Admin Panel
✔ How to create a modular architecture inside MVC
✔ How to organize big apps
✔ How to implement multi-tenant or multi-module systems
✔ How to manage separate routes, layouts, and controllers per module

This is a professional-level skill.

---

# 🔥 1. What Are Areas?

**Areas** allow you to divide your MVC project into **modules**.

Examples of modules:

* Admin
* User
* API
* Dashboard
* Accounts
* HR
* Finance
* Reports
* Inventory

Each module behaves like its own mini-MVC app.

---

# 🧱 2. Creating an Area

In Visual Studio:

**Right-click → Add → Area**

Name it:

```
Admin
```

It generates:

```
/Areas/Admin
    /Controllers
    /Views
    /Models (optional)
    AdminAreaRegistration.cs
```

---

# 🔗 3. Routing Inside Areas

Auto-generated:

```csharp
context.MapRoute(
    "Admin_default",
    "Admin/{controller}/{action}/{id}",
    new { action = "Index", id = UrlParameter.Optional }
);
```

Now URLs look like:

```
/Admin/Dashboard/Index
/Admin/Products/List
/Admin/Users/Edit/5
```

Areas create **clean URL separation**.

---

# 🧭 4. Different Layouts Per Area

Inside:

```
/Areas/Admin/Views/Shared/_Layout.cshtml
```

Admin area can have its own theme.

Set in Views:

```html
@{
    Layout = "~/Areas/Admin/Views/Shared/_Layout.cshtml";
}
```

This allows:
✔ Different menus
✔ Different design
✔ Different scripts
✔ Different security

---

# 🧩 5. Admin Controllers Example

```
/Areas/Admin/Controllers/ProductController.cs
```

```csharp
namespace YourApp.Areas.Admin.Controllers
{
    public class ProductController : Controller
    {
        public ActionResult Index()
        {
            return View();
        }
    }
}
```

---

# 🔐 6. Protecting Entire Admin Area (Role-Based)

Apply Authorization at the Area level:

### Option A — Base Controller

```csharp
[Authorize(Roles = "Admin")]
public class AdminBaseController : Controller {}
```

Then inherit:

```csharp
public class ProductController : AdminBaseController
{
    public ActionResult Index() => View();
}
```

### Option B — Global Filter for Area

```csharp
public class AdminAreaRegistration : AreaRegistration
{
    public override void RegisterArea(AreaRegistrationContext context)
    {
        context.MapRoute(...);

        GlobalFilters.Filters.Add(new AuthorizeAttribute
        {
            Roles = "Admin"
        });
    }
}
```

---

# 🏗 7. Modular Architecture (Enterprise-Level)

Large apps separate code into modules:

```
/Areas/Admin          → Admin Panel
/Areas/Accounts       → Login/Registration
/Areas/API            → REST API
/Areas/Reports        → Charts + Reports
/Areas/Inventory      → Warehouse module
```

Each Area has:

* Controllers
* Views
* Models
* Services
* Routes
* Layout

This is how enterprise ERPs, CRMs, School Management Systems, etc., are built.

---

# 🚀 8. API Area (Optional but Common)

Create:

```
Areas/API
```

Inside:

```
/Areas/API/Controllers/UserController.cs
```

Apply:

```csharp
[RoutePrefix("api/users")]
public class UserController : Controller
{
    [HttpGet]
    [Route("")]
    public JsonResult AllUsers()
    {
        return Json(_service.GetAll(), JsonRequestBehavior.AllowGet);
    }
}
```

Now API routes are clean:

```
/api/users
/api/users/5
/api/users/create
```

---

# 🧲 9. Multi-Tenant Architecture (Advanced)

Areas can separate tenants:

```
/Areas/TenantA
/Areas/TenantB
/Areas/TenantC
```

Each tenant can have:

* Separate views
* Separate themes
* Separate route logic

If needed, you add middleware to detect tenant via subdomain:

```
xyz.company.com → Tenant A  
abc.company.com → Tenant B  
```

---

# 🧰 10. Use Cases for Areas (Real Industry Examples)

✔ Admin dashboards
✔ User dashboards
✔ B2B modules
✔ Payment portal
✔ Vendor portal
✔ Customer portal
✔ API versioning (v1, v2, v3)
✔ Reporting module
✔ Multi-tenant systems
✔ E-commerce backend

---

# 🏆 11. Best Practices

🔥 Use Areas for any module with more than 5-10 controllers
🔥 Give Areas separate layouts
🔥 Split code logically (don’t make Admin messy)
🔥 Protect Areas with role-based auth
🔥 Use Area-specific services if needed
🔥 Prefix routes clearly
🔥 Use Areas for large teams (each team maintains one Area)
🔥 Keep admin UI separate from user UI

---

# 🧪 Mini Example — Admin Module Setup

Create:

✔ Area: `Admin`
✔ Controller: `DashboardController`
✔ View: `Index.cshtml`
✔ Layout: `/Areas/Admin/Views/Shared/_Layout.cshtml`
✔ Role-based protection
✔ Admin menu with links
✔ Added statistics: total users, sales, products

This will give you a fully functional admin panel.

---

# 🧩 **Exercise 18 — Build a Modular Structure**

Build a **Student Management System** with:

### Areas:

* `Admin`
* `Students`
* `Teachers`
* `Accounts`

### Requirements:

* Each area has its own controllers
* Each area uses its own layout
* Admin area uses `[Authorize(Roles="Admin")]`
* Student area uses `[Authorize(Roles="Student")]`
* Accounts area contains login/register

You’ll experience real enterprise structure.

---