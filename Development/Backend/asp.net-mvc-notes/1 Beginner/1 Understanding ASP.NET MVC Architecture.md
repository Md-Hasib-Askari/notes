# ✅ **Module 1 — Understanding ASP.NET MVC Architecture (Beginner Level)**

This is the foundation. If you get this right, everything else becomes easier.

---

# 🔍 1. What Problem Does MVC Solve?

Web apps get messy fast if you mix everything (HTML, database code, logic) in one place.

**MVC separates responsibilities into three layers:**

* **M — Model** → Data and business rules
* **V — View** → UI (HTML templates)
* **C — Controller** → Request handling and deciding what to show

**Why companies love MVC:**

* Clean structure
* Easy to maintain
* Testable
* Scales well for large apps

---

# 🧠 2. How MVC Works (Step-by-Step Flow)

When a user requests a page:

1. **User hits a URL** → `/products/details/5`
2. **Route matches the URL** to a controller action
3. The **controller** runs logic (fetches product from DB)
4. The controller sends data to the **View**
5. The **View** renders HTML
6. Browser displays the page

This flow never changes — it’s the core of MVC.

---

# 🏗 3. Visual Breakdown

```
┌────────────┐      ┌──────────────┐      ┌──────────────┐
│    User    │ ---> │   Controller │ ---> │     Model    │
└────────────┘      └──────────────┘      └──────────────┘
                         |                      ▲
                         v                      │
                     ┌──────────────┐           │
                     │     View     │ <─────────┘
                     └──────────────┘
```

---

# 📦 4. Folder Structure in ASP.NET MVC (Important!)

Your app will always start like this:

```
/Controllers
/Models
/Views
/App_Start
/Content
/Scripts
/web.config
```

### What each folder does:

* **Controllers** → C# classes that handle requests
* **Models** → Your data structures
* **Views** → HTML pages
* **App_Start** → Routing, filters, bundles
* **Content** → CSS, images
* **Scripts** → JavaScript files

---

# 🧪 Mini Example (Your First MVC Flow)

### Step 1 — Create a Controller

```csharp
public class HomeController : Controller
{
    public ActionResult Index()
    {
        ViewBag.Message = "Welcome to MVC!";
        return View();
    }
}
```

### Step 2 — Create the View (HTML)

Inside:
`Views/Home/Index.cshtml`

But since you want **HTML**, write:

```html
<h1>@ViewBag.Message</h1>
<p>This is your first MVC page.</p>
```

> Razor is still required for dynamic content (`@...`).

---

# 📌 Best Practices (Industry-Level)

* Controllers must stay **thin**
  → only coordinate logic
* Views must only display UI
  → no heavy logic
* Models must hold clean data
  → no database code in controllers

Keep this discipline from day one.

---

# 🧩 Exercise 1 (Do This Before Next Module)

1. Create a new ASP.NET MVC project
2. Add a `HomeController`
3. Add actions:

   * `Index()`
   * `About()`
   * `Contact()`
4. Create HTML views for all 3
5. Display your name and current timestamp from the controller
6. Create a new custom route:
   `/my/info/{id}` → goes to HomeController.Info

Example controller action:

```csharp
public ActionResult Info(int id)
{
    ViewBag.Id = id;
    return View();
}
```

---