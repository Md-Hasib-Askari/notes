# ✅ **Module 4 — Views, Forms & UI (Beginner → Intermediate Level)**

You'll learn:

* How Views work
* How to build forms
* How to send user input back to the controller
* How to structure clean UI with industry practices

---

# 🔍 1. What Is a View?

A **View** is simply the HTML page sent to the browser.

Location is always:

```
/Views/{ControllerName}/{ActionName}.cshtml
```

Example:
Controller: `StudentController`
Action: `Details` → View path:

```
Views/Student/Details.cshtml
```

---

# 🧩 2. Types of Views

### 1️⃣ **Regular Views**

HTML pages shown to users.

### 2️⃣ **Partial Views**

Reusable page sections (like components).

### 3️⃣ **Layout Views**

Master templates (header, nav, footer).

---

# 🔧 3. Layouts (Master Pages)

Default layout:
`Views/Shared/_Layout.cshtml`

This wraps every page.

Example layout structure:

```html
<!DOCTYPE html>
<html>
<head>
    <title>@ViewBag.Title</title>
</head>
<body>
    <header>
        <h2>My MVC App</h2>
    </header>

    <div>
        @RenderBody()
    </div>
</body>
</html>
```

Inside a view:

```html
@{
    Layout = "~/Views/Shared/_Layout.cshtml";
}
```

**Industry best practice:**
👉 Keep layout clean, include reusable components, and avoid logic.

---

# 📝 4. Rendering Data in Views

Even if you prefer pure HTML, you still need minimal Razor to output dynamic data.

Example:

```html
<h1>@Model.Name</h1>
```

Another example:

```html
<ul>
    @foreach (var s in ViewBag.Students)
    {
        <li>@s</li>
    }
</ul>
```

---

# ✏️ 5. Creating Forms (Real-World Important)

Forms send user input to your controller.

---

## **A. Basic HTML Form (Simple & Clean)**

```html
<form action="/student/create" method="post">
    <input type="text" name="Name" placeholder="Student Name" />
    <input type="text" name="Department" placeholder="Department" />
    <button type="submit">Save</button>
</form>
```

Controller:

```csharp
[HttpPost]
public ActionResult Create(Student model)
{
    return Content(model.Name + " from " + model.Department);
}
```

---

## **B. Best Practice Form — With Anti-Forgery Protection**

Inside form:

```html
@Html.AntiForgeryToken()
```

Controller:

```csharp
[ValidateAntiForgeryToken]
[HttpPost]
public ActionResult Create(Student model)
{
    // Save
}
```

**Why?**
It protects against CSRF attacks. Industry standard.

---

# 🔄 6. Strongly Typed Views (Professional Approach)

Top of your view:

```html
@model Student
```

Then:

```html
<p>Name: @Model.Name</p>
<p>Department: @Model.Department</p>
```

This is the cleanest and safest way to work with data.

---

# 🧱 7. Using Partial Views (Reusability)

If you have something like a card or table used everywhere:

```
Views/Shared/_StudentCard.cshtml
```

Render it:

```html
@Html.Partial("_StudentCard", Model)
```

Industry practice:
👉 Use partials to avoid copy-paste UI.

---

# 📌 8. Best Practices (Industry-Level)

🔥 Keep HTML simple and semantic
🔥 Use ViewModels for all forms
🔥 Use Layouts for consistent design
🔥 Never put business logic in views
🔥 Use partials for reusable components
🔥 Validate user input on both server & client

---

# 🧪 Mini Example — Clean Form + ViewModel Flow

### ViewModel:

```csharp
public class RegisterVM
{
    [Required]
    public string FullName { get; set; }

    [EmailAddress]
    public string Email { get; set; }
}
```

### View:

```html
@model RegisterVM

<form method="post" action="/account/register">
    @Html.AntiForgeryToken()

    <input name="FullName" placeholder="Full Name" />
    <input name="Email" placeholder="Email" />

    <button type="submit">Register</button>
</form>
```

### Controller:

```csharp
[HttpPost]
[ValidateAntiForgeryToken]
public ActionResult Register(RegisterVM model)
{
    return Content("Welcome, " + model.FullName);
}
```

---

# 🧩 **Exercise 4 — Build a Real Form Page**

Create a full **Create Student** page:

## Form Fields:

* Name
* Department
* Age
* Email

## Requirements:

1. Use a ViewModel
2. Use HTML form
3. Add anti-forgery token
4. Post to controller
5. Display submitted data in a result view

---