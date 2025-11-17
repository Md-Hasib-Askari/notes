# ✅ **Module 3 — Models & ViewModels (Beginner → Intermediate Level)**

Models are the backbone of every MVC app. If you structure them right now, you’ll avoid messy architecture later.

---

# 🔥 1. What Is a Model?

A **Model** represents data in your application.

Examples:

* Student
* Product
* Order
* Employee

It’s a simple C# class (POCO).

Example:

```csharp
public class Student
{
    public int Id { get; set; }
    public string Name { get; set; }
    public string Department { get; set; }
}
```

Models should contain:

* Data
* Validation rules
* Sometimes small helper logic (not heavy business logic)

---

# 🧠 2. Domain Models vs ViewModels

This is **crucial** in professional apps.

## **Domain Model**

Represents your database entity.

Example (Product in the DB):

```csharp
public class Product
{
    public int Id { get; set; }
    public string Name { get; set; }
    public double Price { get; set; }
}
```

## **ViewModel**

Represents data **only needed by the View**.
This protects your database structure and avoids over-posting attacks.

Example:

```csharp
public class ProductViewModel
{
    public string Name { get; set; }
    public double Price { get; set; }
    public string CategoryName { get; set; }
}
```

**Industry best practice:**
👉 Always use **ViewModels** instead of sending DB models directly to Views.

---

# 🎯 3. Passing Models to Views

In controller:

```csharp
public ActionResult Details()
{
    var student = new Student
    {
        Id = 1,
        Name = "Hasib",
        Department = "CSE"
    };

    return View(student);
}
```

In view:

```html
@model Student

<h3>@Model.Name</h3>
<p>Department: @Model.Department</p>
```

Even though you requested "HTML only," ASP.NET MVC **requires Razor** to display model data.

---

# 🔄 4. Binding Models From Forms (Super Important)

If your form has:

```html
<input type="text" name="Name" />
<input type="text" name="Department" />
```

Controller automatically binds inputs:

```csharp
[HttpPost]
public ActionResult Create(Student model)
{
    // model.Name and model.Department are filled automatically
    return Content("Student Added: " + model.Name);
}
```

**This is why naming your form fields correctly is critical.**

---

# 🔒 5. Model Validation (Data Annotations)

Built-in validation you’ll use everywhere:

```csharp
public class RegisterViewModel
{
    [Required]
    [StringLength(50)]
    public string Name { get; set; }

    [EmailAddress]
    public string Email { get; set; }

    [Range(1, 120)]
    public int Age { get; set; }
}
```

MVC automatically:

* Blocks invalid submissions
* Highlights errors
* Shows validation messages

---

# 🏆 6. Best Practices (Industry-Level)

🔥 **Never use domain models in Views**
Always wrap them in a ViewModel.

🔥 **Keep models clean**
Only data + validation — no business logic.

🔥 **Use ViewModels for all forms**
It avoids security issues.

🔥 **Avoid fat models**
Move logic to services.

🔥 **Be explicit**
Don’t use dynamic types for View Models unless necessary.

---

# 🧪 Mini Example — Clean Model + ViewModel Setup

### Domain Model:

```csharp
public class Employee
{
    public int Id { get; set; }
    public string FullName { get; set; }
    public double Salary { get; set; }
}
```

### ViewModel:

```csharp
public class EmployeeDetailsVM
{
    public string FullName { get; set; }
    public double Salary { get; set; }
    public double AnnualSalary => Salary * 12;
}
```

---

# 🧩 **Exercise 3 — Build a Real Model + ViewModel Flow**

Create:

* **Domain Model:** `Book` (Id, Title, Author, Price)
* **ViewModel:** `BookDetailsVM` (Title, Author, Price, PriceWithVat)

VAT = 15%

### Tasks:

1. Create a controller action: `/books/{id}`
2. Create a dummy Book object in controller
3. Map it to BookDetailsVM
4. Pass ViewModel to View
5. Display data in the View

---