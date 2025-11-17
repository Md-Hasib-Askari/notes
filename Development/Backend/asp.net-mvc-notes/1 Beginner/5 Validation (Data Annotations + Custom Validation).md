# ✅ **Module 5 — Validation (Data Annotations + Custom Validation)**

This is one of the most important modules. Real-world apps fail without strong validation.

You’ll learn:
✔ Built-in validation
✔ Custom validation (industry real use)
✔ Server-side + client-side behavior
✔ Best practices

---

# 🔥 1. Why Validation Matters

To prevent:

* Wrong data
* Broken UI
* Security issues
* Invalid database entries

You must validate **before** saving anything.

MVC supports:

* **Server-side validation** (mandatory)
* **Client-side validation** (optional but helpful)

---

# 🏷 2. Built-In Validation (Data Annotations)

Add these on ViewModels — not domain models.

Example:

```csharp
public class RegisterVM
{
    [Required(ErrorMessage = "Name is required")]
    [StringLength(50)]
    public string FullName { get; set; }

    [EmailAddress]
    public string Email { get; set; }

    [Range(18, 60)]
    public int Age { get; set; }
}
```

### Common Attributes:

| Attribute           | Purpose                 |
| ------------------- | ----------------------- |
| `Required`          | Makes field mandatory   |
| `StringLength`      | Limits text length      |
| `EmailAddress`      | Valid email             |
| `Range(min,max)`    | Number range            |
| `RegularExpression` | Custom regex validation |
| `Compare`           | Confirm password fields |

---

# 📌 3. Controller Handling Validation

MVC automatically fills `ModelState`.

Use this pattern:

```csharp
[HttpPost]
public ActionResult Register(RegisterVM model)
{
    if (!ModelState.IsValid)
    {
        return View(model);  // Show errors
    }

    // Process data
    return RedirectToAction("Success");
}
```

This is the **industry standard flow**.

---

# 🖥 4. Showing Validation Errors in View

At the top of the form:

```html
@Html.ValidationSummary(true)
```

Next to each field:

```html
@Html.ValidationMessageFor(m => m.FullName)
```

**Client-side JS validation** works automatically with:

* jQuery
* jQuery Validate
* Unobtrusive validation

These are included by default in MVC templates.

---

# 🎯 5. Custom Validation (Professional Skill)

When built-in rules aren’t enough, create custom logic.

---

## A. Custom Attribute (Simple Example)

```csharp
public class NoBangladeshBannedWordsAttribute : ValidationAttribute
{
    public override bool IsValid(object value)
    {
        var input = value?.ToString() ?? "";
        return !input.Contains("banned");
    }
}
```

Use:

```csharp
[NoBangladeshBannedWords(ErrorMessage = "Invalid word used")]
public string Comment { get; set; }
```

---

## B. Custom Validation Using `IValidatableObject`

Useful when rules depend on multiple fields.

```csharp
public class RegisterVM : IValidatableObject
{
    public string Password { get; set; }
    public string ConfirmPassword { get; set; }

    public IEnumerable<ValidationResult> Validate(ValidationContext context)
    {
        if (Password != ConfirmPassword)
            yield return new ValidationResult("Passwords must match");
    }
}
```

---

# 🔥 6. Common Real-World Validation Scenarios

### Validate Phone Number

```csharp
[RegularExpression(@"^\d{11}$", ErrorMessage = "Invalid phone number")]
public string Phone { get; set; }
```

### Unique Email (Manual Check)

```csharp
if (_userService.EmailExists(model.Email))
{
    ModelState.AddModelError("Email", "Email already exists");
}
```

### Age Must Be 18+

```csharp
[Range(18, 120)]
public int Age { get; set; }
```

---

# 🏆 7. Best Practices (Industry-Level)

🔥 **Always validate on the server**
Client-side JS can be bypassed.

🔥 **Validation only in ViewModels**
Keep domain models clean.

🔥 **Never trust user input**
Even dropdowns must be validated.

🔥 **Keep custom validators small**
Move heavy logic to services.

🔥 **Use regex only when necessary**
Overuse makes code unreadable.

---

# 🧪 Mini Example — Full Validation Flow

### ViewModel:

```csharp
public class LoginVM
{
    [Required]
    [EmailAddress]
    public string Email { get; set; }

    [Required]
    public string Password { get; set; }
}
```

### Controller:

```csharp
[HttpPost]
public ActionResult Login(LoginVM model)
{
    if (!ModelState.IsValid) return View(model);

    return Content("Logged in!");
}
```

### View:

```html
@Html.ValidationSummary(true)

<input name="Email" />
@Html.ValidationMessageFor(m => m.Email)

<input name="Password" />
@Html.ValidationMessageFor(m => m.Password)
```

---

# 🧩 **Exercise 5 — Build a Validated Form**

Create a ViewModel called `ContactVM` with:

* Name (required, max 30 chars)
* Email (email format)
* Message (required, min 10 chars)
* Phone (optional but must be 11 digits if present)

### Tasks:

1. Create a form
2. Add validation messages
3. On POST, show errors on same page
4. When valid, show a “Message Sent!” page

---