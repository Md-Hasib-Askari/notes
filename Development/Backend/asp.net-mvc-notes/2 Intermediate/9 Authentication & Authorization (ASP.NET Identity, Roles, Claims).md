# ✅ **Module 9 — Authentication & Authorization (ASP.NET Identity, Roles, Claims)**

This is where your app becomes secure and ready for real users.

We cover:
✔ Authentication (login, logout)
✔ Authorization (roles, policies, claims)
✔ Password hashing
✔ User management
✔ Protecting controllers & actions

---

# 🔥 1. What Is ASP.NET Identity?

A full authentication framework that handles:

* Users
* Passwords
* Login
* Logout
* Security tokens
* Roles
* Claims
* Password hashing
* User profiles
* Two-factor auth

You never store passwords manually — Identity handles everything securely.

---

# 🧱 2. Default ASP.NET Identity Tables

When Identity is added, EF automatically creates:

* **AspNetUsers**
* **AspNetRoles**
* **AspNetUserRoles**
* **AspNetUserClaims**
* **AspNetUserLogins**

These give you a complete auth system out of the box.

---

# 🚪 3. Authentication — Login & Logout

Let’s start with the basics.

---

## 🔐 **A. Login Example**

### Login ViewModel:

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

### Login Controller Action:

```csharp
public class AccountController : Controller
{
    private readonly SignInManager<ApplicationUser, string> _signInManager;

    public AccountController(SignInManager<ApplicationUser, string> signInManager)
    {
        _signInManager = signInManager;
    }

    [HttpPost]
    public async Task<ActionResult> Login(LoginVM model)
    {
        if (!ModelState.IsValid) return View(model);

        var result = await _signInManager.PasswordSignInAsync(
            model.Email,
            model.Password,
            isPersistent: false,
            shouldLockout: false
        );

        if (result == SignInStatus.Success)
            return RedirectToAction("Index", "Home");

        ModelState.AddModelError("", "Invalid login attempt");
        return View(model);
    }
}
```

---

## 🔓 **B. Logout Example**

```csharp
public async Task<ActionResult> Logout()
{
    await _signInManager.SignOutAsync();
    return RedirectToAction("Login");
}
```

Clean and secure.

---

# 🪪 4. Register (Create Account)

### Register ViewModel:

```csharp
public class RegisterVM
{
    [Required]
    public string UserName { get; set; }

    [Required]
    [EmailAddress]
    public string Email { get; set; }

    [Required]
    public string Password { get; set; }
}
```

### Register Action:

```csharp
public async Task<ActionResult> Register(RegisterVM model)
{
    if (!ModelState.IsValid) return View(model);

    var user = new ApplicationUser
    {
        UserName = model.UserName,
        Email = model.Email
    };

    var result = await _userManager.CreateAsync(user, model.Password);

    if (result.Succeeded)
        return RedirectToAction("Login");

    foreach (var error in result.Errors)
        ModelState.AddModelError("", error);

    return View(model);
}
```

---

# 👮 5. Authorization (Roles)

Roles are the simplest way to control access.

---

## **Create Role**

```csharp
await _roleManager.CreateAsync(new IdentityRole("Admin"));
```

---

## **Assign User to Role**

```csharp
await _userManager.AddToRoleAsync(user.Id, "Admin");
```

---

## **Protect Controller or Action**

### Entire Controller:

```csharp
[Authorize(Roles = "Admin")]
public class AdminController : Controller
{
}
```

### Single Action:

```csharp
[Authorize(Roles = "Manager")]
public ActionResult Dashboard()
{
    return View();
}
```

Companies expect this.

---

# ⚙️ 6. Claims-Based Authorization (Modern Standard)

Roles = simple
Claims = powerful

Examples of claims:

* Department = CSE
* AccessLevel = 5
* Country = BD
* Subscription = Premium

### Adding claim:

```csharp
await _userManager.AddClaimAsync(user.Id, new Claim("Department", "CSE"));
```

### Checking claim:

```csharp
[Authorize(ClaimType="Department", ClaimValue="CSE")]
```

Or manually:

```csharp
var dept = User.Claims.FirstOrDefault(c => c.Type == "Department");
```

---

# 🔐 7. Password Hashing (Built-in Security)

ASP.NET Identity:

* Hashes every password
* Adds salt
* Uses PBKDF2 by default
* Extremely secure

You NEVER store raw passwords — Identity handles it.

---

# 🛡 8. Protecting Pages

### Require Login:

```csharp
[Authorize]
```

### Allow anonymous users:

```csharp
[AllowAnonymous]
```

Used for login/register pages.

---

# 🍪 9. Cookie Authentication (How Login Works)

When a user logs in:

* ASP.NET creates an encrypted auth cookie
* Browser stores it
* Every request sends it

Identity reads the cookie → user is authenticated.

Secure and standard across industry.

---

# 🧩 10. Industry Best Practices

🔥 Always use **ViewModels** for login/register
Never use entity models.

🔥 Use HTTPS to protect login cookies
Non-negotiable.

🔥 Enforce strong password rules
Identity supports this with `PasswordValidator`.

🔥 Never store plain passwords
Identity handles hashing automatically.

🔥 Use roles for simple apps
Use claims for large apps.

🔥 Lockout after failed login attempts
Prevent brute-force attacks.

🔥 Protect admin areas with role-based filters
Never trust client-side checks.

---

# 🧪 **Mini Example — Admin Dashboard Protection**

1. Create `Admin` role
2. Assign your user to Admin
3. Protect controller with:

   ```csharp
   [Authorize(Roles="Admin")]
   ```
4. Try accessing without login
5. Try accessing as non-admin

This mimics real production behavior.

---

# 🧩 **Exercise 9 — Build a Complete Auth System**

Build:

1. **Register** page
2. **Login** page
3. **Logout** button
4. Apply `[Authorize]` to `/dashboard`
5. Create `Admin` role
6. Protect `/admin/panel` with `[Authorize(Roles = "Admin")]`

Optional bonus:

* Add **lockout after 5 failed login attempts**
* Add **Remember Me**

Once you do this, you understand 70% of real app security.

---