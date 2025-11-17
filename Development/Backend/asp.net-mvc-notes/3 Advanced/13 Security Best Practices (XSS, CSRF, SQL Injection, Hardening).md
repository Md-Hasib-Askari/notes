# ✅ **Module 13 — Security Best Practices (XSS, CSRF, SQL Injection, Hardening)**

You'll learn how to protect against the most common and dangerous attacks:

✔ XSS (Cross-Site Scripting)
✔ CSRF (Cross-Site Request Forgery)
✔ SQL Injection
✔ Clickjacking
✔ Session & cookie hijacking
✔ Strong password & auth policies
✔ Security headers
✔ HTTPS & HSTS

Let’s go step-by-step.

---

# 🔥 1. Cross-Site Scripting (XSS)

### The danger:

Hackers inject dangerous JS into your pages.

Example:

```
<script>alert('hacked')</script>
```

### MVC has *built-in* protection:

Everything is **HTML-encoded by default**.

Example:

```html
@Model.Name 
```

Even if the user typed `<script>...</script>`, it becomes safe text.

---

### 🚫 Dangerous:

```html
@Html.Raw(Model.Description)   // allows HTML => XSS risk
```

Only use `Html.Raw()` when you're 100% sure content is safe.

---

# 🛡 2. CSRF Protection (Must-Have)

CSRF = forging requests from another site to trick your logged-in users.

MVC protects you by using anti-forgery tokens.

### In your form:

```html
@Html.AntiForgeryToken()
```

### In your controller:

```csharp
[ValidateAntiForgeryToken]
public ActionResult Create(Student model)
```

This protects you 100% against CSRF attacks.

---

# 🔐 3. SQL Injection Protection

SQL injection happens when you use **raw SQL strings**.

Example of dangerous code:

```csharp
var sql = "SELECT * FROM Users WHERE Name = '" + name + "'";
```

This can be hacked by:

```
' OR '' = '
```

Result: Full table dumped.

---

### ✔ Entity Framework automatically prevents SQL injection

EF uses parameters internally:

```csharp
var user = _db.Users.FirstOrDefault(u => u.Email == email);
```

SAFE.

---

### 🚫 Only dangerous if you use raw SQL:

```csharp
_db.Database.SqlQuery<User>("SELECT * FROM Users WHERE Name = '" + name + "'");
```

Use parameters instead:

```csharp
_db.Database.SqlQuery<User>(
    "SELECT * FROM Users WHERE Name = @p0", name
);
```

---

# 🔏 4. Secure Cookies

Use secure cookies so attackers can’t steal session data.

### In `web.config`:

```xml
<httpCookies requireSSL="true" />
<sessionState cookieName="MyAppSession" />
```

### Also:

```xml
<authentication>
  <forms requireSSL="true" />
</authentication>
```

---

# 🔒 5. HTTPS & HSTS (Mandatory)

### Force HTTPS:

```xml
<rewrite>
  <rules>
    <rule name="HTTPS Redirect" enabled="true">
      <match url="(.*)" />
      <conditions>
        <add input="{HTTPS}" pattern="off" />
      </conditions>
      <action type="Redirect" url="https://{HTTP_HOST}/{R:1}" redirectType="Permanent" />
    </rule>
  </rules>
</rewrite>
```

### Add HSTS header:

Prevents browser from ever using HTTP.

```csharp
Response.Headers["Strict-Transport-Security"] = "max-age=31536000";
```

---

# 🧱 6. Clickjacking Prevention

Add header:

```csharp
Response.Headers["X-Frame-Options"] = "DENY";
```

Stops attackers from framing your site in hidden iframes.

---

# 🧲 7. Password Hardening

Identity already uses:

* Hashing
* Salting
* PBKDF2

You just enforce strong rules:

```csharp
manager.PasswordValidator = new PasswordValidator
{
    RequiredLength = 8,
    RequireDigit = true,
    RequireLowercase = true,
    RequireUppercase = true,
    RequireNonLetterOrDigit = false
};
```

---

# 🔐 8. Login Attempt Lockout

Prevent brute-force attacks:

```csharp
manager.UserLockoutEnabledByDefault = true;
manager.DefaultAccountLockoutTimeSpan = TimeSpan.FromMinutes(5);
manager.MaxFailedAccessAttemptsBeforeLockout = 5;
```

---

# 🔍 9. User Input Sanitization

Use regex for dangerous fields:

```csharp
[RegularExpression("^[a-zA-Z0-9 ]+$")]
public string Title { get; set; }
```

Reject dangerous characters early.

---

# 🧵 10. Don’t Expose Sensitive Data in URLs

Bad:

```
/users/delete?id=12
```

Better:

```
/users/12/delete
```

Never show:

* Tokens
* Password resets
* Keys
* Internal IDs for critical data

---

# 🧩 11. Use Security Headers

Add these in global filter or middleware:

```csharp
Response.Headers["X-XSS-Protection"] = "1; mode=block";
Response.Headers["X-Content-Type-Options"] = "nosniff";
Response.Headers["Referrer-Policy"] = "no-referrer";
Response.Headers["Permissions-Policy"] = "camera=(), microphone=()";
Response.Headers["Content-Security-Policy"] = "default-src 'self'";
```

This protects:

* XSS
* Content injection
* Browser attacks
* Privacy
* Data leaks

---

# 🏆 12. Best Practices (Enterprise Level)

🔥 Always use HTTPS
🔥 Use anti-forgery token on all POST forms
🔥 Never trust user input
🔥 Avoid `Html.Raw()`
🔥 Validate all data using ViewModels
🔥 Don’t use raw SQL without parameters
🔥 Use strong passwords & lockouts
🔥 Rotate keys and secrets
🔥 Sanitize file uploads
🔥 Limit session timeout
🔥 Apply global error handler
🔥 Secure cookies (HTTPOnly + Secure flag)

---

# 🧪 **Mini Task — Make Your MVC App Secure**

Secure your app by implementing:

1. Anti-forgery tokens
2. HTTPS redirect
3. XSS prevention
4. Security headers
5. Repository + EF (no raw SQL)
6. Password lockout
7. Role-based access
8. Custom error page + logging filter

This makes your app production-grade.

---