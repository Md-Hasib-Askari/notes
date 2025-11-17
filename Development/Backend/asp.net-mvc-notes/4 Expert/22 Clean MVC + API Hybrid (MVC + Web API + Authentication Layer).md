# ✅ **Module 22 — Clean MVC + API Hybrid (MVC + Web API + Authentication Layer)**

You’ll learn:

✔ How to build MVC views + Web API in one project
✔ REST API architecture
✔ API controllers vs MVC controllers
✔ Token-based authentication (JWT)
✔ Protecting APIs
✔ Rate limiting
✔ Versioning your API
✔ Clean separation between UI and API layers
✔ How companies build hybrid web systems

---

# 🔥 1. MVC vs Web API (Know the Difference)

### MVC Controller:

* Returns HTML
* Used for UI pages
* Uses Views

### API Controller:

* Returns JSON
* Used for mobile apps, dashboards, external clients
* No Views

In MVC 5:

```csharp
public class StudentController : Controller
```

API Controller:

```csharp
public class StudentApiController : ApiController
```

---

# 🧱 2. Creating an API Layer (Proper Folder Structure)

```
/Controllers
   HomeController
   AccountController

/Api
   /V1
      StudentApiController
      ProductApiController
```

OR inside Areas:

```
/Areas/API
   /Controllers
```

Best for large apps.

---

# 🔗 3. API Routing (Attribute Routing Recommended)

Enable in WebApiConfig:

```csharp
config.MapHttpAttributeRoutes();
```

API example:

```csharp
[RoutePrefix("api/v1/students")]
public class StudentApiController : ApiController
{
    [HttpGet]
    [Route("")]
    public IHttpActionResult GetAll()
    {
        var students = _service.GetAll();
        return Ok(students);
    }
}
```

Result:

```
GET /api/v1/students
```

---

# 🧠 4. Services Shared Between MVC + API

Both MVC and API should call *the same service layer*.

```
MVC Controller --> IStudentService  
API Controller --> IStudentService  
```

This ensures consistency.

---

# 🧪 5. Returning JSON

Common patterns:

```csharp
return Ok(data);
```

or

```csharp
return Json(data);
```

or

```csharp
return Content(JsonConvert.SerializeObject(data), "application/json");
```

---

# 🔐 6. Token-Based Authentication (JWT)

JWT is the industry standard for:

* Mobile apps
* Frontend apps (React/Vue/Angular)
* External clients
* Modern API systems

---

## 🔥 Step 1 — Install JWT Packages

```
Install-Package Microsoft.Owin.Security.Jwt
Install-Package System.IdentityModel.Tokens.Jwt
```

---

## 🔥 Step 2 — Configure JWT in Startup

```csharp
var issuer = "your-company";
var audience = "your-users";
var secret = TextEncodings.Base64Url.Decode("your-secret-key");

app.UseJwtBearerAuthentication(new JwtBearerAuthenticationOptions
{
    AuthenticationMode = AuthenticationMode.Active,
    AllowedAudiences = new[] { audience },
    IssuerSecurityTokenProviders = new[]
    {
        new SymmetricKeyIssuerSecurityTokenProvider(issuer, secret)
    }
});
```

---

## 🔥 Step 3 — Issue Tokens

```csharp
public string GenerateJwtToken(ApplicationUser user)
{
    var tokenHandler = new JwtSecurityTokenHandler();
    var key = Encoding.ASCII.GetBytes("your-secret-key");

    var tokenDescriptor = new SecurityTokenDescriptor
    {
        Subject = new ClaimsIdentity(new Claim[]
        {
            new Claim(ClaimTypes.Name, user.UserName),
            new Claim(ClaimTypes.Email, user.Email),
            new Claim("Role", "Admin")
        }),
        Expires = DateTime.UtcNow.AddHours(3),
        SigningCredentials = new SigningCredentials(
            new SymmetricSecurityKey(key),
            SecurityAlgorithms.HmacSha256Signature)
    };

    var token = tokenHandler.CreateToken(tokenDescriptor);
    return tokenHandler.WriteToken(token);
}
```

---

## 🔥 Step 4 — Protect API Routes

```csharp
[Authorize]
[Route("api/v1/securedata")]
public IHttpActionResult GetSecureData()
{
    return Ok("Secret data only for logged-in users");
}
```

---

# 🧲 7. API Versioning (Must-Have For Stability)

Use route prefix:

```
/api/v1/products
/api/v2/products
```

Example:

```csharp
[RoutePrefix("api/v2/products")]
```

Companies version APIs to avoid breaking older clients.

---

# ⚡ 8. Rate Limiting (Security & Protection)

Protect your API from abuse.

### Basic approach:

Store user request count in cache or Redis.

If `%Requests in last X seconds > limit`, return:

```csharp
return Content(HttpStatusCode.TooManyRequests, "Slow down!");
```

---

# 🌐 9. CORS Support (Allow Frontend Apps to Access API)

If building React/Vue/Angular frontend:

Add to WebApiConfig:

```csharp
var cors = new EnableCorsAttribute("*", "*", "*");
config.EnableCors(cors);
```

---

# 📦 10. MVC + API Hybrid Architecture (Enterprise-Level)

A professional project looks like:

```
/YourApp.Web
  /Controllers         → HTML pages
  /Api                 → JSON endpoints
  /ViewModels
  /Filters
  /Middlewares

/YourApp.Services      → Business logic
/YourApp.Data          → EF + DB
/YourApp.Core          → Entities + Contracts
```

Both MVC and API use the **same** logic.

This architecture scales smoothly to:

* Mobile apps
* Desktop apps
* Public APIs
* Admin dashboards
* SaaS applications

---

# 🏆 11. Best Practices (Expert-Level)

🔥 Keep API controllers in separate folder or Area
🔥 Use attribute routing for APIs
🔥 Use ViewModels for API requests
🔥 Never expose entity models via API
🔥 Always validate API requests
🔥 Protect APIs using JWT
🔥 Apply rate limiting
🔥 Version your API
🔥 Use HTTPS only
🔥 Log every API call
🔥 Don’t mix MVC views and API logic in same controller

---

# 🧪 Mini Example — Create a Secure API Module

Create:

1. `Areas/API/V1/StudentApiController.cs`
2. Endpoints:

   * GET /api/v1/students
   * GET /api/v1/students/{id}
   * POST /api/v1/students
3. Protect POST with `[Authorize]`
4. Issue JWT tokens at `/api/v1/auth/login`
5. Test authorization:

   ```
   Authorization: Bearer <token>
   ```

This mirrors real-world API development.

---

# 🧩 **Exercise 22 — Build a Full MVC + API Hybrid System**

Create:

### MVC Module:

* Product listing
* Product details
* Admin dashboard (Area)

### API Module:

* `api/v1/products` (get all)
* `api/v1/products/{id}` (details)
* `api/v1/products` (POST – token required)

### Security:

* JWT-based auth for API
* Cookie auth for MVC
* Role-based admin access

### Advanced Features:

* Rate limiting
* API versioning (v1 + v2)
* Logging & monitoring for API calls

Once you complete this, you are at **professional, production-ready MVC level**.

---