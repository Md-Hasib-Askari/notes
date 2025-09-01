# Advanced ASP.NET Core (Advanced Notes)

## 1. Middleware

* **Middleware**: Components that handle requests/responses in the pipeline.
* Common examples: Routing, Authentication, Error Handling, Static Files.
* Custom middleware example:

  ```csharp
  public class LoggingMiddleware
  {
      private readonly RequestDelegate _next;
      public LoggingMiddleware(RequestDelegate next) => _next = next;

      public async Task Invoke(HttpContext context)
      {
          Console.WriteLine($"Request: {context.Request.Path}");
          await _next(context);
      }
  }
  ```
* Registered in `Program.cs` or `Startup.cs` with `app.UseMiddleware<LoggingMiddleware>()`.

---

## 2. Authentication & Authorization

* **Authentication**: Verifying user identity.

  * Options: ASP.NET Core Identity, JWT (JSON Web Token), OAuth2, OpenID Connect.
* **Authorization**: Defining access rights.

  * Role-based (`[Authorize(Roles="Admin")]`).
  * Policy-based (custom rules).
* Example with JWT:

  ```csharp
  services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
      .AddJwtBearer(options => { options.TokenValidationParameters = ... });
  ```

---

## 3. Caching & Logging

* **Caching**:

  * **In-Memory Cache**: `IMemoryCache` for temporary data.
  * **Distributed Cache**: Redis/SQL for multiple servers.

  ```csharp
  services.AddMemoryCache();
  ```
* **Logging**:

  * Built-in logging providers: Console, Debug, EventLog.
  * Third-party: Serilog, NLog, Seq.
  * Usage:

  ```csharp
  private readonly ILogger<HomeController> _logger;
  _logger.LogInformation("Processing request");
  ```

---

## 4. SignalR (Real-time Communication)

* **SignalR**: Enables real-time communication between server and clients.
* Supports WebSockets, Server-Sent Events, Long Polling.
* Example Hub:

  ```csharp
  public class ChatHub : Hub
  {
      public async Task SendMessage(string user, string message)
      {
          await Clients.All.SendAsync("ReceiveMessage", user, message);
      }
  }
  ```
* Clients (JS, .NET, Java) can connect and receive push updates instantly.
