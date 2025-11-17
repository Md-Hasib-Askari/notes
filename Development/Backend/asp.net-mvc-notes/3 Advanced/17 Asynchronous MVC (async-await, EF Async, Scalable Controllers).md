# ✅ **Module 17 — Asynchronous MVC (async/await, EF Async, Scalable Controllers)**

In this module you’ll learn:

✔ Why async matters
✔ How MVC handles async actions
✔ Async EF queries
✔ Async file operations
✔ Avoiding thread starvation
✔ Making services async
✔ Best practices for real systems

Let’s go.

---

# 🔥 1. Why Async Is Mandatory in Modern MVC

### Synchronous method:

* Each request uses 1 thread
* Thread waits during DB/API/file I/O
* High traffic overloads server
* App becomes slow or crashes

### Asynchronous method:

* Thread is *released* during DB/API I/O
* Server handles more parallel requests
* Response time improves
* App becomes scalable

Real-world companies build everything async.

---

# 🧱 2. Async Action Methods

Basic async action:

```csharp
public async Task<ActionResult> Index()
{
    await Task.Delay(2000); // Simulate long operation
    return View();
}
```

MVC automatically handles the async pipeline.

---

# 🧠 3. EF Async Methods (Real Use)

Entity Framework provides async versions of all major calls:

* `ToListAsync()`
* `FindAsync()`
* `FirstOrDefaultAsync()`
* `SingleAsync()`
* `SaveChangesAsync()`

Example:

```csharp
public async Task<ActionResult> Students()
{
    var list = await _db.Students.ToListAsync();
    return View(list);
}
```

This releases the thread while waiting for DB.

---

# 💾 4. Async in Service Layer (Industry Standard)

Controller → awaits → Service → awaits → Repository

### Service:

```csharp
public class StudentService : IStudentService
{
    private readonly AppDbContext _db;

    public async Task<List<Student>> GetAllAsync()
    {
        return await _db.Students.ToListAsync();
    }
}
```

### Controller:

```csharp
public async Task<ActionResult> Index()
{
    var data = await _service.GetAllAsync();
    return View(data);
}
```

This is the correct architecture.

---

# 📁 5. Async File Operations

File read:

```csharp
var text = await File.ReadAllTextAsync(path);
```

File write:

```csharp
await File.WriteAllTextAsync(path, data);
```

File copy:

```csharp
await using var stream = new FileStream(path, FileMode.Create);
await file.InputStream.CopyToAsync(stream);
```

---

# 🌐 6. Async API Calls (Common in Real Apps)

Using `HttpClient`:

```csharp
var http = new HttpClient();
var result = await http.GetStringAsync("https://api.example.com/data");
```

This is heavily used in:

* Payment gateways
* Third-party integrations
* Microservices

---

# ⚙️ 7. Common Mistakes to Avoid

### ❌ Mistake 1: Mixing Sync + Async

Bad:

```csharp
var users = _db.Users.ToList();  // sync
var logs = await _db.Logs.ToListAsync();
```

Always keep the flow async.

---

### ❌ Mistake 2: Using `.Result` or `.Wait()`

Bad:

```csharp
var data = http.GetStringAsync(url).Result;
```

This causes:

* Deadlocks
* Blocking
* Crashes under load

Always use `await`.

---

### ❌ Mistake 3: Async void (for non-event methods)

Bad:

```csharp
public async void SaveData()
```

Use:

```csharp
public async Task SaveData()
```

---

# 🧲 8. Real-World Async Pattern (Recommended)

```
Controller → IService → IRepository → EF Core
```

All async.

Example:

### IEntityService

```csharp
Task<List<Student>> GetAllAsync();
Task<Student> GetAsync(int id);
Task CreateAsync(Student student);
```

### Controller

```csharp
public async Task<ActionResult> Details(int id)
{
    var student = await _service.GetAsync(id);
    return View(student);
}
```

---

# 🧩 9. When NOT to use async

❌ For CPU-heavy tasks
❌ When operations are 100% synchronous (pure CPU work)
❌ Small apps with low traffic (but still recommended)

Use async mainly for:

* DB
* File
* Network
* External API
* Cloud storage

---

# 🏆 10. Best Practices (Enterprise-Level)

🔥 Make your entire pipeline async
🔥 Don’t mix sync & async
🔥 Use EF async for all database operations
🔥 Don’t use `.Result` or `.Wait()`
🔥 Use `ConfigureAwait(false)` in libraries (optional)
🔥 Use dependency injection for HttpClient (avoid socket exhaustion)
🔥 Use async file uploads
🔥 Use cancellation tokens for long operations

---

# 🧪 Mini Example — Async CRUD

### Service:

```csharp
public async Task CreateAsync(Student model)
{
    _db.Students.Add(model);
    await _db.SaveChangesAsync();
}
```

### Controller:

```csharp
[HttpPost]
public async Task<ActionResult> Create(StudentVM vm)
{
    if (!ModelState.IsValid)
        return View(vm);

    var student = _mapper.Map<Student>(vm);
    await _service.CreateAsync(student);

    return RedirectToAction("Index");
}
```

---

# 🧩 **Exercise 17 — Convert a Full Module to Async**

Convert your **Product** module into full async:

✔ Async repository
✔ Async service
✔ Async controller actions
✔ Async view rendering
✔ Async DB operations
✔ Validate no sync calls remain

This is exactly what real-world systems do.

---