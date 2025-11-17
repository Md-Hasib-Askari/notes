# ✅ **Module 15 — ViewModels & AutoMapper (Advanced UI Handling)**

Goals:
✔ Learn why ViewModels are mandatory
✔ Protect your app from over-posting
✔ Cleanly map Entity ↔ ViewModel
✔ Use AutoMapper like real companies
✔ Reduce boilerplate data transformation code

---

# 🔥 1. Why ViewModels Are Required (Not Optional)

### ❌ Wrong approach (beginner mistake):

Returning EF entity directly to View:

```csharp
return View(student);
```

Problems:

* Exposes DB fields
* Over-posting attacks
* Tight coupling
* Hard to maintain
* You often send more data than needed

---

### ✔ Correct approach (industry standard):

Use a **ViewModel** that contains only what the View needs.

```csharp
public class StudentDetailsVM
{
    public string Name { get; set; }
    public string Department { get; set; }
}
```

ViewModels improve:

* Security
* Maintainability
* Flexibility
* Performance (smaller payload)

---

# 🧱 2. ViewModel Example (Clean + Safe)

### Entity (Database model):

```csharp
public class Student
{
    public int Id { get; set; }
    public string Name { get; set; }
    public string Department { get; set; }
    public decimal GPA { get; set; }
    public DateTime CreatedAt { get; set; }
}
```

### ViewModel (for display only):

```csharp
public class StudentListVM
{
    public string Name { get; set; }
    public string Department { get; set; }
}
```

This is **perfect separation of layers**.

---

# ⚠️ 3. Over-Posting Protection (Very Important)

Suppose your form:

```html
<input name="Name" />
<input name="Department" />
```

If your controller receives the **entity**:

```csharp
public ActionResult Edit(Student model)
```

A hacker can inject fields like:

```
GPA=4.0
CreatedAt=2001-01-01
```

Because MVC model binding fills **all matching fields**.

To prevent this, we use a dedicated **EditStudentVM** that only includes fields you allow editing.

---

# 🧠 4. AutoMapper — The Ultimate Time Saver

AutoMapper automatically maps entity <→ ViewModel.

### Install:

```
Install-Package AutoMapper
Install-Package AutoMapper.Extensions.Microsoft.DependencyInjection
```

---

## ⭐ Step 1 — Create a Mapping Profile

```csharp
public class MappingProfile : Profile
{
    public MappingProfile()
    {
        CreateMap<Student, StudentDetailsVM>();
        CreateMap<CreateStudentVM, Student>();
    }
}
```

This tells AutoMapper how to convert objects.

---

## ⭐ Step 2 — Register AutoMapper in Startup

```csharp
var config = new MapperConfiguration(cfg =>
{
    cfg.AddProfile(new MappingProfile());
});

var mapper = config.CreateMapper();
```

Or via DI container (recommended).

---

## ⭐ Step 3 — Inject IMapper into Controller

```csharp
private readonly IMapper _mapper;

public StudentController(IMapper mapper)
{
    _mapper = mapper;
}
```

---

## ⭐ Step 4 — Use AutoMapper

### Entity → ViewModel:

```csharp
var vm = _mapper.Map<StudentDetailsVM>(student);
return View(vm);
```

### ViewModel → Entity:

```csharp
var entity = _mapper.Map<Student>(createVM);
```

No more manual mapping.

---

# 🧪 5. Manual Mapping (Without AutoMapper)

Sometimes needed for complex situations.

```csharp
var vm = new StudentDetailsVM
{
    Name = student.Name,
    Department = student.Department
};
```

This is ok for tiny apps, but AutoMapper saves time at scale.

---

# 🧲 6. AutoMapper Advanced Features

### ✔ Ignore fields:

```csharp
ForMember(dest => dest.CreatedAt, opt => opt.Ignore());
```

### ✔ Conditional mapping:

```csharp
ForMember(dest => dest.Name,
          opt => opt.Condition(src => src.Name != null));
```

### ✔ Custom transformations:

```csharp
ForMember(dest => dest.FullName,
          opt => opt.MapFrom(src => src.FirstName + " " + src.LastName));
```

### ✔ Complex object graph mapping:

* AutoMapper handles nested objects
* Collections
* Child entities

Perfect for dashboards & admin views.

---

# 🧩 7. Industry Best Practices

🔥 Never pass EF entities to Views
🔥 Use ViewModels for all UI pages
🔥 Use different ViewModels for create/edit/detail
🔥 Keep ViewModel small and purpose-specific
🔥 Use AutoMapper for medium/large projects
🔥 Map inside services, not controllers
🔥 Validate ViewModels, not entities

---

# 🧪 Mini Example — Complete Mapping

### ViewModel (Create Student)

```csharp
public class CreateStudentVM
{
    public string Name { get; set; }
    public string Department { get; set; }
}
```

### Controller

```csharp
[HttpPost]
public ActionResult Create(CreateStudentVM model)
{
    if (!ModelState.IsValid) return View(model);

    var entity = _mapper.Map<Student>(model);

    _service.Create(entity);

    return RedirectToAction("Index");
}
```

This is **clean, safe, and professional**.

---

# 🧩 **Exercise 15 — Build a Full VM + AutoMapper Flow**

Create:

1. **Entity:** Product

   * Id, Name, Price, Description, CreatedAt

2. **ViewModels:**

   * `ProductListVM` (Name, Price)
   * `ProductDetailsVM` (Name, Price, Description)
   * `CreateProductVM` (Name, Price, Description)

3. Configure AutoMapper

4. Build controller actions using ViewModels

5. Test all CRUD flows

You will now build UI like a real company.

---