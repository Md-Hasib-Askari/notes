# ✅ **Module 16 — File Uploads, Downloads, Images, Import/Export**

You’ll learn how to safely handle:

✔ File uploads
✔ Image uploads (profile pictures, product images)
✔ File downloads
✔ Export to PDF, Excel, CSV
✔ Importing data from CSV/Excel
✔ Storage best practices
✔ Security rules for file handling

Let’s break it down.

---

# 🔥 1. File Upload Basics (Beginner → Intermediate)

## Step 1 — Add `<input type="file">` in View

```html
<form action="/files/upload" method="post" enctype="multipart/form-data">
    <input type="file" name="file" />
    <button type="submit">Upload</button>
</form>
```

> Important: `enctype="multipart/form-data"` is mandatory.

---

## Step 2 — Controller Action

```csharp
[HttpPost]
public ActionResult Upload(HttpPostedFileBase file)
{
    if (file != null && file.ContentLength > 0)
    {
        var path = Server.MapPath("~/Uploads/" + file.FileName);
        file.SaveAs(path);
    }

    return Content("Uploaded!");
}
```

This is the simplest upload.

---

# 🔒 2. Secure File Uploads (Industry Standard)

Attacks can happen if you don't validate files.

### Always validate:

✔ File extension
✔ Content type
✔ File size
✔ Rename file to prevent script execution

Example:

```csharp
var allowed = new[] { ".jpg", ".png", ".pdf" };
var ext = Path.GetExtension(file.FileName).ToLower();

if (!allowed.Contains(ext))
{
    return Content("File type not allowed");
}

if (file.ContentLength > 5 * 1024 * 1024) // 5MB
{
    return Content("File too large");
}
```

---

# 🖼 3. Image Upload (Profile Pictures, Products)

Save with unique filename:

```csharp
var fileName = Guid.NewGuid() + Path.GetExtension(file.FileName);
file.SaveAs(Server.MapPath("~/Images/" + fileName));
```

Store fileName in database, not the actual file.

---

# 📥 4. File Download (Simple & Clean)

```csharp
public FileResult Download(string filename)
{
    var path = Server.MapPath("~/Uploads/" + filename);
    return File(path, MimeMapping.GetMimeMapping(path), filename);
}
```

---

# 📤 5. Export to CSV (Very Common)

```csharp
public FileResult ExportCsv()
{
    var students = _db.Students.ToList();

    var sb = new StringBuilder();
    sb.AppendLine("Id,Name,Department");

    foreach (var s in students)
    {
        sb.AppendLine($"{s.Id},{s.Name},{s.Department}");
    }

    return File(Encoding.UTF8.GetBytes(sb.ToString()),
                "text/csv",
                "students.csv");
}
```

---

# 📊 6. Export to Excel (Enterprise Feature)

Using **EPPlus** (most common lib):

Install:

```
Install-Package EPPlus
```

Example:

```csharp
public FileResult ExportExcel()
{
    using (var excel = new ExcelPackage())
    {
        var ws = excel.Workbook.Worksheets.Add("Students");
        ws.Cells["A1"].LoadFromCollection(_db.Students.ToList(), true);
        
        var bytes = excel.GetAsByteArray();
        return File(bytes, 
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
            "students.xlsx");
    }
}
```

---

# 📄 7. Export to PDF (Using Rotativa)

Install Rotativa:

```
Install-Package Rotativa
```

Now export any view as PDF:

```csharp
public ActionResult ExportPdf(int id)
{
    return new Rotativa.ActionAsPdf("Details", new { id = id });
}
```

Companies use this for:

* Invoices
* Reports
* Certificates
* Receipts

---

# 📥 8. Import CSV (Bulk Upload)

```csharp
[HttpPost]
public ActionResult ImportCsv(HttpPostedFileBase file)
{
    using (var reader = new StreamReader(file.InputStream))
    {
        while (!reader.EndOfStream)
        {
            var line = reader.ReadLine();
            var data = line.Split(',');

            var student = new Student
            {
                Id = int.Parse(data[0]),
                Name = data[1],
                Department = data[2]
            };

            _db.Students.Add(student);
        }

        _db.SaveChanges();
    }

    return RedirectToAction("Index");
}
```

---

# 📦 9. Storing Files (Local vs Cloud)

### Local (simple apps):

```
~/Uploads/
~/Images/
```

### Cloud (professional apps):

* AWS S3
* Azure Blob Storage
* Google Cloud Storage

Cloud storage pros:

* Faster
* Cheaper at scale
* Huge capacity
* Globally accessible

Large companies avoid file uploads on the server itself.

---

# 🔐 10. Security Best Practices (Enterprise-Level)

🔥 Never trust file extensions
🔥 Validate MIME type
🔥 Sanitize file names
🔥 Always rename uploaded files
🔥 Store uploads outside `/wwwroot` if sensitive
🔥 Limit file size
🔥 Strictly whitelist file types
🔥 Scan files for malware (if enterprise-level)

---

# 🧪 Mini Example — Profile Picture Upload

### Controller:

```csharp
public ActionResult UploadProfilePicture(HttpPostedFileBase picture)
{
    if (picture == null) return Content("No file");

    var ext = Path.GetExtension(picture.FileName).ToLower();

    if (ext != ".jpg" && ext != ".png")
        return Content("Only JPG/PNG allowed");

    var fileName = Guid.NewGuid() + ext;
    
    picture.SaveAs(Server.MapPath("~/Images/Profiles/" + fileName));

    // Save filename in user's profile
    // _service.UpdateProfilePicture(userId, fileName);

    return RedirectToAction("Profile");
}
```

---

# 🧩 **Exercise 16 — Implement a Complete File Handling Feature**

Build a **Document Manager Module**:

### Features to implement:

✔ Upload documents (pdf, jpg, png, docx)
✔ Validate size ≤ 5MB
✔ Rename files with GUID
✔ Show file list
✔ Download files
✔ Delete files
✔ Display image preview for images
✔ Store metadata (file name, size, upload date) in DB

Bonus:

* Implement pagination for document list
* Implement Excel export of all documents

This module gives you real-world MVC project experience.

---