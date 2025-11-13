## 🧠 **1. What is Regex in .NET Annotations**

In .NET, regex (regular expressions) is commonly used with the **`[RegularExpression]`** data annotation attribute to validate strings that must match a specific pattern.

This is part of the **System.ComponentModel.DataAnnotations** namespace and is typically used in **ASP.NET MVC** or **Razor Pages models**.

---

## ⚙️ **2. Basic Syntax**

```csharp
[RegularExpression("pattern", ErrorMessage = "Custom error message")]
public string PropertyName { get; set; }
```

* `pattern`: the regex pattern to match
* `ErrorMessage`: optional; displays validation feedback to the user

---

## 💡 **3. Common Use Cases**

Here are practical regex patterns for common validation tasks.

### ✅ **a. Email Validation**

```csharp
[RegularExpression(@"^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$", 
    ErrorMessage = "Invalid email format")]
public string Email { get; set; }
```

✔️ Accepts `abc@example.com`, `user.name@domain.co.uk`

---

### ✅ **b. Phone Number Validation**

```csharp
[RegularExpression(@"^\+?\d{10,15}$", 
    ErrorMessage = "Enter a valid phone number")]
public string PhoneNumber { get; set; }
```

✔️ Accepts `+8801234567890`, `01712345678`

---

### ✅ **c. Password Strength Validation**

```csharp
[RegularExpression(@"^(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&]).{8,}$", 
    ErrorMessage = "Password must be at least 8 characters, include one uppercase letter, one number, and one special character.")]
public string Password { get; set; }
```

✔️ Matches `Abcd@123`

---

### ✅ **d. Postal Code Validation (e.g., Bangladesh)**

```csharp
[RegularExpression(@"^\d{4}$", ErrorMessage = "Enter a valid 4-digit postal code.")]
public string PostalCode { get; set; }
```

✔️ Matches `1207`

---

### ✅ **e. Username (Alphanumeric Only)**

```csharp
[RegularExpression(@"^[a-zA-Z0-9_]{3,16}$", 
    ErrorMessage = "Username must be 3–16 characters and alphanumeric only.")]
public string Username { get; set; }
```

✔️ Matches `Hasib_01`, ❌ `hasib!@#`

---

## 🔍 **4. Explanation of Common Regex Tokens**

| Token   | Meaning                             | Example                            |
| ------- | ----------------------------------- | ---------------------------------- |
| `^`     | Start of string                     | `^abc` → must start with "abc"     |
| `$`     | End of string                       | `xyz$` → must end with "xyz"       |
| `.`     | Any character except newline        | `a.b` → matches "acb", "a0b"       |
| `\d`    | Digit (0–9)                         | `\d{4}` → 4 digits                 |
| `\w`    | Word character (A–Z, a–z, 0–9, _)   | `\w+` → one or more word chars     |
| `[A-Z]` | Character class (uppercase letters) | `[A-Z]{2}` → two uppercase letters |
| `()`    | Grouping                            | `(abc)+` → one or more "abc"       |
| `?`     | Optional                            | `colou?r` → "color" or "colour"    |
| `+`     | One or more                         | `\d+` → one or more digits         |
| `*`     | Zero or more                        | `\d*` → any number of digits       |
| `{n,m}` | Range                               | `\d{4,6}` → 4 to 6 digits          |

---

## 🧩 **5. Using Regex Annotation with MVC Forms**

Example model:

```csharp
public class RegisterModel
{
    [Required]
    [RegularExpression(@"^[A-Za-z\s]+$", ErrorMessage = "Name must contain letters only.")]
    public string FullName { get; set; }

    [RegularExpression(@"^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$", ErrorMessage = "Invalid email format.")]
    public string Email { get; set; }

    [RegularExpression(@"^(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&]).{8,}$", ErrorMessage = "Weak password.")]
    public string Password { get; set; }
}
```

Controller Action:

```csharp
[HttpPost]
public IActionResult Register(RegisterModel model)
{
    if (ModelState.IsValid)
    {
        // Proceed with registration
        return RedirectToAction("Success");
    }
    return View(model);
}
```

---

## 🧠 **6. Testing Regex Before Use**

You can test and refine regex patterns using tools like:

* 🔗 [regex101.com](https://regex101.com)
* 🔗 [regexr.com](https://regexr.com)

Always validate with real sample data before finalizing patterns.

---

## 🧪 **7. Exercise Ideas**

Try implementing these:

1. Validate Bangladeshi NID number (`10–17 digits only`).
2. Validate vehicle registration (like `DHAKA-METRO-BA-12-3456`).
3. Validate date format (`dd/mm/yyyy`).
4. Validate URL input.

Example challenge:

```csharp
[RegularExpression(@"^\d{10,17}$", ErrorMessage = "Invalid NID format.")]
public string NID { get; set; }
```

---

## 🧾 **8. Key Takeaways**

* Use `[RegularExpression]` for client + server validation.
* Always anchor regex with `^` and `$` for strict matching.
* Keep regex simple; use custom validation if the pattern is too complex.
* Combine with `[Required]`, `[StringLength]`, or `[Compare]` for better coverage.


# 50 regex pattern examples

Each includes:
✅ **Pattern** | 🧠 **Purpose** | 💬 **Example Match**

---

## 🔤 **1. Basic Text Patterns**

| #  | Regex               | Purpose                      | Example Match   |
| -- | ------------------- | ---------------------------- | --------------- |
| 1  | `^[A-Za-z]+$`       | Only letters                 | `Hasib`         |
| 2  | `^[A-Za-z\s]+$`     | Letters and spaces           | `Hasib Khan`    |
| 3  | `^[A-Za-z0-9]+$`    | Letters and digits           | `Hasib123`      |
| 4  | `^[A-Za-z0-9_]+$`   | Alphanumeric with underscore | `Hasib_01`      |
| 5  | `^[A-Za-z0-9_.-]+$` | Allow `.`, `_`, `-`          | `hasib.khan-01` |
| 6  | `^[^0-9]+$`         | No digits allowed            | `NoNumbers`     |
| 7  | `^[^A-Za-z]+$`      | No letters                   | `123_@`         |
| 8  | `^[A-Z]+$`          | Uppercase only               | `HELLO`         |
| 9  | `^[a-z]+$`          | Lowercase only               | `hello`         |
| 10 | `^[A-Za-z]{5,10}$`  | Letters, length 5–10         | `HasibKhan`     |

---

## 🔢 **2. Numbers and Digits**

| #  | Regex               | Purpose                             | Example Match       |
| -- | ------------------- | ----------------------------------- | ------------------- |
| 11 | `^\d+$`             | Only digits                         | `12345`             |
| 12 | `^\d{4}$`           | Exactly 4 digits                    | `1207`              |
| 13 | `^\d{10,17}$`       | Bangladeshi NID length              | `12345678901234567` |
| 14 | `^-?\d+$`           | Integer (with optional minus)       | `-42`, `99`         |
| 15 | `^\d+\.\d+$`        | Decimal number                      | `3.1416`            |
| 16 | `^\+?\d{10,15}$`    | Phone number                        | `+8801712345678`    |
| 17 | `^\d{1,3}$`         | Range-limited (1–3 digits)          | `99`                |
| 18 | `^\d{2,4}-\d{2,4}$` | Range format                        | `1999-2024`         |
| 19 | `^[1-9]\d*$`        | Positive integer (no leading zeros) | `42`                |
| 20 | `^0\d+$`            | Leading zero number                 | `0123`              |

---

## 🗓️ **3. Dates and Times**

| #  | Regex                 | Purpose               | Example Match        |                |                       |              |
| -- | --------------------- | --------------------- | -------------------- | -------------- | --------------------- | ------------ |
| 21 | `^\d{2}/\d{2}/\d{4}$` | Date (dd/mm/yyyy)     | `12/11/2025`         |                |                       |              |
| 22 | `^\d{4}-\d{2}-\d{2}$` | ISO date (yyyy-mm-dd) | `2025-11-12`         |                |                       |              |
| 23 | `^\d{2}:\d{2}$`       | Time (hh:mm)          | `23:59`              |                |                       |              |
| 24 | `^(0[1-9]             | 1[0-2])/(0[1-9]       | [12]\d               | 3[01])/\d{4}$` | Valid month/day range | `11/12/2025` |
| 25 | `^(0[0-9]             | 1[0-9]                | 2[0-3]):[0-5][0-9]$` | 24-hour time   | `14:45`               |              |

---

## 💬 **4. Email, Username, and Passwords**

| #  | Regex                                              | Purpose                           | Example Match         |
| -- | -------------------------------------------------- | --------------------------------- | --------------------- |
| 26 | `^[\w.-]+@([\w-]+\.)+[A-Za-z]{2,10}$`              | Email (modern)                    | `hasib.khan@aiub.edu` |
| 27 | `^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$` | RFC-like email                    | `user+dev@gmail.com`  |
| 28 | `^[\w.-]+$`                                        | Simple username                   | `hasib_khan-01`       |
| 29 | `^[A-Za-z][A-Za-z0-9_]{2,15}$`                     | Username (must start with letter) | `Hasib01`             |
| 30 | `^(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&]).{8,}$`        | Strong password                   | `Abc@1234`            |

---

## 📦 **5. Files, Paths, and Extensions**

| #  | Regex                                   | Purpose                         | Example Match      |                           |                      |             |
| -- | --------------------------------------- | ------------------------------- | ------------------ | ------------------------- | -------------------- | ----------- |
| 31 | `^[\w,\s-]+\.[A-Za-z]{3}$`              | File name with 3-char extension | `report_2025.pdf`  |                           |                      |             |
| 32 | `^.*.(jpg                               | jpeg                            | png                | gif)$`                    | Image file extension | `photo.png` |
| 33 | `^[A-Za-z]:[\\S                         | *\S]?.*$`                       | Windows path       | `C:\Users\Hasib\file.txt` |                      |             |
| 34 | `^/([A-Za-z0-9-_+]+/)*[A-Za-z0-9-_+]+$` | Linux-style path                | `/home/hasib/docs` |                           |                      |             |

---

## 🌐 **6. URLs, Domains, and IPs**

| #  | Regex                                           | Purpose                | Example Match          |
| -- | ----------------------------------------------- | ---------------------- | ---------------------- |
| 35 | `^https?:\/\/[A-Za-z0-9.-]+\.[A-Za-z]{2,}$`     | HTTP/HTTPS URL         | `https://example.com`  |
| 36 | `^https?:\/\/([\w-]+\.)+[\w-]{2,10}(/[\w-]*)*$` | URL with optional path | `https://aiub.edu/cse` |
| 37 | `^www\.[A-Za-z0-9.-]+\.[A-Za-z]{2,}$`           | Starts with `www.`     | `www.google.com`       |
| 38 | `^\d{1,3}(\.\d{1,3}){3}$`                       | IPv4 address           | `192.168.0.1`          |
| 39 | `^\[[A-Fa-f0-9:]+\]$`                           | IPv6 address           | `[2001:0db8::1]`       |
| 40 | `^([A-Za-z0-9-]+\.)+[A-Za-z]{2,}$`              | Domain name            | `sub.domain.org`       |

---

## 💳 **7. IDs and Codes**

| #  | Regex                       | Purpose                     | Example Match         |
| -- | --------------------------- | --------------------------- | --------------------- |
| 41 | `^[A-Z]{2}\d{6}$`           | Passport code format        | `AB123456`            |
| 42 | `^[A-Z]{3}-\d{3}$`          | Product code                | `ABC-123`             |
| 43 | `^\d{4}-\d{4}-\d{4}-\d{4}$` | Credit card (basic pattern) | `1234-5678-9101-1121` |
| 44 | `^[A-Fa-f0-9]{8}$`          | Hex code (8 chars)          | `1A2B3C4D`            |
| 45 | `^#[A-Fa-f0-9]{6}$`         | Hex color                   | `#FF00AA`             |

---

## 🧾 **8. Miscellaneous Patterns**

| #  | Regex                       | Purpose                    | Example Match        |                             |      |          |              |                |
| -- | --------------------------- | -------------------------- | -------------------- | --------------------------- | ---- | -------- | ------------ | -------------- |
| 46 | `^[A-Za-z0-9\s,.'-]{3,}$`   | Person’s name (flexible)   | `Hasib Uddin Khan`   |                             |      |          |              |                |
| 47 | `^[A-Za-z0-9\s,'-]{5,100}$` | Address validation         | `123 Main Street`    |                             |      |          |              |                |
| 48 | `^(?:Yes                    | No                         | Y                    | N                           | True | False)$` | Boolean text | `Yes`, `False` |
| 49 | `^[1-9][0-9]?$              | ^100$`                     | Percent (0–100)      | `85`                        |      |          |              |                |
| 50 | `^(https?                   | ftp)://[^\s/$.?#].[^\s]*$` | General URL (robust) | `ftp://server.com/file.txt` |      |          |              |                |

---

## ⚙️ **Bonus Tip for .NET**

In your model:

```csharp
[RegularExpression(@"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", 
    ErrorMessage = "Invalid email format.")]
public string Email { get; set; }
```

Regex annotations are evaluated automatically during model validation in:

* ASP.NET MVC
* Razor Pages
* Blazor
