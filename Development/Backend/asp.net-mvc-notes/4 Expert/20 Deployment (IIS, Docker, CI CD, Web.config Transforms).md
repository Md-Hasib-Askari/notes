# ✅ **Module 20 — Deployment (IIS, Docker, CI/CD, Web.config Transforms)**

You will learn:

✔ Deploying to IIS
✔ Configuring web.config for production
✔ Using Web.config transforms
✔ Publishing with Visual Studio
✔ Running MVC apps in Docker
✔ CI/CD pipelines
✔ Environment variables
✔ Debug vs Release builds
✔ How companies deploy large systems

This is a must-have skill.

---

# 🔥 1. Deploying to IIS (Most Common Production Setup)

### Step 1 — Enable IIS

Windows Features →

* Internet Information Services
* ASP.NET
* .NET Extensibility
* Static Content
* Request Filtering
* Windows Authentication (optional)

### Step 2 — Create a new IIS Site

In IIS Manager:

* Right-click **Sites → Add Website**
* Select physical path: your published folder
* Bind host + port

### Step 3 — Set Correct Application Pool

Use:

```
.NET CLR Version: v4.0
Managed Pipeline: Integrated
```

### Step 4 — Give folder permissions

IIS needs access to your published folder.
Give these permissions to the folder:

✔ IIS_IUSRS
✔ IUSR

Otherwise you’ll get **401 / 500 errors**.

---

# 🧱 2. Publishing from Visual Studio

Choose:

```
Build → Publish
```

You can publish to:

* Folder
* FTP
* Web Deploy
* Azure
* IIS directly

Common method: **publish to folder**, then upload to server.

---

# ⚙️ 3. Web.config Transforms (Production Settings)

This is **essential industry practice**.

You maintain:

* `Web.Debug.config`
* `Web.Release.config`

### Example Transform — Change connection string for production

```xml
<connectionStrings>
  <add name="AppDbContext"
       connectionString="PRODUCTION_CONNECTION_STRING"
       xdt:Transform="SetAttributes" xdt:Locator="Match(name)" />
</connectionStrings>
```

### Example — Disable debug mode

```xml
<compilation xdt:Transform="SetAttributes" debug="false" />
```

### Example — Enable custom errors

```xml
<customErrors mode="RemoteOnly" xdt:Transform="SetAttributes" />
```

Production builds automatically apply `Web.Release.config`.

---

# 🔒 4. Environment Variables (Don’t Hardcode Secrets)

Never store:

* DB passwords
* API keys
* JWT secrets

Instead, store in server environment variables or Azure Key Vault.

Example:

```xml
<add key="PaymentApiKey" value="#{PaymentApiKey}#" />
```

Then replace via CI/CD pipeline.

---

# 🐳 5. Running ASP.NET MVC in Docker (Modern Deployment)

### Step 1 — Create a Dockerfile

```dockerfile
FROM mcr.microsoft.com/dotnet/framework/aspnet:4.8
COPY . /inetpub/wwwroot
```

### Step 2 — Build Image

```
docker build -t mvc-app .
```

### Step 3 — Run Container

```
docker run -d -p 8080:80 mvc-app
```

Now your MVC app runs inside Docker.

---

# 🚀 6. CI/CD Pipelines (Industry-Level Deployment)

Companies use:

* GitHub Actions
* Azure DevOps Pipelines
* GitLab CI
* Jenkins
* Bitbucket Pipelines

Basic CI/CD tasks:

✔ Build project
✔ Run tests
✔ Transform web.config
✔ Publish artifacts
✔ Deploy to server (IIS, Azure, Docker, etc.)

### Example GitHub Actions Workflow (Simple)

```yaml
name: Build MVC

on: [push]

jobs:
  build:
    runs-on: windows-latest
    steps:
    - uses: actions/checkout@v2
    - name: Setup MSBuild
      uses: microsoft/setup-msbuild@v1
    - name: Build
      run: msbuild MyApp.sln /p:Configuration=Release
```

---

# 🔥 7. Production Hardening (Deployment Security)

✔ Turn off debug mode
✔ Enable custom errors
✔ Remove stack traces
✔ Make sure HTTPS is forced
✔ Add HSTS header
✔ Apply file size limits
✔ Disable directory browsing
✔ Add request filtering
✔ Enable logging
✔ Rotate log files

---

# 🌍 8. Hosting Choices in Real World

### Cheapest Hosting

* Shared hosting (not recommended for scaling)

### Robust & Professional

* A dedicated Windows Server
* Azure Web App
* AWS EC2 + Windows
* Docker + Kubernetes
* Virtual Machines

### Enterprise-level

* Azure App Service
* Azure VM Scale Sets
* On-premises Windows Server clusters

---

# 🧠 9. Common Deployment Issues (And Fixes)

### ❌ 500 Internal Server Error

✔ Incorrect folder permissions
✔ App pool misconfigured
✔ Missing .NET Framework
✔ Missing dependencies

### ❌ 404 Not Found

✔ Routes not configured
✔ Wrong virtual directory

### ❌ SQL connection fails

✔ Wrong connection string
✔ Firewall blocking SQL port

### ❌ CSS/JS not loading

✔ bundling/minification config
✔ wrong relative paths

These issues are normal — developers deal with them constantly.

---

# 🧪 Mini Example — Full Production Deployment Checklist

Before deploying:

✔ Set debug = false
✔ Enable custom error pages
✔ Apply Release web.config transforms
✔ Ensure HTTPS redirect
✔ Compress static content
✔ Bundle/minify CSS & JS
✔ Enable output caching
✔ Configure logging (Serilog/NLog)
✔ Disable directory browsing
✔ Test all admin pages
✔ Run full regression tests
✔ Validate performance (Chrome Lighthouse)

This checklist is exactly what companies use.

---

# 🧩 **Exercise 20 — Deploy a Real MVC App**

Your task:

1. Build a simple product management module
2. Publish using Release mode
3. Deploy to local IIS
4. Use web.config transform
5. Enable HTTPS
6. Log errors to file
7. Add custom 404/500 pages
8. Try deploying inside Docker

Once you complete this, you are **deployment-ready**.

---