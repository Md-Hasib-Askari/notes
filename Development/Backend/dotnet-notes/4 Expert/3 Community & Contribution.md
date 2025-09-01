# Community & Contribution (Expert Notes)

## 1. Reading .NET Runtime Source Code

* .NET runtime (CoreCLR, libraries, Roslyn) is fully open-source on GitHub.
* Explore source code for deeper understanding of internals.
* Benefits:

  * Learn advanced C# patterns and optimization.
  * Understand how the runtime manages memory, JIT compilation, GC.
  * Improves debugging skills for complex issues.

---

## 2. Building NuGet Packages

* **NuGet** is the standard package manager for .NET.
* Steps to create a package:

  * Create a class library project.
  * Add metadata in `.csproj` (package id, version, author).
  * Pack with CLI:

    ```bash
    dotnet pack --configuration Release
    ```
  * Publish:

    ```bash
    dotnet nuget push bin/Release/MyLib.nupkg -k <API_KEY> -s https://api.nuget.org/v3/index.json
    ```
* Useful for sharing libraries across projects or contributing to the ecosystem.

---

## 3. Contributing to Open-Source .NET Projects

* Microsoft hosts .NET repos on GitHub: `dotnet/runtime`, `aspnet/AspNetCore`, `dotnet/efcore`.
* Contribution steps:

  * Fork, clone, create branch.
  * Implement fix/feature, write tests, update docs.
  * Submit Pull Request (PR).
* Benefits:

  * Gain recognition, improve skills.
  * Collaborate with the global .NET community.

---

## 4. Following Microsoft’s Official .NET Release Roadmap

* Microsoft publishes a public roadmap on [dotnet.microsoft.com](https://dotnet.microsoft.com).
* **LTS (Long-Term Support) releases** every 2 years (e.g., .NET 6, .NET 8).
* **STS (Standard-Term Support)** releases every year.
* Stay updated with:

  * New language features (C# versions).
  * Performance improvements.
  * Ecosystem tooling and libraries.
* Engaging with roadmap helps plan migrations and leverage new features early.
