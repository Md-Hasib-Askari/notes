# Performance & Security (Advanced Notes)

## 1. Profiling & Debugging Tools

* **Runtime tools**: `dotnet-trace`, `dotnet-counters`, `dotnet-dump` (collect traces, counters, memory dumps).
* **Profilers**: Visual Studio Profiler (CPU/memory/async), JetBrains dotTrace/dotMemory, PerfView.
* **Benchmarking**: `BenchmarkDotNet` for micro-benchmarks.
* **Observability**: Structured logging (Serilog/NLog), metrics (Prometheus), tracing (OpenTelemetry). Use Application Insights/Grafana for dashboards.
* **Tips**: Measure before optimizing; watch allocations (boxing, LINQ in hot paths), minimize sync-over-async, pool HttpClient via `IHttpClientFactory`.

---

## 2. Unit Testing (xUnit, NUnit, MSTest)

* **Frameworks**: xUnit (modern default), NUnit (feature-rich), MSTest (Microsoft).
* **Pattern**: Arrange–Act–Assert (AAA), small/fast/isolated.
* **Mocking**: Moq/NSubstitute/FakeItEasy; avoid hitting real DB/network.
* **Examples** (xUnit):

  ```csharp
  public class CalculatorTests
  {
      [Fact]
      public void Add_ReturnsSum() => Assert.Equal(4, new Calc().Add(2,2));
  }
  ```
* **Coverage**: Use coverlet/ReportGenerator; aim for meaningful coverage, not 100%.

---

## 3. Integration & End-to-End Testing

* **ASP.NET Core**: `WebApplicationFactory<TEntry>` + TestServer for in-memory HTTP tests.
* **DB**: Use ephemeral databases (Testcontainers for SQL Server/Postgres) and reset state (Respawn).
* **Messaging**: Spin up Kafka/RabbitMQ via containers for realistic tests.
* **API**: Postman/Newman or REST-assured; verify contracts with Pact (consumer-driven contract testing).
* **UI E2E**: Playwright/Selenium for browser automation; keep flows stable and focused on critical paths.

---

## 4. Secure Coding Practices

* **Input & Output**: Validate/whitelist inputs (FluentValidation); encode output to prevent XSS.
* **AuthN/Z**: Prefer OpenID Connect/OAuth2; use ASP.NET Core Identity for local accounts; apply policy-based authorization.
* **Secrets**: Never hardcode; use User Secrets (dev) and Azure Key Vault/Env vars (prod).
* **Data protection**: Enforce HTTPS/HSTS; use ASP.NET Data Protection APIs for cookies/tokens.
* **Storage**: Hash passwords with strong KDFs (ASP.NET Identity default PBKDF2); encrypt sensitive data at rest (DB TDE, column encryption).
* **SQL Injection**: Always parameterize queries; let EF Core generate SQL.
* **JWT**: Short expirations, issuer/audience validation, rotate/secure signing keys, refresh tokens.
* **Web**: Enable CORS precisely; anti-forgery tokens for state-changing requests; set secure/samesite cookies; rate limit & lockout.
* **Logging**: No secrets/PII; centralize and monitor alerts; keep audit trails.
* **Compliance**: Follow OWASP ASVS/Cheat Sheets; run SAST/DAST (e.g., GitHub CodeQL) and dependency scanning (Dependabot).
