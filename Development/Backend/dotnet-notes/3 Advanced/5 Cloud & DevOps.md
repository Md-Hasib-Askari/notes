# Cloud & DevOps (Advanced Notes)

## 1. Azure Services

* **Azure App Service**: PaaS for hosting web apps/APIs without managing infrastructure.
* **Azure Functions**: Serverless compute for event-driven workloads (pay-per-execution).
* **Cosmos DB**: Globally distributed NoSQL DB with multi-model APIs (SQL, MongoDB, Gremlin).
* **Azure DevOps Pipelines**: End-to-end CI/CD service integrated with repos, artifacts, and test plans.

---

## 2. CI/CD Pipelines

* **GitHub Actions**:

  * Workflow YAML for build/test/deploy.
  * Strong integration with GitHub repos.
* **Jenkins**:

  * Open-source automation server.
  * Plugins for .NET builds, Docker, Kubernetes.
* **Azure DevOps**:

  * Hosted pipelines with pre-built agents.
  * YAML or classic pipelines for builds and releases.
* Best Practices:

  * Automate tests in pipeline.
  * Use environments & approvals.
  * Secure secrets with Key Vault/Secrets Manager.

---

## 3. Logging & Monitoring

* **Serilog**: Structured logging library for .NET.

  ```csharp
  Log.Logger = new LoggerConfiguration()
      .WriteTo.Console()
      .WriteTo.File("logs/log.txt")
      .CreateLogger();
  ```
* **ELK Stack** (Elasticsearch, Logstash, Kibana): Centralized log storage, processing, visualization.
* **Application Insights** (Azure): Telemetry for performance, availability, usage.
* **Best Practices**:

  * Correlate logs with request IDs.
  * Collect metrics (CPU, memory, latency).
  * Configure alerts and dashboards for proactive monitoring.
