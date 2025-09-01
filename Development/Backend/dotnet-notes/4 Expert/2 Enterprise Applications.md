# Enterprise Applications (Expert Notes)

## 1. Multi-Tenant SaaS Apps

* **Multi-tenancy**: One application serves multiple customers (tenants).
* **Models**:

  * **Database-per-tenant**: Isolated DB for each customer (better security, harder to scale).
  * **Shared DB, separate schema**: Balance between isolation and efficiency.
  * **Shared DB, shared schema**: Highest efficiency, requires strong row-level security.
* **Considerations**:

  * Tenant isolation, security.
  * Customization per tenant (branding, features).
  * Billing, provisioning, onboarding automation.

---

## 2. High Availability & Scaling

* **High Availability (HA)**:

  * Deploy apps in multiple regions/availability zones.
  * Use load balancers and health checks.
  * Automatic failover (e.g., SQL Always On, Cosmos DB multi-region).
* **Scaling**:

  * **Vertical scaling**: Add more resources to one machine (limited).
  * **Horizontal scaling**: Add more instances/servers (cloud-friendly).
  * Use Kubernetes, Azure App Service autoscaling, or container orchestrators.

---

## 3. Distributed Caching (Redis, NCache)

* **Redis**:

  * In-memory, key-value store.
  * Supports strings, lists, sets, hashes.
  * Commonly used for session storage, caching DB queries, pub/sub.
  * Cloud-hosted options: Azure Cache for Redis.
* **NCache**:

  * .NET-focused distributed cache.
  * High performance, supports object caching and clustering.
* **Benefits of distributed caching**:

  * Reduce database load.
  * Faster response times.
  * Scalability in high-traffic enterprise apps.
