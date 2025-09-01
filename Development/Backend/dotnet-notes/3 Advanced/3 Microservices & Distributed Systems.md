# Microservices & Distributed Systems (Advanced Notes)

## 1. gRPC with .NET

* **gRPC**: High-performance, contract-based communication framework.
* Uses Protocol Buffers (`.proto` files) for defining services.
* Faster than REST for internal microservice communication.

  ```proto
  service Greeter {
    rpc SayHello (HelloRequest) returns (HelloReply);
  }
  ```
* Supports streaming (client, server, bidirectional).

---

## 2. Docker & Kubernetes for .NET Applications

* **Docker**:

  * Containerizes .NET apps for consistency across environments.
  * Example Dockerfile:

    ```dockerfile
    FROM mcr.microsoft.com/dotnet/aspnet:6.0
    COPY . /app
    WORKDIR /app
    ENTRYPOINT ["dotnet", "MyApp.dll"]
    ```
* **Kubernetes**:

  * Orchestrates multiple containers.
  * Features: scaling, load balancing, rolling updates, self-healing.
  * Deploy .NET apps using YAML manifests (`Deployment`, `Service`).

---

## 3. API Gateways & Service Mesh

* **API Gateway**:

  * Central entry point for client requests.
  * Handles routing, rate limiting, authentication, logging.
  * Example: Ocelot for .NET microservices.
* **Service Mesh**:

  * Manages internal microservice-to-microservice communication.
  * Features: traffic routing, retries, observability, security.
  * Examples: Istio, Linkerd.

---

## 4. Event-Driven Architecture (Kafka, RabbitMQ)

* **Event-driven systems** use events for loose coupling.
* **Kafka**:

  * Distributed event streaming platform.
  * High throughput, persistence, partitioning.
* **RabbitMQ**:

  * Message broker with queue-based communication.
  * Supports pub/sub, request/reply.
* Benefits: Scalability, decoupling, asynchronous workflows.
