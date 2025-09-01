# Design & Architecture (Advanced Notes)

## 1. SOLID Principles

* **S**ingle Responsibility: Each class should have one job.
* **O**pen/Closed: Open for extension, closed for modification.
* **L**iskov Substitution: Subclasses should replace base classes without breaking functionality.
* **I**nterface Segregation: Prefer small, specific interfaces over large general ones.
* **D**ependency Inversion: Depend on abstractions, not concrete implementations.

---

## 2. Design Patterns

* **Singleton**: Ensures a class has only one instance.

  ```csharp
  public class Logger
  {
      private static Logger _instance;
      private Logger() { }
      public static Logger Instance => _instance ??= new Logger();
  }
  ```
* **Repository**: Abstracts data access logic from business logic.
* **Factory**: Centralizes object creation.
* **Observer**: Publisher-subscriber model for event-driven systems.
* **Adapter**: Converts interface of one class into another expected by the client.

---

## 3. Clean Architecture

* Emphasizes separation of concerns.
* Layers:

  * **Domain**: Core business rules (entities).
  * **Application**: Use cases, services.
  * **Infrastructure**: Database, external services.
  * **Presentation**: UI/API.
* Benefits: Testability, scalability, flexibility.

---

## 4. Domain-Driven Design (DDD)

* Focus on the **domain model** as the core of the application.
* Uses a **Ubiquitous Language** shared by developers and domain experts.
* Key building blocks:

  * **Entities**: Objects with identity (e.g., `User`).
  * **Value Objects**: Immutable, defined by values (e.g., `Money`).
  * **Aggregates**: Clusters of entities with a root.
  * **Repositories**: Access to aggregates.
  * **Services**: Domain logic that doesn’t fit entities/values.
* Encourages modular and business-focused systems.
