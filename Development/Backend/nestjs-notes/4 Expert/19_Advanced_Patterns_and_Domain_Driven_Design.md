
## 🟣 **19. Advanced Patterns & Domain-Driven Design (DDD)**

In large-scale, complex applications, using **Domain-Driven Design (DDD)** and **advanced architectural patterns** like CQRS and Event Sourcing can help manage complexity, scalability, and maintainability.

---

### ✅ 1. CQRS (Command Query Responsibility Segregation)

CQRS separates:

* **Commands**: Write operations (create/update/delete).
* **Queries**: Read operations (fetch data).

This allows for:

* Optimized read/write paths.
* Independent scaling.
* Clean separation of concerns.

#### 🔧 Setup with `@nestjs/cqrs`

```bash
npm install @nestjs/cqrs
```

#### Structure:

```
/src
  /todo
    commands/
      create-todo.command.ts
      handlers/create-todo.handler.ts
    queries/
      get-todo.query.ts
      handlers/get-todo.handler.ts
```

```ts
// create-todo.command.ts
export class CreateTodoCommand {
  constructor(public readonly title: string) {}
}
```

```ts
// create-todo.handler.ts
@CommandHandler(CreateTodoCommand)
export class CreateTodoHandler {
  constructor(private readonly repo: TodoRepository) {}

  async execute(command: CreateTodoCommand) {
    const todo = new Todo(command.title);
    return this.repo.save(todo);
  }
}
```

---

### ✅ 2. Event Sourcing

Event sourcing stores the state of a system as a series of events instead of the current state. The state is derived by replaying these events.

Example:

* `UserCreatedEvent`
* `EmailChangedEvent`
* `PasswordUpdatedEvent`

Use an event store like Kafka, EventStoreDB, or even Redis Streams.

---

### ✅ 3. Domain-Driven Design (DDD)

Focus on the **domain model** and ensure it drives the app’s structure.

#### 🔹 Building Blocks:

* **Entities**: Objects with identity (`User`, `Order`)
* **Value Objects**: Immutable, no identity (`Email`, `Money`)
* **Aggregates**: Root entities that enforce consistency boundaries.
* **Repositories**: Abstractions to access aggregates.
* **Services**: Domain logic that doesn't fit in entities.

#### Example: `User` Aggregate

```ts
export class User {
  private constructor(
    private readonly id: string,
    private email: Email,
    private password: Password,
  ) {}

  static create(email: Email, password: Password): User {
    const id = uuid();
    return new User(id, email, password);
  }

  changeEmail(newEmail: Email) {
    this.email = newEmail;
  }
}
```

---

### ✅ 4. Folder Structure Example (DDD + CQRS)

```
/src
  /users
    /domain
      entities/
      value-objects/
    /application
      commands/
      queries/
      services/
    /infrastructure
      repositories/
    user.module.ts
```

This structure helps isolate domain logic, making the codebase more maintainable and testable.

---

### 🧪 Exercise

✅ **Refactor a business-critical module (e.g., Auth or Orders)**:

* Apply CQRS pattern using `@nestjs/cqrs`.
* Structure code using DDD principles.
* Create `domain`, `application`, and `infrastructure` layers.
* Emit events after key actions (e.g., `UserCreatedEvent`).

