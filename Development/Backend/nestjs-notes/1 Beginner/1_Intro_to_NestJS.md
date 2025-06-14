

## 🟢 **Beginner Level**

### ✅ **1. Introduction to NestJS**

---

### 🔹 What is NestJS?

* A progressive Node.js framework for building efficient and scalable server-side applications.
* Built on top of **Express** (default) or optionally **Fastify**.
* Uses **TypeScript** heavily.
* Inspired by Angular’s architecture (decorators, DI, modules).

---

### 🔹 Key Features

* **Modular architecture**: Encourages clean and maintainable code structure.
* **Dependency Injection (DI)**: Promotes testability and flexibility.
* **Scalable**: Built with microservices support and modular components.
* **Built-in tools** for validation, serialization, guards, interceptors, etc.

---

### 🔹 NestJS vs Express

| Feature        | NestJS               | Express                |
| -------------- | -------------------- | ---------------------- |
| TypeScript     | First-class support  | Optional               |
| Architecture   | Opinionated, modular | Minimal, unopinionated |
| DI             | Built-in             | Manual                 |
| Microservices  | First-class support  | External libs needed   |
| Learning curve | Higher               | Lower                  |

---

### 🔹 Prerequisites

* Knowledge of **JavaScript/TypeScript**
* Basics of **Node.js**
* Familiarity with **OOP (Classes, Interfaces)**

---

### 🔹 Installation & Hello World

**Install Nest CLI:**

```bash
npm i -g @nestjs/cli
```

**Create a new project:**

```bash
nest new project-name
```

**Run the server:**

```bash
npm run start:dev
```

Project structure after scaffold:

```
src/
  app.controller.ts
  app.service.ts
  app.module.ts
main.ts
```

---

### 🔹 Sample Hello World (Controller)

```ts
@Controller()
export class AppController {
  @Get()
  getHello(): string {
    return 'Hello World!';
  }
}
```

