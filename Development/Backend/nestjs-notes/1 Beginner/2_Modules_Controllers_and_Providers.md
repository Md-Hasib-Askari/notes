

## 🟢 **2. Modules, Controllers, and Providers**

---

### 🔹 **1. Modules**

#### ✅ What Are Modules?

* Fundamental building blocks in NestJS.
* Organize your app into cohesive blocks of functionality.
* Every NestJS app has at least one root module (`AppModule`).

#### ✅ How to Define a Module

```ts
import { Module } from '@nestjs/common';

@Module({
  imports: [],     // other modules
  controllers: [], // controller classes
  providers: [],   // services/providers
})
export class UserModule {}
```

#### ✅ Nest CLI Command

```bash
nest generate module user
```

#### ✅ Tip

* Use **feature modules** to organize by domain (e.g., AuthModule, ProductModule).

---

### 🔹 **2. Controllers**

#### ✅ What Are Controllers?

* Handle incoming HTTP requests and return responses.
* Route traffic using decorators like `@Get()`, `@Post()`, etc.

#### ✅ Example Controller

```ts
import { Controller, Get } from '@nestjs/common';

@Controller('user')
export class UserController {
  @Get()
  findAll() {
    return ['User1', 'User2'];
  }
}
```

#### ✅ Nest CLI Command

```bash
nest generate controller user
```

#### ✅ Route Mapping

* `@Controller('user')` + `@Get()` → `GET /user`
* `@Post()`, `@Put()`, `@Delete()`, etc. are available.

---

### 🔹 **3. Providers**

#### ✅ What Are Providers?

* Classes annotated with `@Injectable()`.
* Used for services, repositories, or any logic-heavy classes.
* Injected via **dependency injection** system.

#### ✅ Example Provider (Service)

```ts
import { Injectable } from '@nestjs/common';

@Injectable()
export class UserService {
  findAll() {
    return ['User1', 'User2'];
  }
}
```

#### ✅ Using the Provider in a Controller

```ts
@Controller('user')
export class UserController {
  constructor(private userService: UserService) {}

  @Get()
  findAll() {
    return this.userService.findAll();
  }
}
```

#### ✅ Nest CLI Command

```bash
nest generate service user
```

---

### 🧠 Summary

| Component  | Purpose                                 |
| ---------- | --------------------------------------- |
| Module     | Logical group of components             |
| Controller | Handles HTTP requests                   |
| Provider   | Business logic layer (usually services) |

