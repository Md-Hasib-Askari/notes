

## 🟢 **3. Dependency Injection (DI)** in NestJS

---

### 🔹 **What is Dependency Injection?**

* **DI** is a design pattern where a class receives its dependencies (services, providers) from an external source instead of creating them.
* Promotes **loose coupling**, **reusability**, **testability**, and **scalability**.
* NestJS has a **built-in powerful DI system** using **TypeScript decorators**.

---

### 🔹 **How DI Works in NestJS**

* Classes are marked with `@Injectable()` to register them as **providers**.
* Providers can be injected into constructors of other classes.
* Nest uses **metadata reflection** (via `reflect-metadata`) to identify dependencies.

---

### ✅ **Step-by-Step DI Example**

#### 1. Create a Service (Provider)

```ts
import { Injectable } from '@nestjs/common';

@Injectable()
export class LoggerService {
  log(message: string) {
    console.log(`[Logger]: ${message}`);
  }
}
```

#### 2. Inject into Another Class (Controller)

```ts
import { Controller, Get } from '@nestjs/common';
import { LoggerService } from './logger.service';

@Controller('example')
export class ExampleController {
  constructor(private logger: LoggerService) {}

  @Get()
  logSomething() {
    this.logger.log('This is a log message');
    return 'Logged!';
  }
}
```

---

### 🔹 **Custom Providers (Manual Injection)**

You can define a custom provider with a token:

```ts
{
  provide: 'CONFIG',
  useValue: { debug: true }
}
```

Inject it like this:

```ts
constructor(@Inject('CONFIG') private config) {}
```

---

### 🔹 **useClass, useValue, useFactory, useExisting**

| Type          | Description                                 |
| ------------- | ------------------------------------------- |
| `useClass`    | Provide a class (default behavior)          |
| `useValue`    | Provide a constant value                    |
| `useFactory`  | Provide via a function (can use other deps) |
| `useExisting` | Alias an existing provider                  |

#### ✅ Example: useFactory

```ts
{
  provide: 'DATABASE_CONNECTION',
  useFactory: () => {
    return createConnection(); // some function
  },
}
```

---

### 🔹 **Scoped Providers**

* By default, all providers are **singleton-scoped**.
* You can scope them as **transient** or **request** scoped if needed.

```ts
@Injectable({ scope: Scope.REQUEST })
export class RequestScopedService {}
```

---

### 🧠 Summary

| Term            | Meaning                                        |
| --------------- | ---------------------------------------------- |
| `@Injectable()` | Marks a class as a provider                    |
| DI              | Injects dependencies via constructor           |
| `@Inject()`     | Manual injection using custom token            |
| Scope           | Controls provider lifetime (singleton/request) |

