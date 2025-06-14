
## 🟡 **6. Middleware, Pipes, Guards, and Interceptors in NestJS**

These are core tools in NestJS for controlling the **request-response lifecycle**.

---

### ✅ **1. Middleware – Pre-route Logic**

Middleware is executed **before** the route handler.

#### 🔧 Use cases:

* Logging
* Request transformation
* Authentication checks

#### 🧱 Example:

```ts
@Injectable()
export class LoggerMiddleware implements NestMiddleware {
  use(req: Request, res: Response, next: NextFunction) {
    console.log(`${req.method} ${req.url}`);
    next();
  }
}
```

Apply in `AppModule`:

```ts
export class AppModule implements NestModule {
  configure(consumer: MiddlewareConsumer) {
    consumer
      .apply(LoggerMiddleware)
      .forRoutes('*'); // or specific routes
  }
}
```

---

### ✅ **2. Pipes – Input Validation & Transformation**

Pipes work on method parameters, often used with DTOs.

#### Built-in Pipes:

* `ValidationPipe`
* `ParseIntPipe`, `ParseBoolPipe`
* `DefaultValuePipe`

#### 🧱 Example:

```ts
@UsePipes(new ValidationPipe())
@Post()
create(@Body() createTodoDto: CreateTodoDto) {
  return this.todoService.create(createTodoDto.title);
}
```

#### With a param:

```ts
@Get(':id')
findOne(@Param('id', ParseIntPipe) id: number) {
  return this.todoService.findOne(id);
}
```

---

### ✅ **3. Guards – Auth & Role-based Access**

Guards determine **if a request can proceed**.

#### Built-in: `AuthGuard`, `RolesGuard`

#### 🧱 Example: Role-based Guard

```ts
@Injectable()
export class RolesGuard implements CanActivate {
  canActivate(context: ExecutionContext): boolean {
    const request = context.switchToHttp().getRequest();
    const user = request.user;
    return user?.roles?.includes('admin');
  }
}
```

Apply guard:

```ts
@UseGuards(RolesGuard)
@Get()
findAll() {
  return this.todoService.findAll();
}
```

---

### ✅ **4. Interceptors – Response Manipulation, Logging, Caching**

Executed **before and after** route handlers.

#### 🔧 Use cases:

* Logging
* Modifying response
* Caching
* Exception mapping

#### 🧱 Example: Logging Interceptor

```ts
@Injectable()
export class LoggingInterceptor implements NestInterceptor {
  intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
    console.log('Before route handler...');
    const now = Date.now();
    return next
      .handle()
      .pipe(tap(() => console.log(`After... ${Date.now() - now}ms`)));
  }
}
```

Apply:

```ts
@UseInterceptors(LoggingInterceptor)
@Get()
findAll() {
  return this.todoService.findAll();
}
```

---

### 🔁 Summary

| Feature     | Role in Lifecycle           | Use Case                         |
| ----------- | --------------------------- | -------------------------------- |
| Middleware  | Before controller           | Logging, basic auth              |
| Pipe        | On controller method params | Validation, parsing              |
| Guard       | Before route execution      | Auth, RBAC                       |
| Interceptor | Before/after controller     | Logging, caching, response edits |

---

### 💪 Exercise

✅ **Implement a request logger and a role-based guard:**

1. Create `LoggerMiddleware` to log method and path.
2. Create `RolesGuard` that allows access only to `admin`.
3. Apply both in a module.

