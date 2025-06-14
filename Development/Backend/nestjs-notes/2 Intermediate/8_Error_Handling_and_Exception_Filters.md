

## 🟡 **8. Error Handling & Exception Filters in NestJS**

Exception handling in NestJS is powerful and structured using **filters** that catch and process thrown exceptions.

---

### ✅ **1. Default Exception Handling**

NestJS uses `HttpException` by default.

#### Example:

```ts
throw new HttpException('Forbidden', HttpStatus.FORBIDDEN);
```

Or:

```ts
throw new ForbiddenException('Access denied');
```

---

### ✅ **2. Built-in Exceptions**

Nest provides several ready-to-use HTTP exceptions:

| Exception                      | Status Code |
| ------------------------------ | ----------- |
| `BadRequestException`          | 400         |
| `UnauthorizedException`        | 401         |
| `ForbiddenException`           | 403         |
| `NotFoundException`            | 404         |
| `ConflictException`            | 409         |
| `InternalServerErrorException` | 500         |

---

### ✅ **3. Creating a Custom Exception Filter**

Filters let you handle exceptions and modify the response globally or locally.

#### 🧱 Example:

```ts
@Catch(HttpException)
export class AllExceptionsFilter implements ExceptionFilter {
  catch(exception: HttpException, host: ArgumentsHost) {
    const ctx = host.switchToHttp();
    const response = ctx.getResponse<Response>();
    const status = exception.getStatus();
    const message = exception.message;

    response.status(status).json({
      statusCode: status,
      message,
      timestamp: new Date().toISOString(),
    });
  }
}
```

---

### ✅ **4. Applying Filters**

* **Method level:**

```ts
@UseFilters(AllExceptionsFilter)
@Get()
findAll() {
  throw new NotFoundException('Items not found');
}
```

* **Controller level:**

```ts
@UseFilters(AllExceptionsFilter)
@Controller('items')
export class ItemsController {}
```

* **Global level (main.ts):**

```ts
const app = await NestFactory.create(AppModule);
app.useGlobalFilters(new AllExceptionsFilter());
```

---

### ✅ **5. Custom Exception Class**

Define custom error behavior.

#### 🧱 Example:

```ts
export class ResourceNotFoundException extends HttpException {
  constructor(resource: string) {
    super(`${resource} not found`, HttpStatus.NOT_FOUND);
  }
}
```

Use:

```ts
throw new ResourceNotFoundException('User');
```

---

### ✅ **6. Filter with Non-HTTP Exceptions**

You can catch any error, not just `HttpException`:

```ts
@Catch()
export class GenericExceptionFilter implements ExceptionFilter {
  catch(exception: unknown, host: ArgumentsHost) {
    const ctx = host.switchToHttp();
    const response = ctx.getResponse<Response>();
    console.error('Unhandled exception:', exception);

    response.status(500).json({
      message: 'Internal server error',
      error: exception instanceof Error ? exception.message : 'Unknown error',
    });
  }
}
```

---

### 💪 Exercise

✅ **Create a custom exception for "Resource Not Found"**

1. Define a `ResourceNotFoundException` class.
2. Use it in a route that checks a DB record.
3. Create an `ExceptionFilter` that formats all HTTP exceptions with timestamps and paths.

