

## 🟢 **4. Routing and Request Handling in NestJS**

---

### 🔹 **What is Routing?**

* Routing maps **incoming HTTP requests** to **controller methods**.
* NestJS uses **decorators** like `@Get()`, `@Post()`, etc., to define routes.

---

### ✅ **1. Defining Routes with Decorators**

| Decorator    | HTTP Method |
| ------------ | ----------- |
| `@Get()`     | GET         |
| `@Post()`    | POST        |
| `@Put()`     | PUT         |
| `@Delete()`  | DELETE      |
| `@Patch()`   | PATCH       |
| `@Options()` | OPTIONS     |
| `@Head()`    | HEAD        |
| `@All()`     | ALL Methods |

#### Example:

```ts
@Controller('users')
export class UsersController {
  @Get()
  getAllUsers() {
    return 'All users';
  }

  @Post()
  createUser() {
    return 'User created';
  }
}
```

This sets up:

* `GET /users` → `getAllUsers()`
* `POST /users` → `createUser()`

---

### ✅ **2. Route Parameters**

#### URL Params: `@Param()`

```ts
@Get(':id')
getUser(@Param('id') id: string) {
  return `User ID: ${id}`;
}
```

URL: `GET /users/42` → `id = '42'`

#### Query Params: `@Query()`

```ts
@Get()
find(@Query('role') role: string) {
  return `Role: ${role}`;
}
```

URL: `GET /users?role=admin` → `role = 'admin'`

#### Request Body: `@Body()`

```ts
@Post()
create(@Body() userData: any) {
  return userData;
}
```

#### Headers: `@Headers()`

```ts
@Get()
getHeader(@Headers('user-agent') ua: string) {
  return ua;
}
```

#### Request Object: `@Req()`

```ts
@Get()
getRequest(@Req() req: Request) {
  return req.url;
}
```

---

### ✅ **3. NestJS Route Prefixing**

Global route prefix:

```ts
// main.ts
app.setGlobalPrefix('api');
```

Now, `/users` becomes `/api/users`.

---

### ✅ **4. Wildcard and Optional Routes**

#### Wildcard:

```ts
@Get('*')
handleWildcard() {
  return 'Catch-all';
}
```

#### Optional Params:

```ts
@Get(':id?')
optionalId(@Param('id') id?: string) {
  return id || 'No ID provided';
}
```

---

### ✅ **5. Request Lifecycle Flow**

```
Client Request
   ↓
Router → Controller → Service
   ↓
  Response
```

---

### 🧠 Summary

| Feature         | Usage                               |
| --------------- | ----------------------------------- |
| `@Controller()` | Group route handlers under one path |
| `@Get()`, etc.  | Define HTTP method routes           |
| `@Param()`      | Extract route parameters            |
| `@Query()`      | Extract query strings               |
| `@Body()`       | Extract request body                |
| `@Headers()`    | Extract headers                     |
| `@Req()`        | Access full request object          |

