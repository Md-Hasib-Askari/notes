

## ✅ 1. **CRM Backend with Multi-Role Permissions**

### 🏗 Key Features

* User roles: Admin, Manager, Sales Rep, Support
* Role-based access control (RBAC)
* Customer and Lead management
* Notes, Tasks, Reminders per lead
* API for integration with WhatsApp or Email

### 🧰 Technologies

* **NestJS** (RBAC via `@Roles()` guard + custom decorators)
* **TypeORM/Prisma** for persistence
* **JWT + Passport** for auth
* **Redis** (optional) for caching lead search
* **Swagger** for API documentation

### 📂 Structure

```
/src
  /auth
  /users
  /roles
  /customers
  /leads
  /tasks
```

### 🔐 Sample Route Guard

```ts
@UseGuards(AuthGuard('jwt'), RolesGuard)
@Roles('admin', 'sales')
@Post('/leads')
createLead() {}
```

---

## ✅ 2. **Real-Time Stock Tracking App**

### 🏗 Key Features

* Live stock price updates via WebSocket
* RESTful APIs for search, favorites, and profile
* JWT-based authentication
* Push notifications for price alerts
* Chart data history

### 🧰 Technologies

* **NestJS** + `@WebSocketGateway()`
* **WebSocket server** + event emitters
* **Redis** (Pub/Sub or caching)
* **Rate limiter** for WebSocket
* **JWT + Guards** for secure socket channels

### 📂 Structure

```
/src
  /auth
  /users
  /stocks
  /alerts
  /ws
```

### 💬 WebSocket Event Example

```ts
@WebSocketGateway()
export class StockGateway {
  @SubscribeMessage('track_stock')
  handleTrack(client, { symbol }) {
    // Subscribe client to real-time stock updates
  }
}
```

---

## ✅ 3. **Microservice-Based E-commerce Backend**

### 🏗 Microservices

* **Auth Service** (JWT, refresh token)
* **Product Service** (catalog, inventory)
* **Order Service** (cart, order, checkout)
* **Payment Service** (mock payment, status updates)
* Optional: **Notification Service** (email/SMS)

### 🧰 Technologies

* **NestJS** Microservices (`@nestjs/microservices`)
* **Redis** or **RabbitMQ** for message broker
* **PostgreSQL** or **MongoDB** per service
* **Swagger/OpenAPI** docs for each service
* **API Gateway** (optional with `@nestjs/apigateway`)
* Use **Nx Monorepo** for organized structure

### 🔄 Communication Example (Redis Transport)

```ts
ClientsModule.register([
  {
    name: 'ORDER_SERVICE',
    transport: Transport.REDIS,
    options: { url: 'redis://localhost:6379' },
  },
]);
```

### 📦 Project Layout (Nx Monorepo)

```
/apps
  /api-gateway
  /auth-service
  /product-service
  /order-service
  /payment-service
/libs
  /shared
  /dto
```

---

### 🛠 Bonus Tips for All Projects

* Add **unit tests & E2E tests** using Jest and Supertest
* Implement **Swagger** for REST API docs
* Add **rate-limiting** and **CORS** guards
* Use **Docker Compose** for multi-service dev setup
* Integrate **GitHub Actions** or **GitLab CI** for CI/CD

