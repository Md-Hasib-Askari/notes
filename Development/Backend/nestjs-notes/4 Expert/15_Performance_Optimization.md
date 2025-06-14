

## 🟣 **15. Performance Optimization**

Performance tuning is critical for scalable production apps. NestJS provides tools to optimize request latency, memory usage, and responsiveness.

---

### ✅ 1. Caching Strategies

**Why?** To reduce DB load and speed up frequent reads.

* **Built-in CacheModule**

  ```ts
  import { CacheModule } from '@nestjs/common';

  @Module({
    imports: [CacheModule.register()],
    controllers: [YourController],
    providers: [YourService],
  })
  export class YourModule {}
  ```

* **Caching in Controller**

  ```ts
  @UseInterceptors(CacheInterceptor)
  @Get('popular-products')
  async getPopularProducts() {
    return this.productService.getPopular();
  }
  ```

* **Use Redis for distributed caching**

  ```bash
  npm install cache-manager-ioredis ioredis
  ```

  ```ts
  CacheModule.register({
    store: redisStore,
    host: 'localhost',
    port: 6379,
  });
  ```

---

### ✅ 2. Lazy Loading Modules

Load heavy modules only when needed using dynamic imports.

```ts
@Module({
  imports: [
    forwardRef(() => import('./heavy/heavy.module').then(m => m.HeavyModule)),
  ],
})
export class AppModule {}
```

Avoid bloated root modules by organizing features into smaller, focused modules.

---

### ✅ 3. Avoid Memory Leaks

Common causes:

* Long-lived subscriptions (e.g., Observables, EventEmitters)
* Not closing DB connections or timeouts
* Global state misuse

**Tips:**

* Use `onModuleDestroy()` or `onApplicationShutdown()` lifecycle hooks to clean up.
* Use profiling tools (`node --inspect`, Chrome DevTools, `clinic.js`) to check memory.

---

### ✅ 4. Avoiding Slow Dependencies

Audit 3rd-party packages:

* Avoid unmaintained or bloated ones.
* Check for async-heavy operations in hot paths.
* Prefer native or compiled libraries when possible (e.g., `bcrypt` vs `bcryptjs`).

---

### 🧪 Exercise

✅ **Implement caching in a high-read route**
Example:

* A product search endpoint
* Add caching with Redis for 30 seconds
* Benchmark performance with and without caching using tools like Postman or `autocannon`

