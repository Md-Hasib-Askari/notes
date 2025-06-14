

## 🔵 **10. Modular Architecture and Microservices**

NestJS shines in large-scale projects due to its modular design and microservice capabilities.

---

### ✅ 1. **Dynamic Modules**

Modules in NestJS can be dynamically configured at runtime.

#### Example:

```ts
@Module({})
export class DatabaseModule {
  static register(options: DatabaseOptions): DynamicModule {
    return {
      module: DatabaseModule,
      providers: [{ provide: 'DB_OPTIONS', useValue: options }],
      exports: ['DB_OPTIONS'],
    };
  }
}
```

Use:

```ts
DatabaseModule.register({ host: 'localhost' })
```

---

### ✅ 2. **Shared Modules**

These modules export services that are used in multiple other modules. Be cautious with `providers` scope to avoid singletons.

```ts
@Module({
  providers: [CommonService],
  exports: [CommonService],
})
export class SharedModule {}
```

---

### ✅ 3. **Feature Modules**

These encapsulate features of the app (e.g., AuthModule, UserModule), promoting clean architecture and separation of concerns.

```ts
@Module({
  controllers: [UsersController],
  providers: [UsersService],
})
export class UsersModule {}
```

---

### ✅ 4. **Microservices Architecture in NestJS**

NestJS supports microservices via:

* **TCP** (default protocol)
* **Redis**
* **NATS**
* **gRPC**
* **MQs** like RabbitMQ, Kafka

#### Basic Microservice:

```ts
const app = await NestFactory.createMicroservice(AppModule, {
  transport: Transport.TCP,
  options: { port: 3001 },
});
await app.listen();
```

#### Send from main app:

```ts
this.client.send('ping', {});
```

---

### ✅ 5. **Message Patterns**

Controller in microservice listens to specific messages.

```ts
@Controller()
export class NotificationController {
  @MessagePattern('send_notification')
  handleNotification(data: any) {
    console.log('Notification received:', data);
  }
}
```

---

### 💪 Exercise

✅ **Modularize a large project and create one microservice**

1. Refactor your app into feature modules.
2. Create a microservice (e.g., `NotificationService`) using TCP or Redis transport.
3. Send messages from your main app to the microservice.

