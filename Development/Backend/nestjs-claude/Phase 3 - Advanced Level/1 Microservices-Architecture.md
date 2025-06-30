# NestJS Phase 3: Microservices Architecture

## 1. NestJS Microservices Patterns

### Basic Microservice Setup
```bash
# Install microservices package
npm install @nestjs/microservices

# For message brokers
npm install redis
npm install amqplib
npm install kafkajs
```

### Creating a Microservice
```typescript
// main.ts for microservice
import { NestFactory } from '@nestjs/core';
import { MicroserviceOptions, Transport } from '@nestjs/microservices';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.createMicroservice<MicroserviceOptions>(AppModule, {
    transport: Transport.TCP,
    options: {
      host: '0.0.0.0',
      port: 3001,
    },
  });

  await app.listen();
  console.log('Microservice is listening on port 3001');
}
bootstrap();
```

### Hybrid Application (HTTP + Microservice)
```typescript
// main.ts for hybrid app
import { NestFactory } from '@nestjs/core';
import { MicroserviceOptions, Transport } from '@nestjs/microservices';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  
  // Add microservice
  app.connectMicroservice<MicroserviceOptions>({
    transport: Transport.TCP,
    options: {
      port: 3001,
    },
  });

  await app.startAllMicroservices();
  await app.listen(3000);
  
  console.log('HTTP server running on port 3000');
  console.log('Microservice running on port 3001');
}
bootstrap();
```

### Microservice Controller
```typescript
// users/users.controller.ts
import { Controller } from '@nestjs/common';
import { MessagePattern, Payload } from '@nestjs/microservices';
import { UsersService } from './users.service';
import { CreateUserDto, UpdateUserDto } from './dto';

@Controller()
export class UsersController {
  constructor(private readonly usersService: UsersService) {}

  @MessagePattern('users.create')
  async createUser(@Payload() createUserDto: CreateUserDto) {
    return this.usersService.create(createUserDto);
  }

  @MessagePattern('users.findAll')
  async findAllUsers(@Payload() filters: any) {
    return this.usersService.findAll(filters);
  }

  @MessagePattern('users.findOne')
  async findOneUser(@Payload() id: number) {
    return this.usersService.findOne(id);
  }

  @MessagePattern('users.update')
  async updateUser(@Payload() data: { id: number; updateUserDto: UpdateUserDto }) {
    return this.usersService.update(data.id, data.updateUserDto);
  }

  @MessagePattern('users.remove')
  async removeUser(@Payload() id: number) {
    return this.usersService.remove(id);
  }

  @MessagePattern('users.validate')
  async validateUser(@Payload() credentials: { email: string; password: string }) {
    return this.usersService.validateCredentials(credentials.email, credentials.password);
  }
}
```

### Client Proxy for Inter-Service Communication
```typescript
// gateway/gateway.module.ts
import { Module } from '@nestjs/common';
import { ClientsModule, Transport } from '@nestjs/microservices';
import { GatewayController } from './gateway.controller';

@Module({
  imports: [
    ClientsModule.register([
      {
        name: 'USERS_SERVICE',
        transport: Transport.TCP,
        options: {
          host: 'localhost',
          port: 3001,
        },
      },
      {
        name: 'ORDERS_SERVICE',
        transport: Transport.TCP,
        options: {
          host: 'localhost',
          port: 3002,
        },
      },
    ]),
  ],
  controllers: [GatewayController],
})
export class GatewayModule {}
```

### API Gateway Implementation
```typescript
// gateway/gateway.controller.ts
import { Controller, Get, Post, Put, Delete, Body, Param, Inject } from '@nestjs/common';
import { ClientProxy } from '@nestjs/microservices';
import { Observable, firstValueFrom } from 'rxjs';
import { CreateUserDto, UpdateUserDto } from './dto';

@Controller('api')
export class GatewayController {
  constructor(
    @Inject('USERS_SERVICE') private readonly usersClient: ClientProxy,
    @Inject('ORDERS_SERVICE') private readonly ordersClient: ClientProxy,
  ) {}

  // Users endpoints
  @Get('users')
  getUsers(): Observable<any> {
    return this.usersClient.send('users.findAll', {});
  }

  @Get('users/:id')
  getUser(@Param('id') id: string): Observable<any> {
    return this.usersClient.send('users.findOne', parseInt(id));
  }

  @Post('users')
  createUser(@Body() createUserDto: CreateUserDto): Observable<any> {
    return this.usersClient.send('users.create', createUserDto);
  }

  @Put('users/:id')
  updateUser(
    @Param('id') id: string,
    @Body() updateUserDto: UpdateUserDto,
  ): Observable<any> {
    return this.usersClient.send('users.update', {
      id: parseInt(id),
      updateUserDto,
    });
  }

  @Delete('users/:id')
  deleteUser(@Param('id') id: string): Observable<any> {
    return this.usersClient.send('users.remove', parseInt(id));
  }

  // Complex operation involving multiple services
  @Get('users/:id/orders')
  async getUserOrders(@Param('id') id: string) {
    const user = await firstValueFrom(
      this.usersClient.send('users.findOne', parseInt(id))
    );
    
    if (!user) {
      throw new Error('User not found');
    }

    const orders = await firstValueFrom(
      this.ordersClient.send('orders.findByUserId', parseInt(id))
    );

    return {
      user,
      orders,
    };
  }
}
```

## 2. Message Brokers (Redis, RabbitMQ, Kafka)

### Redis Transport
```typescript
// Redis microservice setup
import { NestFactory } from '@nestjs/core';
import { MicroserviceOptions, Transport } from '@nestjs/microservices';

async function bootstrap() {
  const app = await NestFactory.createMicroservice<MicroserviceOptions>(AppModule, {
    transport: Transport.REDIS,
    options: {
      host: 'localhost',
      port: 6379,
      retryAttempts: 5,
      retryDelay: 3000,
    },
  });

  await app.listen();
}
bootstrap();

// Redis client configuration
ClientsModule.register([
  {
    name: 'REDIS_SERVICE',
    transport: Transport.REDIS,
    options: {
      host: 'localhost',
      port: 6379,
    },
  },
]);
```

### RabbitMQ Transport
```bash
npm install amqplib
npm install -D @types/amqplib
```

```typescript
// RabbitMQ microservice setup
async function bootstrap() {
  const app = await NestFactory.createMicroservice<MicroserviceOptions>(AppModule, {
    transport: Transport.RMQ,
    options: {
      urls: ['amqp://localhost:5672'],
      queue: 'users_queue',
      queueOptions: {
        durable: false,
      },
    },
  });

  await app.listen();
}

// RabbitMQ client configuration
ClientsModule.register([
  {
    name: 'RABBITMQ_SERVICE',
    transport: Transport.RMQ,
    options: {
      urls: ['amqp://localhost:5672'],
      queue: 'orders_queue',
      queueOptions: {
        durable: false,
      },
    },
  },
]);
```

### Kafka Transport
```bash
npm install kafkajs
```

```typescript
// Kafka microservice setup
async function bootstrap() {
  const app = await NestFactory.createMicroservice<MicroserviceOptions>(AppModule, {
    transport: Transport.KAFKA,
    options: {
      client: {
        clientId: 'users-service',
        brokers: ['localhost:9092'],
      },
      consumer: {
        groupId: 'users-consumer',
      },
    },
  });

  await app.listen();
}

// Kafka client configuration
ClientsModule.register([
  {
    name: 'KAFKA_SERVICE',
    transport: Transport.KAFKA,
    options: {
      client: {
        clientId: 'gateway',
        brokers: ['localhost:9092'],
      },
      consumer: {
        groupId: 'gateway-consumer',
      },
    },
  },
]);

// Kafka controller with topics
@Controller()
export class OrdersController {
  @EventPattern('order.created')
  handleOrderCreated(@Payload() data: any) {
    console.log('Order created:', data);
  }

  @EventPattern('order.updated')
  handleOrderUpdated(@Payload() data: any) {
    console.log('Order updated:', data);
  }
}
```

### Event-Driven Architecture with RabbitMQ
```typescript
// events/order.events.ts
export class OrderCreatedEvent {
  constructor(
    public readonly orderId: string,
    public readonly userId: string,
    public readonly total: number,
    public readonly items: any[],
  ) {}
}

// orders/orders.service.ts
@Injectable()
export class OrdersService {
  constructor(
    @Inject('EVENTS_SERVICE') private readonly eventsClient: ClientProxy,
  ) {}

  async createOrder(createOrderDto: CreateOrderDto) {
    const order = await this.ordersRepository.save(createOrderDto);
    
    // Emit event
    this.eventsClient.emit('order.created', new OrderCreatedEvent(
      order.id,
      order.userId,
      order.total,
      order.items,
    ));

    return order;
  }
}

// notifications/notifications.controller.ts
@Controller()
export class NotificationsController {
  @EventPattern('order.created')
  async handleOrderCreated(@Payload() event: OrderCreatedEvent) {
    await this.emailService.sendOrderConfirmation(
      event.userId,
      event.orderId,
      event.total,
    );
  }

  @EventPattern('user.registered')
  async handleUserRegistered(@Payload() event: any) {
    await this.emailService.sendWelcomeEmail(event.email, event.name);
  }
}
```

## 3. Inter-Service Communication

### Request-Response Pattern
```typescript
// Synchronous communication
@Injectable()
export class OrdersService {
  constructor(
    @Inject('USERS_SERVICE') private readonly usersClient: ClientProxy,
    @Inject('INVENTORY_SERVICE') private readonly inventoryClient: ClientProxy,
  ) {}

  async createOrder(createOrderDto: CreateOrderDto) {
    // Validate user exists
    const user = await firstValueFrom(
      this.usersClient.send('users.findOne', createOrderDto.userId)
    );

    if (!user) {
      throw new BadRequestException('User not found');
    }

    // Check inventory
    const inventoryCheck = await firstValueFrom(
      this.inventoryClient.send('inventory.check', createOrderDto.items)
    );

    if (!inventoryCheck.available) {
      throw new BadRequestException('Insufficient inventory');
    }

    // Create order
    const order = await this.ordersRepository.save(createOrderDto);

    // Update inventory
    await firstValueFrom(
      this.inventoryClient.send('inventory.reserve', {
        orderId: order.id,
        items: createOrderDto.items,
      })
    );

    return order;
  }
}
```

### Publish-Subscribe Pattern
```typescript
// Event publisher
@Injectable()
export class PaymentService {
  constructor(
    @Inject('EVENTS_SERVICE') private readonly eventsClient: ClientProxy,
  ) {}

  async processPayment(paymentData: any) {
    try {
      const payment = await this.paymentProvider.charge(paymentData);
      
      // Publish success event
      this.eventsClient.emit('payment.success', {
        paymentId: payment.id,
        orderId: paymentData.orderId,
        amount: payment.amount,
      });

      return payment;
    } catch (error) {
      // Publish failure event
      this.eventsClient.emit('payment.failed', {
        orderId: paymentData.orderId,
        error: error.message,
      });
      
      throw error;
    }
  }
}

// Event subscribers
@Controller()
export class OrdersEventController {
  @EventPattern('payment.success')
  async handlePaymentSuccess(@Payload() data: any) {
    await this.ordersService.markAsPaid(data.orderId);
    await this.fulfillmentService.scheduleShipping(data.orderId);
  }

  @EventPattern('payment.failed')
  async handlePaymentFailed(@Payload() data: any) {
    await this.ordersService.markAsPaymentFailed(data.orderId);
    await this.inventoryService.releaseReservation(data.orderId);
  }
}
```

### Circuit Breaker Pattern
```typescript
// circuit-breaker.decorator.ts
import { createDecorator } from '@nestjs/common';

export const CircuitBreaker = (options?: {
  threshold?: number;
  timeout?: number;
  resetTimeout?: number;
}) => createDecorator('circuit-breaker', options);

// circuit-breaker.service.ts
@Injectable()
export class CircuitBreakerService {
  private circuits = new Map();

  async execute<T>(
    key: string,
    operation: () => Promise<T>,
    options: any = {},
  ): Promise<T> {
    const circuit = this.getCircuit(key, options);
    
    if (circuit.isOpen()) {
      throw new ServiceUnavailableException('Circuit breaker is open');
    }

    try {
      const result = await operation();
      circuit.recordSuccess();
      return result;
    } catch (error) {
      circuit.recordFailure();
      throw error;
    }
  }

  private getCircuit(key: string, options: any) {
    if (!this.circuits.has(key)) {
      this.circuits.set(key, new Circuit(options));
    }
    return this.circuits.get(key);
  }
}

class Circuit {
  private failures = 0;
  private lastFailureTime = 0;

  constructor(private options: any) {}

  isOpen(): boolean {
    if (this.failures >= (this.options.threshold || 5)) {
      const now = Date.now();
      const resetTimeout = this.options.resetTimeout || 60000;
      
      if (now - this.lastFailureTime > resetTimeout) {
        this.reset();
        return false;
      }
      return true;
    }
    return false;
  }

  recordSuccess(): void {
    this.reset();
  }

  recordFailure(): void {
    this.failures++;
    this.lastFailureTime = Date.now();
  }

  private reset(): void {
    this.failures = 0;
    this.lastFailureTime = 0;
  }
}
```

## 4. Service Discovery and Load Balancing

### Service Registry
```typescript
// service-registry.service.ts
@Injectable()
export class ServiceRegistryService {
  private services = new Map<string, ServiceInstance[]>();

  registerService(service: ServiceInstance): void {
    const serviceName = service.name;
    
    if (!this.services.has(serviceName)) {
      this.services.set(serviceName, []);
    }
    
    const instances = this.services.get(serviceName);
    const existingIndex = instances.findIndex(
      (instance) => instance.id === service.id
    );
    
    if (existingIndex >= 0) {
      instances[existingIndex] = service;
    } else {
      instances.push(service);
    }
    
    this.services.set(serviceName, instances);
  }

  deregisterService(serviceName: string, serviceId: string): void {
    const instances = this.services.get(serviceName) || [];
    const filteredInstances = instances.filter(
      (instance) => instance.id !== serviceId
    );
    this.services.set(serviceName, filteredInstances);
  }

  getService(serviceName: string): ServiceInstance | null {
    const instances = this.services.get(serviceName) || [];
    const healthyInstances = instances.filter(
      (instance) => instance.health === 'healthy'
    );
    
    if (healthyInstances.length === 0) {
      return null;
    }
    
    // Round-robin load balancing
    const index = Math.floor(Math.random() * healthyInstances.length);
    return healthyInstances[index];
  }

  getAllServices(): Map<string, ServiceInstance[]> {
    return this.services;
  }
}

interface ServiceInstance {
  id: string;
  name: string;
  host: string;
  port: number;
  health: 'healthy' | 'unhealthy';
  metadata?: Record<string, any>;
  lastHeartbeat?: Date;
}
```

### Health Checks
```typescript
// health/health.controller.ts
import { Controller, Get } from '@nestjs/common';
import { HealthCheck, HealthCheckService, TypeOrmHealthIndicator } from '@nestjs/terminus';

@Controller('health')
export class HealthController {
  constructor(
    private health: HealthCheckService,
    private db: TypeOrmHealthIndicator,
  ) {}

  @Get()
  @HealthCheck()
  check() {
    return this.health.check([
      () => this.db.pingCheck('database'),
      () => this.checkExternalServices(),
    ]);
  }

  private async checkExternalServices() {
    // Check dependencies
    const checks = await Promise.allSettled([
      this.pingService('users-service', 'http://users-service:3001/health'),
      this.pingService('orders-service', 'http://orders-service:3002/health'),
    ]);
    
    return {
      external_services: {
        status: checks.every((check) => check.status === 'fulfilled') ? 'up' : 'down',
        checks: checks.map((check, index) => ({
          service: ['users-service', 'orders-service'][index],
          status: check.status,
        })),
      },
    };
  }

  private async pingService(name: string, url: string) {
    // Implementation for pinging service
    return { [name]: 'up' };
  }
}
```

### Load Balancer Implementation
```typescript
// load-balancer.service.ts
@Injectable()
export class LoadBalancerService {
  private roundRobinCounters = new Map<string, number>();

  constructor(private serviceRegistry: ServiceRegistryService) {}

  getServiceInstance(serviceName: string, strategy: 'round-robin' | 'random' | 'least-connections' = 'round-robin'): ServiceInstance | null {
    const instances = this.serviceRegistry.getHealthyInstances(serviceName);
    
    if (instances.length === 0) {
      return null;
    }

    switch (strategy) {
      case 'round-robin':
        return this.roundRobin(serviceName, instances);
      case 'random':
        return this.random(instances);
      case 'least-connections':
        return this.leastConnections(instances);
      default:
        return this.roundRobin(serviceName, instances);
    }
  }

  private roundRobin(serviceName: string, instances: ServiceInstance[]): ServiceInstance {
    const counter = this.roundRobinCounters.get(serviceName) || 0;
    const instance = instances[counter % instances.length];
    this.roundRobinCounters.set(serviceName, counter + 1);
    return instance;
  }

  private random(instances: ServiceInstance[]): ServiceInstance {
    const index = Math.floor(Math.random() * instances.length);
    return instances[index];
  }

  private leastConnections(instances: ServiceInstance[]): ServiceInstance {
    return instances.reduce((least, current) => 
      (current.connections || 0) < (least.connections || 0) ? current : least
    );
  }
}
```

## 5. Distributed Tracing

### OpenTelemetry Integration
```bash
npm install @opentelemetry/api @opentelemetry/sdk-node
npm install @opentelemetry/auto-instrumentations-node
npm install @opentelemetry/exporter-jaeger
```

```typescript
// tracing.ts
import { NodeSDK } from '@opentelemetry/sdk-node';
import { getNodeAutoInstrumentations } from '@opentelemetry/auto-instrumentations-node';
import { JaegerExporter } from '@opentelemetry/exporter-jaeger';

const jaegerExporter = new JaegerExporter({
  endpoint: 'http://localhost:14268/api/traces',
});

const sdk = new NodeSDK({
  traceExporter: jaegerExporter,
  instrumentations: [getNodeAutoInstrumentations()],
});

sdk.start();
```

### Custom Tracing Service
```typescript
// tracing/tracing.service.ts
import { Injectable } from '@nestjs/common';
import { trace, context, SpanKind } from '@opentelemetry/api';

@Injectable()
export class TracingService {
  private tracer = trace.getTracer('nestjs-microservice');

  async traceAsyncOperation<T>(
    name: string,
    operation: () => Promise<T>,
    attributes?: Record<string, string>,
  ): Promise<T> {
    const span = this.tracer.startSpan(name, {
      kind: SpanKind.INTERNAL,
      attributes,
    });

    try {
      const result = await context.with(trace.setSpan(context.active(), span), operation);
      span.setStatus({ code: trace.SpanStatusCode.OK });
      return result;
    } catch (error) {
      span.setStatus({
        code: trace.SpanStatusCode.ERROR,
        message: error.message,
      });
      span.recordException(error);
      throw error;
    } finally {
      span.end();
    }
  }

  createChildSpan(name: string, attributes?: Record<string, string>) {
    return this.tracer.startSpan(name, {
      kind: SpanKind.INTERNAL,
      attributes,
    });
  }
}
```

### Tracing Interceptor
```typescript
// tracing/tracing.interceptor.ts
@Injectable()
export class TracingInterceptor implements NestInterceptor {
  constructor(private tracingService: TracingService) {}

  intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
    const request = context.switchToHttp().getRequest();
    const operationName = `${context.getClass().name}.${context.getHandler().name}`;
    
    return from(
      this.tracingService.traceAsyncOperation(
        operationName,
        () => next.handle().pipe(
          tap((data) => {
            // Log successful operation
          }),
          catchError((error) => {
            // Log error
            throw error;
          }),
        ).toPromise(),
        {
          'http.method': request.method,
          'http.url': request.url,
          'user.id': request.user?.id,
        },
      ),
    );
  }
}
```

### Distributed Context Propagation
```typescript
// context/context.service.ts
@Injectable()
export class ContextService {
  private context = new Map<string, any>();

  set(key: string, value: any): void {
    this.context.set(key, value);
  }

  get<T>(key: string): T | undefined {
    return this.context.get(key);
  }

  clear(): void {
    this.context.clear();
  }

  getCorrelationId(): string {
    return this.get('correlationId') || this.generateCorrelationId();
  }

  private generateCorrelationId(): string {
    const correlationId = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    this.set('correlationId', correlationId);
    return correlationId;
  }
}

// context/context.middleware.ts
@Injectable()
export class ContextMiddleware implements NestMiddleware {
  constructor(private contextService: ContextService) {}

  use(req: Request, res: Response, next: NextFunction) {
    const correlationId = req.headers['x-correlation-id'] as string || 
                         this.contextService.generateCorrelationId();
    
    this.contextService.set('correlationId', correlationId);
    this.contextService.set('userId', req.user?.id);
    
    res.setHeader('x-correlation-id', correlationId);
    
    next();
  }
}
```

### Docker Compose for Microservices
```yaml
# docker-compose.yml
version: '3.8'

services:
  gateway:
    build: ./gateway
    ports:
      - "3000:3000"
    environment:
      - USERS_SERVICE_URL=users-service:3001
      - ORDERS_SERVICE_URL=orders-service:3002
    depends_on:
      - users-service
      - orders-service
      - redis

  users-service:
    build: ./users-service
    ports:
      - "3001:3001"
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/users_db
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis

  orders-service:
    build: ./orders-service
    ports:
      - "3002:3002"
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/orders_db
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:13
    environment:
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
      - "14268:14268"

volumes:
  postgres_data:
```

This comprehensive guide covers all essential microservices architecture topics for Phase 3, providing practical implementations and patterns for building scalable distributed systems with NestJS.
