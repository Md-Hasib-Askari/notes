# Advanced Project: Microservices-based Application with Event Sourcing and CQRS

## Project Overview
Build a distributed microservices architecture implementing CQRS (Command Query Responsibility Segregation) and Event Sourcing patterns for a comprehensive order management system.

## Architecture Overview
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │    │  User Service   │    │ Product Service │
│   (NestJS)      │◄──►│   (NestJS)      │    │   (NestJS)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Order Service  │    │Notification Svc │    │ Analytics Svc   │
│   (NestJS)      │    │   (NestJS)      │    │   (NestJS)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────┐
                    │  Event Store    │
                    │  (EventStore)   │
                    └─────────────────┘
```

## Features
- Microservices architecture with NestJS
- CQRS implementation
- Event Sourcing pattern
- Distributed messaging with RabbitMQ
- API Gateway for service orchestration
- Event Store for persistence
- Read/Write model separation
- Eventual consistency
- Distributed tracing

## Tech Stack
- NestJS with TypeScript
- RabbitMQ for messaging
- EventStore for event persistence
- Redis for read models
- PostgreSQL for projections
- Docker & Docker Compose
- Prometheus & Grafana monitoring

## Core Services

### 1. API Gateway
```typescript
// apps/api-gateway/src/app.module.ts
import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { ClientsModule, Transport } from '@nestjs/microservices';
import { UsersModule } from './users/users.module';
import { ProductsModule } from './products/products.module';
import { OrdersModule } from './orders/orders.module';

@Module({
  imports: [
    ConfigModule.forRoot({ isGlobal: true }),
    ClientsModule.register([
      {
        name: 'USER_SERVICE',
        transport: Transport.RMQ,
        options: {
          urls: ['amqp://localhost:5672'],
          queue: 'user_queue',
          queueOptions: { durable: false },
        },
      },
      {
        name: 'PRODUCT_SERVICE',
        transport: Transport.RMQ,
        options: {
          urls: ['amqp://localhost:5672'],
          queue: 'product_queue',
          queueOptions: { durable: false },
        },
      },
      {
        name: 'ORDER_SERVICE',
        transport: Transport.RMQ,
        options: {
          urls: ['amqp://localhost:5672'],
          queue: 'order_queue',
          queueOptions: { durable: false },
        },
      },
    ]),
    UsersModule,
    ProductsModule,
    OrdersModule,
  ],
})
export class AppModule {}
```

```typescript
// apps/api-gateway/src/orders/orders.controller.ts
import { Controller, Post, Get, Body, Param, Inject } from '@nestjs/common';
import { ClientProxy } from '@nestjs/microservices';
import { CreateOrderDto } from './dto/create-order.dto';

@Controller('orders')
export class OrdersController {
  constructor(
    @Inject('ORDER_SERVICE') private orderService: ClientProxy,
  ) {}

  @Post()
  async createOrder(@Body() createOrderDto: CreateOrderDto) {
    return this.orderService.send('order.create', createOrderDto);
  }

  @Get(':id')
  async getOrder(@Param('id') id: string) {
    return this.orderService.send('order.get', { id });
  }

  @Get()
  async getOrders() {
    return this.orderService.send('orders.getAll', {});
  }
}
```

### 2. Order Service with CQRS and Event Sourcing
```typescript
// apps/order-service/src/order.module.ts
import { Module } from '@nestjs/common';
import { CqrsModule } from '@nestjs/cqrs';
import { OrderController } from './order.controller';
import { OrderCommandHandlers } from './commands/handlers';
import { OrderQueryHandlers } from './queries/handlers';
import { OrderEventHandlers } from './events/handlers';
import { OrderSagas } from './sagas';
import { EventStoreModule } from './event-store/event-store.module';

@Module({
  imports: [CqrsModule, EventStoreModule],
  controllers: [OrderController],
  providers: [
    ...OrderCommandHandlers,
    ...OrderQueryHandlers,
    ...OrderEventHandlers,
    ...OrderSagas,
  ],
})
export class OrderModule {}
```

### 3. Order Aggregate
```typescript
// apps/order-service/src/aggregates/order.aggregate.ts
import { AggregateRoot } from '@nestjs/cqrs';
import { OrderCreatedEvent } from '../events/order-created.event';
import { OrderItemAddedEvent } from '../events/order-item-added.event';
import { OrderConfirmedEvent } from '../events/order-confirmed.event';

export class OrderAggregate extends AggregateRoot {
  constructor(
    private readonly id: string,
    private items: OrderItem[] = [],
    private status: OrderStatus = OrderStatus.PENDING,
    private total: number = 0,
  ) {
    super();
  }

  static fromHistory(events: any[]): OrderAggregate {
    const order = new OrderAggregate('');
    events.forEach(event => {
      order.apply(event, false);
    });
    return order;
  }

  createOrder(customerId: string, items: CreateOrderItemDto[]) {
    const orderItems = items.map(item => new OrderItem(
      item.productId,
      item.quantity,
      item.price,
    ));

    const total = orderItems.reduce((sum, item) => sum + item.getTotal(), 0);

    this.apply(new OrderCreatedEvent(
      this.id,
      customerId,
      orderItems,
      total,
    ));
  }

  addItem(productId: string, quantity: number, price: number) {
    if (this.status !== OrderStatus.PENDING) {
      throw new Error('Cannot add items to non-pending order');
    }

    this.apply(new OrderItemAddedEvent(
      this.id,
      productId,
      quantity,
      price,
    ));
  }

  confirmOrder() {
    if (this.status !== OrderStatus.PENDING) {
      throw new Error('Order already confirmed or cancelled');
    }

    this.apply(new OrderConfirmedEvent(this.id, new Date()));
  }

  // Event handlers
  onOrderCreatedEvent(event: OrderCreatedEvent) {
    this.items = event.items;
    this.total = event.total;
    this.status = OrderStatus.PENDING;
  }

  onOrderItemAddedEvent(event: OrderItemAddedEvent) {
    const item = new OrderItem(event.productId, event.quantity, event.price);
    this.items.push(item);
    this.total += item.getTotal();
  }

  onOrderConfirmedEvent(event: OrderConfirmedEvent) {
    this.status = OrderStatus.CONFIRMED;
  }
}

export class OrderItem {
  constructor(
    public readonly productId: string,
    public readonly quantity: number,
    public readonly price: number,
  ) {}

  getTotal(): number {
    return this.quantity * this.price;
  }
}

export enum OrderStatus {
  PENDING = 'pending',
  CONFIRMED = 'confirmed',
  SHIPPED = 'shipped',
  DELIVERED = 'delivered',
  CANCELLED = 'cancelled',
}
```

### 4. Command Handlers
```typescript
// apps/order-service/src/commands/handlers/create-order.handler.ts
import { CommandHandler, ICommandHandler, EventPublisher } from '@nestjs/cqrs';
import { Inject } from '@nestjs/common';
import { CreateOrderCommand } from '../create-order.command';
import { OrderAggregate } from '../../aggregates/order.aggregate';
import { EventStoreService } from '../../event-store/event-store.service';

@CommandHandler(CreateOrderCommand)
export class CreateOrderHandler implements ICommandHandler<CreateOrderCommand> {
  constructor(
    @Inject('EVENT_STORE') private eventStore: EventStoreService,
    private publisher: EventPublisher,
  ) {}

  async execute(command: CreateOrderCommand): Promise<void> {
    const { orderId, customerId, items } = command;

    const order = this.publisher.mergeObjectContext(
      new OrderAggregate(orderId),
    );

    order.createOrder(customerId, items);

    // Save events to event store
    await this.eventStore.saveEvents(orderId, order.getUncommittedEvents());

    order.commit();
  }
}
```

### 5. Query Handlers
```typescript
// apps/order-service/src/queries/handlers/get-order.handler.ts
import { QueryHandler, IQueryHandler } from '@nestjs/cqrs';
import { Inject } from '@nestjs/common';
import { GetOrderQuery } from '../get-order.query';
import { OrderReadModel } from '../../read-models/order.read-model';

@QueryHandler(GetOrderQuery)
export class GetOrderHandler implements IQueryHandler<GetOrderQuery> {
  constructor(
    @Inject('ORDER_READ_REPOSITORY')
    private orderReadRepository: any,
  ) {}

  async execute(query: GetOrderQuery): Promise<OrderReadModel> {
    return this.orderReadRepository.findById(query.orderId);
  }
}
```

### 6. Event Handlers (Projections)
```typescript
// apps/order-service/src/events/handlers/order-created.handler.ts
import { EventsHandler, IEventHandler } from '@nestjs/cqrs';
import { Inject } from '@nestjs/common';
import { OrderCreatedEvent } from '../order-created.event';
import { OrderReadModel } from '../../read-models/order.read-model';

@EventsHandler(OrderCreatedEvent)
export class OrderCreatedHandler implements IEventHandler<OrderCreatedEvent> {
  constructor(
    @Inject('ORDER_READ_REPOSITORY')
    private orderReadRepository: any,
    @Inject('REDIS_CLIENT')
    private redisClient: any,
  ) {}

  async handle(event: OrderCreatedEvent) {
    // Update read model in PostgreSQL
    const orderReadModel = new OrderReadModel({
      id: event.orderId,
      customerId: event.customerId,
      items: event.items,
      total: event.total,
      status: 'pending',
      createdAt: new Date(),
    });

    await this.orderReadRepository.save(orderReadModel);

    // Update Redis cache
    await this.redisClient.setex(
      `order:${event.orderId}`,
      3600,
      JSON.stringify(orderReadModel),
    );

    // Publish to message bus for other services
    await this.publishEvent(event);
  }

  private async publishEvent(event: OrderCreatedEvent) {
    // Publish to RabbitMQ for other microservices
    // Implementation depends on your message bus setup
  }
}
```

### 7. Event Store Service
```typescript
// apps/order-service/src/event-store/event-store.service.ts
import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { EventStoreRecord } from './event-store-record.entity';

@Injectable()
export class EventStoreService {
  constructor(
    @InjectRepository(EventStoreRecord)
    private eventStoreRepository: Repository<EventStoreRecord>,
  ) {}

  async saveEvents(aggregateId: string, events: any[]): Promise<void> {
    const lastEvent = await this.eventStoreRepository.findOne({
      where: { aggregateId },
      order: { version: 'DESC' },
    });

    let version = lastEvent ? lastEvent.version : 0;

    for (const event of events) {
      version++;
      const record = new EventStoreRecord();
      record.aggregateId = aggregateId;
      record.eventType = event.constructor.name;
      record.eventData = JSON.stringify(event);
      record.version = version;
      record.timestamp = new Date();

      await this.eventStoreRepository.save(record);
    }
  }

  async getEvents(aggregateId: string): Promise<any[]> {
    const records = await this.eventStoreRepository.find({
      where: { aggregateId },
      order: { version: 'ASC' },
    });

    return records.map(record => ({
      ...JSON.parse(record.eventData),
      eventType: record.eventType,
      version: record.version,
      timestamp: record.timestamp,
    }));
  }

  async getEventsFromVersion(aggregateId: string, version: number): Promise<any[]> {
    const records = await this.eventStoreRepository.find({
      where: { aggregateId, version: MoreThan(version) },
      order: { version: 'ASC' },
    });

    return records.map(record => JSON.parse(record.eventData));
  }
}
```

### 8. Saga for Process Management
```typescript
// apps/order-service/src/sagas/order.saga.ts
import { Injectable } from '@nestjs/common';
import { Saga, ICommand, ofType } from '@nestjs/cqrs';
import { Observable } from 'rxjs';
import { map, delay } from 'rxjs/operators';
import { OrderCreatedEvent } from '../events/order-created.event';
import { ReserveInventoryCommand } from '../commands/reserve-inventory.command';
import { ProcessPaymentCommand } from '../commands/process-payment.command';

@Injectable()
export class OrderSaga {
  @Saga()
  orderCreated = (events$: Observable<any>): Observable<ICommand> => {
    return events$.pipe(
      ofType(OrderCreatedEvent),
      map(event => new ReserveInventoryCommand(event.orderId, event.items)),
    );
  };

  @Saga()
  inventoryReserved = (events$: Observable<any>): Observable<ICommand> => {
    return events$.pipe(
      ofType('InventoryReservedEvent'),
      map(event => new ProcessPaymentCommand(event.orderId, event.total)),
    );
  };

  @Saga()
  paymentProcessed = (events$: Observable<any>): Observable<ICommand> => {
    return events$.pipe(
      ofType('PaymentProcessedEvent'),
      delay(1000), // Simulate processing time
      map(event => new ConfirmOrderCommand(event.orderId)),
    );
  };
}
```

### 9. Microservice Main Files
```typescript
// apps/order-service/src/main.ts
import { NestFactory } from '@nestjs/core';
import { MicroserviceOptions, Transport } from '@nestjs/microservices';
import { OrderModule } from './order.module';

async function bootstrap() {
  const app = await NestFactory.createMicroservice<MicroserviceOptions>(
    OrderModule,
    {
      transport: Transport.RMQ,
      options: {
        urls: ['amqp://localhost:5672'],
        queue: 'order_queue',
        queueOptions: {
          durable: false,
        },
      },
    },
  );

  await app.listen();
  console.log('Order microservice is listening');
}
bootstrap();
```

### 10. Docker Compose Configuration
```yaml
# docker-compose.yml
version: '3.8'

services:
  # API Gateway
  api-gateway:
    build:
      context: .
      dockerfile: apps/api-gateway/Dockerfile
    ports:
      - "3000:3000"
    environment:
      - RABBITMQ_URL=amqp://rabbitmq:5672
    depends_on:
      - rabbitmq
      - redis

  # Order Service
  order-service:
    build:
      context: .
      dockerfile: apps/order-service/Dockerfile
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/orders
      - RABBITMQ_URL=amqp://rabbitmq:5672
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - rabbitmq
      - redis

  # User Service
  user-service:
    build:
      context: .
      dockerfile: apps/user-service/Dockerfile
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/users
      - RABBITMQ_URL=amqp://rabbitmq:5672
    depends_on:
      - postgres
      - rabbitmq

  # Product Service
  product-service:
    build:
      context: .
      dockerfile: apps/product-service/Dockerfile
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/products
      - RABBITMQ_URL=amqp://rabbitmq:5672
    depends_on:
      - postgres
      - rabbitmq

  # Infrastructure
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: microservices
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  rabbitmq:
    image: rabbitmq:3-management-alpine
    ports:
      - "5672:5672"
      - "15672:15672"
    environment:
      RABBITMQ_DEFAULT_USER: admin
      RABBITMQ_DEFAULT_PASS: password

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  # Event Store
  eventstore:
    image: eventstore/eventstore:22.10.0-buster-slim
    environment:
      - EVENTSTORE_CLUSTER_SIZE=1
      - EVENTSTORE_RUN_PROJECTIONS=All
      - EVENTSTORE_START_STANDARD_PROJECTIONS=true
      - EVENTSTORE_EXT_TCP_PORT=1113
      - EVENTSTORE_HTTP_PORT=2113
      - EVENTSTORE_INSECURE=true
      - EVENTSTORE_ENABLE_EXTERNAL_TCP=true
      - EVENTSTORE_ENABLE_ATOM_PUB_OVER_HTTP=true
    ports:
      - "1113:1113"
      - "2113:2113"

  # Monitoring
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

volumes:
  postgres_data:
```

### 11. Message Patterns
```typescript
// shared/message-patterns.ts
export const ORDER_PATTERNS = {
  CREATE_ORDER: 'order.create',
  GET_ORDER: 'order.get',
  GET_ORDERS: 'orders.getAll',
  UPDATE_ORDER_STATUS: 'order.updateStatus',
};

export const USER_PATTERNS = {
  CREATE_USER: 'user.create',
  GET_USER: 'user.get',
  UPDATE_USER: 'user.update',
};

export const PRODUCT_PATTERNS = {
  CREATE_PRODUCT: 'product.create',
  GET_PRODUCT: 'product.get',
  UPDATE_INVENTORY: 'product.updateInventory',
  RESERVE_INVENTORY: 'product.reserveInventory',
};
```

### 12. Distributed Tracing
```typescript
// shared/tracing.ts
import { Injectable, NestInterceptor, ExecutionContext, CallHandler } from '@nestjs/common';
import { Observable } from 'rxjs';
import { tap } from 'rxjs/operators';
import * as opentracing from 'opentracing';

@Injectable()
export class TracingInterceptor implements NestInterceptor {
  intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
    const tracer = opentracing.globalTracer();
    const span = tracer.startSpan('microservice-request');

    return next.handle().pipe(
      tap({
        next: () => span.setTag('success', true),
        error: (error) => {
          span.setTag('error', true);
          span.setTag('error.message', error.message);
        },
        complete: () => span.finish(),
      }),
    );
  }
}
```

## Running the Project
```bash
# Start all services
docker-compose up -d

# Build and start individual services
npm run build:api-gateway
npm run start:api-gateway

npm run build:order-service
npm run start:order-service

# Run tests
npm run test:e2e
```

## Learning Objectives
- Implement microservices architecture
- Apply CQRS and Event Sourcing patterns
- Handle distributed transactions with Sagas
- Design event-driven systems
- Implement eventual consistency
- Build scalable message-driven applications
- Handle distributed system challenges
- Apply domain-driven design principles

This advanced project demonstrates enterprise-level patterns and distributed system concepts using NestJS microservices.
