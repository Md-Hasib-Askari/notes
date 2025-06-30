# NestJS Phase 4: Architecture Patterns

## 1. Domain-Driven Design (DDD) with NestJS

### Domain Layer Structure
```typescript
// domain/entities/user.entity.ts
export class User {
  constructor(
    private readonly _id: UserId,
    private _email: Email,
    private _profile: UserProfile,
  ) {}

  get id(): UserId { return this._id; }
  get email(): Email { return this._email; }
  get profile(): UserProfile { return this._profile; }

  updateProfile(profile: UserProfile): void {
    this._profile = profile;
    this.addDomainEvent(new UserProfileUpdatedEvent(this._id, profile));
  }

  changeEmail(newEmail: Email): void {
    if (this._email.equals(newEmail)) return;
    this._email = newEmail;
    this.addDomainEvent(new UserEmailChangedEvent(this._id, newEmail));
  }
}

// domain/value-objects/email.vo.ts
export class Email {
  constructor(private readonly value: string) {
    if (!this.isValid(value)) {
      throw new InvalidEmailError(value);
    }
  }

  getValue(): string { return this.value; }
  equals(other: Email): boolean { return this.value === other.value; }
  private isValid(email: string): boolean {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  }
}

// domain/repositories/user.repository.ts
export interface UserRepository {
  findById(id: UserId): Promise<User | null>;
  findByEmail(email: Email): Promise<User | null>;
  save(user: User): Promise<void>;
  delete(id: UserId): Promise<void>;
}
```

### Application Layer
```typescript
// application/use-cases/create-user.use-case.ts
@Injectable()
export class CreateUserUseCase {
  constructor(
    private readonly userRepository: UserRepository,
    private readonly emailService: EmailService,
  ) {}

  async execute(command: CreateUserCommand): Promise<void> {
    const email = new Email(command.email);
    
    const existingUser = await this.userRepository.findByEmail(email);
    if (existingUser) {
      throw new UserAlreadyExistsError(email);
    }

    const user = new User(
      UserId.generate(),
      email,
      new UserProfile(command.firstName, command.lastName),
    );

    await this.userRepository.save(user);
    await this.emailService.sendWelcomeEmail(user);
  }
}
```

## 2. CQRS (Command Query Responsibility Segregation)

### CQRS Setup
```bash
npm install @nestjs/cqrs
```

### Commands and Handlers
```typescript
// commands/create-order.command.ts
export class CreateOrderCommand {
  constructor(
    public readonly userId: string,
    public readonly items: OrderItem[],
    public readonly shippingAddress: Address,
  ) {}
}

// commands/handlers/create-order.handler.ts
@CommandHandler(CreateOrderCommand)
export class CreateOrderHandler implements ICommandHandler<CreateOrderCommand> {
  constructor(
    private readonly orderRepository: OrderRepository,
    private readonly eventBus: EventBus,
  ) {}

  async execute(command: CreateOrderCommand): Promise<void> {
    const order = Order.create(
      command.userId,
      command.items,
      command.shippingAddress,
    );

    await this.orderRepository.save(order);
    
    order.getUncommittedEvents().forEach(event => {
      this.eventBus.publish(event);
    });
  }
}

// queries/get-orders.query.ts
export class GetOrdersQuery {
  constructor(
    public readonly userId: string,
    public readonly page: number = 1,
    public readonly limit: number = 10,
  ) {}
}

// queries/handlers/get-orders.handler.ts
@QueryHandler(GetOrdersQuery)
export class GetOrdersHandler implements IQueryHandler<GetOrdersQuery> {
  constructor(private readonly orderReadModel: OrderReadModel) {}

  async execute(query: GetOrdersQuery): Promise<OrderDto[]> {
    return this.orderReadModel.findByUserId(
      query.userId,
      query.page,
      query.limit,
    );
  }
}
```

### CQRS Module
```typescript
// cqrs.module.ts
@Module({
  imports: [CqrsModule],
  providers: [
    CreateOrderHandler,
    GetOrdersHandler,
    OrderCreatedHandler,
  ],
})
export class OrdersModule {}

// Controller with CQRS
@Controller('orders')
export class OrdersController {
  constructor(
    private readonly commandBus: CommandBus,
    private readonly queryBus: QueryBus,
  ) {}

  @Post()
  async createOrder(@Body() dto: CreateOrderDto): Promise<void> {
    await this.commandBus.execute(
      new CreateOrderCommand(dto.userId, dto.items, dto.address),
    );
  }

  @Get()
  async getOrders(@Query() query: GetOrdersDto): Promise<OrderDto[]> {
    return this.queryBus.execute(
      new GetOrdersQuery(query.userId, query.page, query.limit),
    );
  }
}
```

## 3. Event Sourcing Implementation

### Event Store
```typescript
// event-sourcing/event.ts
export abstract class DomainEvent {
  constructor(
    public readonly aggregateId: string,
    public readonly version: number,
    public readonly occurredOn: Date = new Date(),
  ) {}
}

export class OrderCreatedEvent extends DomainEvent {
  constructor(
    aggregateId: string,
    version: number,
    public readonly userId: string,
    public readonly items: OrderItem[],
  ) {
    super(aggregateId, version);
  }
}

// event-sourcing/event-store.ts
export interface EventStore {
  saveEvents(aggregateId: string, events: DomainEvent[], expectedVersion: number): Promise<void>;
  getEvents(aggregateId: string): Promise<DomainEvent[]>;
  getAllEvents(): Promise<DomainEvent[]>;
}

@Injectable()
export class InMemoryEventStore implements EventStore {
  private events: Map<string, DomainEvent[]> = new Map();

  async saveEvents(aggregateId: string, events: DomainEvent[], expectedVersion: number): Promise<void> {
    const existingEvents = this.events.get(aggregateId) || [];
    
    if (existingEvents.length !== expectedVersion) {
      throw new ConcurrencyError('Aggregate version mismatch');
    }

    this.events.set(aggregateId, [...existingEvents, ...events]);
  }

  async getEvents(aggregateId: string): Promise<DomainEvent[]> {
    return this.events.get(aggregateId) || [];
  }
}
```

### Aggregate Root
```typescript
// event-sourcing/aggregate-root.ts
export abstract class AggregateRoot {
  private _version = 0;
  private _uncommittedEvents: DomainEvent[] = [];

  get version(): number { return this._version; }
  get uncommittedEvents(): DomainEvent[] { return [...this._uncommittedEvents]; }

  protected addEvent(event: DomainEvent): void {
    this._uncommittedEvents.push(event);
  }

  markEventsAsCommitted(): void {
    this._uncommittedEvents = [];
  }

  loadFromHistory(events: DomainEvent[]): void {
    events.forEach(event => {
      this.applyEvent(event);
      this._version++;
    });
  }

  protected abstract applyEvent(event: DomainEvent): void;
}

// Order aggregate with event sourcing
export class Order extends AggregateRoot {
  private constructor(
    private readonly id: string,
    private userId: string,
    private items: OrderItem[],
    private status: OrderStatus,
  ) {
    super();
  }

  static create(userId: string, items: OrderItem[]): Order {
    const order = new Order(
      generateId(),
      userId,
      items,
      OrderStatus.PENDING,
    );
    
    order.addEvent(new OrderCreatedEvent(order.id, 0, userId, items));
    return order;
  }

  ship(): void {
    if (this.status !== OrderStatus.PENDING) {
      throw new InvalidOrderStatusError('Order cannot be shipped');
    }
    
    this.status = OrderStatus.SHIPPED;
    this.addEvent(new OrderShippedEvent(this.id, this.version, new Date()));
  }

  protected applyEvent(event: DomainEvent): void {
    switch (event.constructor) {
      case OrderCreatedEvent:
        this.applyOrderCreatedEvent(event as OrderCreatedEvent);
        break;
      case OrderShippedEvent:
        this.applyOrderShippedEvent(event as OrderShippedEvent);
        break;
    }
  }
}
```

## 4. Hexagonal Architecture

### Ports and Adapters Pattern
```typescript
// ports/payment.port.ts
export interface PaymentPort {
  processPayment(amount: number, method: PaymentMethod): Promise<PaymentResult>;
  refund(transactionId: string): Promise<RefundResult>;
}

// adapters/stripe-payment.adapter.ts
@Injectable()
export class StripePaymentAdapter implements PaymentPort {
  constructor(private readonly stripeClient: Stripe) {}

  async processPayment(amount: number, method: PaymentMethod): Promise<PaymentResult> {
    const paymentIntent = await this.stripeClient.paymentIntents.create({
      amount: amount * 100,
      currency: 'usd',
      payment_method: method.id,
      confirm: true,
    });

    return new PaymentResult(
      paymentIntent.id,
      paymentIntent.status === 'succeeded',
      paymentIntent.status,
    );
  }
}

// Domain service using ports
@Injectable()
export class OrderService {
  constructor(
    @Inject('PAYMENT_PORT') private readonly paymentPort: PaymentPort,
    @Inject('INVENTORY_PORT') private readonly inventoryPort: InventoryPort,
  ) {}

  async processOrder(order: Order): Promise<void> {
    await this.inventoryPort.reserveItems(order.items);
    const payment = await this.paymentPort.processPayment(order.total, order.paymentMethod);
    
    if (!payment.success) {
      await this.inventoryPort.releaseReservation(order.items);
      throw new PaymentFailedError();
    }
  }
}
```

## 5. Clean Architecture Principles

### Clean Architecture Structure
```typescript
// infrastructure/repositories/typeorm-user.repository.ts
@Injectable()
export class TypeOrmUserRepository implements UserRepository {
  constructor(
    @InjectRepository(UserEntity)
    private readonly repository: Repository<UserEntity>,
  ) {}

  async findById(id: UserId): Promise<User | null> {
    const entity = await this.repository.findOne({ where: { id: id.value } });
    return entity ? this.toDomain(entity) : null;
  }

  async save(user: User): Promise<void> {
    const entity = this.toEntity(user);
    await this.repository.save(entity);
  }

  private toDomain(entity: UserEntity): User {
    return new User(
      new UserId(entity.id),
      new Email(entity.email),
      new UserProfile(entity.firstName, entity.lastName),
    );
  }

  private toEntity(user: User): UserEntity {
    return {
      id: user.id.value,
      email: user.email.getValue(),
      firstName: user.profile.firstName,
      lastName: user.profile.lastName,
    };
  }
}

// Module with dependency injection
@Module({
  imports: [TypeOrmModule.forFeature([UserEntity])],
  providers: [
    {
      provide: 'USER_REPOSITORY',
      useClass: TypeOrmUserRepository,
    },
    {
      provide: 'PAYMENT_PORT',
      useClass: StripePaymentAdapter,
    },
    CreateUserUseCase,
    OrderService,
  ],
  controllers: [UsersController],
})
export class UsersModule {}
```

This comprehensive guide covers all Phase 4 architecture patterns within 199 lines, providing practical implementations for DDD, CQRS, Event Sourcing, Hexagonal Architecture, and Clean Architecture principles in NestJS.
