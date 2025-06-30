# Phase 4: Advanced Integrations

## Overview
This guide covers advanced integration patterns and techniques for building complex NestJS applications that interact with external services and systems.

## 1. Third-party API Integrations

### HTTP Client Service with Retry Logic
```typescript
// http-client.service.ts
import { Injectable, Logger } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { ConfigService } from '@nestjs/config';
import { AxiosRequestConfig, AxiosResponse } from 'axios';
import { Observable, throwError } from 'rxjs';
import { retry, catchError, map } from 'rxjs/operators';

@Injectable()
export class HttpClientService {
  private readonly logger = new Logger(HttpClientService.name);

  constructor(
    private readonly httpService: HttpService,
    private readonly configService: ConfigService,
  ) {}

  get<T>(url: string, config?: AxiosRequestConfig): Observable<T> {
    return this.httpService.get<T>(url, config).pipe(
      retry(3),
      map((response: AxiosResponse<T>) => response.data),
      catchError((error) => {
        this.logger.error(`GET request failed: ${url}`, error.stack);
        return throwError(() => error);
      }),
    );
  }

  post<T>(url: string, data?: any, config?: AxiosRequestConfig): Observable<T> {
    return this.httpService.post<T>(url, data, config).pipe(
      retry(2),
      map((response: AxiosResponse<T>) => response.data),
      catchError((error) => {
        this.logger.error(`POST request failed: ${url}`, error.stack);
        return throwError(() => error);
      }),
    );
  }

  async getWithTimeout<T>(url: string, timeout = 5000): Promise<T> {
    const config: AxiosRequestConfig = {
      timeout,
      headers: {
        'Authorization': `Bearer ${this.configService.get('API_TOKEN')}`,
        'Content-Type': 'application/json',
      },
    };

    return this.get<T>(url, config).toPromise();
  }
}
```

### External API Service Example
```typescript
// external-api.service.ts
import { Injectable, BadRequestException } from '@nestjs/common';
import { HttpClientService } from './http-client.service';
import { ConfigService } from '@nestjs/config';

interface PaymentResponse {
  id: string;
  status: 'success' | 'failed' | 'pending';
  amount: number;
  currency: string;
}

@Injectable()
export class PaymentApiService {
  private readonly baseUrl: string;
  private readonly apiKey: string;

  constructor(
    private readonly httpClient: HttpClientService,
    private readonly configService: ConfigService,
  ) {
    this.baseUrl = this.configService.get<string>('PAYMENT_API_BASE_URL');
    this.apiKey = this.configService.get<string>('PAYMENT_API_KEY');
  }

  async processPayment(amount: number, currency: string, token: string): Promise<PaymentResponse> {
    try {
      const payload = { amount, currency, source: token };
      
      const response = await this.httpClient.post<PaymentResponse>(
        `${this.baseUrl}/charges`,
        payload,
        {
          headers: {
            'Authorization': `Bearer ${this.apiKey}`,
            'Stripe-Version': '2022-11-15',
          },
        }
      ).toPromise();

      return response;
    } catch (error) {
      throw new BadRequestException('Payment processing failed');
    }
  }

  async getPaymentStatus(paymentId: string): Promise<PaymentResponse> {
    return this.httpClient.getWithTimeout<PaymentResponse>(
      `${this.baseUrl}/charges/${paymentId}`
    );
  }
}
```

## 2. Message Queues and Event-Driven Architecture

### Redis Queue Implementation
```typescript
// queue.module.ts
import { Module } from '@nestjs/common';
import { BullModule } from '@nestjs/bull';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { EmailProcessor } from './processors/email.processor';
import { QueueService } from './queue.service';

@Module({
  imports: [
    BullModule.forRootAsync({
      imports: [ConfigModule],
      useFactory: async (configService: ConfigService) => ({
        redis: {
          host: configService.get('REDIS_HOST'),
          port: configService.get('REDIS_PORT'),
          password: configService.get('REDIS_PASSWORD'),
        },
        defaultJobOptions: {
          removeOnComplete: 10,
          removeOnFail: 5,
          attempts: 3,
          backoff: {
            type: 'exponential',
            delay: 2000,
          },
        },
      }),
      inject: [ConfigService],
    }),
    BullModule.registerQueue(
      { name: 'email' },
      { name: 'notifications' },
      { name: 'data-processing' },
    ),
  ],
  providers: [EmailProcessor, QueueService],
  exports: [QueueService],
})
export class QueueModule {}
```

### Queue Service
```typescript
// queue.service.ts
import { Injectable } from '@nestjs/common';
import { InjectQueue } from '@nestjs/bull';
import { Queue } from 'bull';

interface EmailJob {
  to: string;
  subject: string;
  template: string;
  data: Record<string, any>;
}

@Injectable()
export class QueueService {
  constructor(
    @InjectQueue('email') private emailQueue: Queue,
    @InjectQueue('notifications') private notificationQueue: Queue,
    @InjectQueue('data-processing') private dataProcessingQueue: Queue,
  ) {}

  async addEmailJob(emailData: EmailJob, priority = 0) {
    return this.emailQueue.add('send-email', emailData, {
      priority,
      delay: 0,
    });
  }

  async addBulkEmailJob(emails: EmailJob[]) {
    const jobs = emails.map((email, index) => ({
      name: 'send-email',
      data: email,
      opts: { priority: 5, delay: index * 1000 }, // Stagger emails
    }));

    return this.emailQueue.addBulk(jobs);
  }

  async addNotificationJob(userId: string, message: string) {
    return this.notificationQueue.add('push-notification', {
      userId,
      message,
      timestamp: new Date(),
    });
  }

  async addDataProcessingJob(data: any) {
    return this.dataProcessingQueue.add('process-data', data, {
      attempts: 5,
      backoff: {
        type: 'exponential',
        delay: 5000,
      },
    });
  }

  async getJobStatus(queueName: string, jobId: string) {
    const queue = this.getQueueByName(queueName);
    const job = await queue.getJob(jobId);
    return job ? job.opts : null;
  }

  private getQueueByName(name: string): Queue {
    switch (name) {
      case 'email':
        return this.emailQueue;
      case 'notifications':
        return this.notificationQueue;
      case 'data-processing':
        return this.dataProcessingQueue;
      default:
        throw new Error(`Queue ${name} not found`);
    }
  }
}
```

### Queue Processor
```typescript
// processors/email.processor.ts
import { Processor, Process } from '@nestjs/bull';
import { Job } from 'bull';
import { Logger } from '@nestjs/common';
import { EmailService } from '../email/email.service';

@Processor('email')
export class EmailProcessor {
  private readonly logger = new Logger(EmailProcessor.name);

  constructor(private readonly emailService: EmailService) {}

  @Process('send-email')
  async handleSendEmail(job: Job) {
    this.logger.log(`Processing email job ${job.id}`);
    
    try {
      const { to, subject, template, data } = job.data;
      await this.emailService.sendTemplatedEmail(to, subject, template, data);
      
      this.logger.log(`Email sent successfully to ${to}`);
      return { success: true, recipient: to };
    } catch (error) {
      this.logger.error(`Failed to send email: ${error.message}`, error.stack);
      throw error;
    }
  }

  @Process('bulk-email')
  async handleBulkEmail(job: Job) {
    const { emails } = job.data;
    const results = [];

    for (const email of emails) {
      try {
        await this.emailService.sendTemplatedEmail(
          email.to,
          email.subject,
          email.template,
          email.data,
        );
        results.push({ success: true, recipient: email.to });
      } catch (error) {
        results.push({ success: false, recipient: email.to, error: error.message });
      }
    }

    return results;
  }
}
```

## 3. Event-Driven Architecture with Event Emitters

### Event Module Setup
```typescript
// events/events.module.ts
import { Module } from '@nestjs/common';
import { EventEmitterModule } from '@nestjs/event-emitter';
import { UserEventsListener } from './listeners/user-events.listener';
import { OrderEventsListener } from './listeners/order-events.listener';

@Module({
  imports: [
    EventEmitterModule.forRoot({
      wildcard: false,
      delimiter: '.',
      newListener: false,
      removeListener: false,
      maxListeners: 10,
      verboseMemoryLeak: false,
      ignoreErrors: false,
    }),
  ],
  providers: [UserEventsListener, OrderEventsListener],
})
export class EventsModule {}
```

### Event DTOs
```typescript
// events/dto/user-created.event.ts
export class UserCreatedEvent {
  constructor(
    public readonly userId: string,
    public readonly email: string,
    public readonly name: string,
    public readonly createdAt: Date,
  ) {}
}

// events/dto/order-completed.event.ts
export class OrderCompletedEvent {
  constructor(
    public readonly orderId: string,
    public readonly userId: string,
    public readonly amount: number,
    public readonly items: any[],
  ) {}
}
```

### Event Listeners
```typescript
// events/listeners/user-events.listener.ts
import { Injectable, Logger } from '@nestjs/common';
import { OnEvent } from '@nestjs/event-emitter';
import { UserCreatedEvent } from '../dto/user-created.event';
import { EmailService } from '../../email/email.service';
import { QueueService } from '../../queue/queue.service';

@Injectable()
export class UserEventsListener {
  private readonly logger = new Logger(UserEventsListener.name);

  constructor(
    private readonly emailService: EmailService,
    private readonly queueService: QueueService,
  ) {}

  @OnEvent('user.created')
  async handleUserCreated(event: UserCreatedEvent) {
    this.logger.log(`Handling user created event for user: ${event.userId}`);

    // Send welcome email
    await this.queueService.addEmailJob({
      to: event.email,
      subject: 'Welcome to our platform!',
      template: 'welcome',
      data: { name: event.name },
    });

    // Create user profile
    await this.createUserProfile(event);

    // Send analytics event
    await this.trackUserRegistration(event);
  }

  @OnEvent('user.verified')
  async handleUserVerified(payload: { userId: string; email: string }) {
    await this.queueService.addEmailJob({
      to: payload.email,
      subject: 'Account verified successfully',
      template: 'account-verified',
      data: { userId: payload.userId },
    });
  }

  private async createUserProfile(event: UserCreatedEvent) {
    // Implementation for creating user profile
    this.logger.log(`Creating profile for user: ${event.userId}`);
  }

  private async trackUserRegistration(event: UserCreatedEvent) {
    // Implementation for analytics tracking
    this.logger.log(`Tracking registration for user: ${event.userId}`);
  }
}
```

### Service with Event Emission
```typescript
// users/users.service.ts
import { Injectable } from '@nestjs/common';
import { EventEmitter2 } from '@nestjs/event-emitter';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { User } from './entities/user.entity';
import { CreateUserDto } from './dto/create-user.dto';
import { UserCreatedEvent } from '../events/dto/user-created.event';

@Injectable()
export class UsersService {
  constructor(
    @InjectRepository(User)
    private readonly usersRepository: Repository<User>,
    private readonly eventEmitter: EventEmitter2,
  ) {}

  async create(createUserDto: CreateUserDto): Promise<User> {
    const user = this.usersRepository.create(createUserDto);
    const savedUser = await this.usersRepository.save(user);

    // Emit user created event
    const userCreatedEvent = new UserCreatedEvent(
      savedUser.id,
      savedUser.email,
      savedUser.name,
      savedUser.createdAt,
    );

    this.eventEmitter.emit('user.created', userCreatedEvent);

    return savedUser;
  }

  async verifyUser(userId: string): Promise<void> {
    await this.usersRepository.update(userId, { isVerified: true });
    
    const user = await this.usersRepository.findOne({ where: { id: userId } });
    
    this.eventEmitter.emit('user.verified', {
      userId: user.id,
      email: user.email,
    });
  }
}
```

## 4. Database Sharding Strategies

### Database Configuration for Multiple Connections
```typescript
// database/database.module.ts
import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { ConfigModule, ConfigService } from '@nestjs/config';

@Module({
  imports: [
    // Primary database
    TypeOrmModule.forRootAsync({
      name: 'default',
      imports: [ConfigModule],
      useFactory: (configService: ConfigService) => ({
        type: 'postgres',
        host: configService.get('DB_PRIMARY_HOST'),
        port: configService.get('DB_PRIMARY_PORT'),
        username: configService.get('DB_PRIMARY_USERNAME'),
        password: configService.get('DB_PRIMARY_PASSWORD'),
        database: configService.get('DB_PRIMARY_NAME'),
        entities: ['dist/**/*.entity{.ts,.js}'],
        synchronize: false,
      }),
      inject: [ConfigService],
    }),
    
    // Read replica
    TypeOrmModule.forRootAsync({
      name: 'read-replica',
      imports: [ConfigModule],
      useFactory: (configService: ConfigService) => ({
        type: 'postgres',
        host: configService.get('DB_REPLICA_HOST'),
        port: configService.get('DB_REPLICA_PORT'),
        username: configService.get('DB_REPLICA_USERNAME'),
        password: configService.get('DB_REPLICA_PASSWORD'),
        database: configService.get('DB_REPLICA_NAME'),
        entities: ['dist/**/*.entity{.ts,.js}'],
        synchronize: false,
      }),
      inject: [ConfigService],
    }),
    
    // Analytics database
    TypeOrmModule.forRootAsync({
      name: 'analytics',
      imports: [ConfigModule],
      useFactory: (configService: ConfigService) => ({
        type: 'postgres',
        host: configService.get('DB_ANALYTICS_HOST'),
        port: configService.get('DB_ANALYTICS_PORT'),
        username: configService.get('DB_ANALYTICS_USERNAME'),
        password: configService.get('DB_ANALYTICS_PASSWORD'),
        database: configService.get('DB_ANALYTICS_NAME'),
        entities: ['dist/**/*.analytics-entity{.ts,.js}'],
        synchronize: false,
      }),
      inject: [ConfigService],
    }),
  ],
})
export class DatabaseModule {}
```

### Sharding Service
```typescript
// database/sharding.service.ts
import { Injectable } from '@nestjs/common';
import { InjectConnection } from '@nestjs/typeorm';
import { Connection } from 'typeorm';

@Injectable()
export class ShardingService {
  constructor(
    @InjectConnection('default') private primaryConnection: Connection,
    @InjectConnection('read-replica') private replicaConnection: Connection,
    @InjectConnection('analytics') private analyticsConnection: Connection,
  ) {}

  getConnectionForUser(userId: string): Connection {
    // Simple sharding based on user ID
    const shard = this.getUserShard(userId);
    
    switch (shard) {
      case 0:
        return this.primaryConnection;
      case 1:
        return this.replicaConnection;
      default:
        return this.primaryConnection;
    }
  }

  getReadConnection(): Connection {
    // Use read replica for read operations
    return this.replicaConnection;
  }

  getWriteConnection(): Connection {
    // Use primary for write operations
    return this.primaryConnection;
  }

  getAnalyticsConnection(): Connection {
    return this.analyticsConnection;
  }

  private getUserShard(userId: string): number {
    // Simple hash-based sharding
    const hash = this.hashUserId(userId);
    return hash % 2; // 2 shards in this example
  }

  private hashUserId(userId: string): number {
    let hash = 0;
    for (let i = 0; i < userId.length; i++) {
      const char = userId.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }
}
```

## 5. Multi-tenant Applications

### Tenant Context Service
```typescript
// tenant/tenant-context.service.ts
import { Injectable, Scope } from '@nestjs/common';

@Injectable({ scope: Scope.REQUEST })
export class TenantContextService {
  private tenantId: string;

  setTenantId(tenantId: string): void {
    this.tenantId = tenantId;
  }

  getTenantId(): string {
    return this.tenantId;
  }
}
```

### Tenant Middleware
```typescript
// tenant/tenant.middleware.ts
import { Injectable, NestMiddleware, BadRequestException } from '@nestjs/common';
import { Request, Response, NextFunction } from 'express';
import { TenantContextService } from './tenant-context.service';

@Injectable()
export class TenantMiddleware implements NestMiddleware {
  constructor(private readonly tenantContext: TenantContextService) {}

  use(req: Request, res: Response, next: NextFunction) {
    const tenantId = req.headers['x-tenant-id'] as string || 
                    req.subdomains[0] ||
                    this.extractTenantFromDomain(req.hostname);

    if (!tenantId) {
      throw new BadRequestException('Tenant ID not provided');
    }

    this.tenantContext.setTenantId(tenantId);
    next();
  }

  private extractTenantFromDomain(hostname: string): string {
    // Extract tenant from subdomain
    const parts = hostname.split('.');
    return parts.length > 2 ? parts[0] : null;
  }
}
```

### Tenant-aware Repository
```typescript
// tenant/tenant-aware.repository.ts
import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository, SelectQueryBuilder } from 'typeorm';
import { TenantContextService } from './tenant-context.service';

@Injectable()
export class TenantAwareRepository<T> {
  constructor(
    @InjectRepository(Entity)
    private readonly repository: Repository<T>,
    private readonly tenantContext: TenantContextService,
  ) {}

  async find(options?: any): Promise<T[]> {
    return this.repository.find({
      ...options,
      where: {
        ...options?.where,
        tenantId: this.tenantContext.getTenantId(),
      },
    });
  }

  async findOne(options?: any): Promise<T> {
    return this.repository.findOne({
      ...options,
      where: {
        ...options?.where,
        tenantId: this.tenantContext.getTenantId(),
      },
    });
  }

  async save(entity: any): Promise<T> {
    entity.tenantId = this.tenantContext.getTenantId();
    return this.repository.save(entity);
  }

  createQueryBuilder(alias?: string): SelectQueryBuilder<T> {
    const qb = this.repository.createQueryBuilder(alias);
    return qb.where(`${alias || 'entity'}.tenantId = :tenantId`, {
      tenantId: this.tenantContext.getTenantId(),
    });
  }
}
```

## Best Practices

### Error Handling for External Services
```typescript
// common/filters/external-api-exception.filter.ts
import {
  ExceptionFilter,
  Catch,
  ArgumentsHost,
  HttpStatus,
  Logger,
} from '@nestjs/common';
import { Response } from 'express';
import { AxiosError } from 'axios';

@Catch(AxiosError)
export class ExternalApiExceptionFilter implements ExceptionFilter {
  private readonly logger = new Logger(ExternalApiExceptionFilter.name);

  catch(exception: AxiosError, host: ArgumentsHost) {
    const ctx = host.switchToHttp();
    const response = ctx.getResponse<Response>();

    this.logger.error(
      `External API error: ${exception.message}`,
      exception.stack,
    );

    const status = exception.response?.status || HttpStatus.SERVICE_UNAVAILABLE;
    const message = 'External service temporarily unavailable';

    response.status(status).json({
      statusCode: status,
      timestamp: new Date().toISOString(),
      message,
    });
  }
}
```

### Circuit Breaker Pattern
```typescript
// common/circuit-breaker.service.ts
import { Injectable, Logger } from '@nestjs/common';

interface CircuitBreakerOptions {
  failureThreshold: number;
  recoveryTimeout: number;
  monitoringPeriod: number;
}

@Injectable()
export class CircuitBreakerService {
  private readonly logger = new Logger(CircuitBreakerService.name);
  private failureCount = 0;
  private lastFailureTime = 0;
  private state: 'CLOSED' | 'OPEN' | 'HALF_OPEN' = 'CLOSED';

  constructor(private readonly options: CircuitBreakerOptions) {}

  async execute<T>(operation: () => Promise<T>): Promise<T> {
    if (this.state === 'OPEN') {
      if (Date.now() - this.lastFailureTime > this.options.recoveryTimeout) {
        this.state = 'HALF_OPEN';
      } else {
        throw new Error('Circuit breaker is OPEN');
      }
    }

    try {
      const result = await operation();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  private onSuccess(): void {
    this.failureCount = 0;
    this.state = 'CLOSED';
  }

  private onFailure(): void {
    this.failureCount++;
    this.lastFailureTime = Date.now();

    if (this.failureCount >= this.options.failureThreshold) {
      this.state = 'OPEN';
      this.logger.warn('Circuit breaker opened due to failures');
    }
  }
}
```

This guide provides comprehensive patterns for advanced integrations in NestJS applications, enabling robust and scalable system architectures.
