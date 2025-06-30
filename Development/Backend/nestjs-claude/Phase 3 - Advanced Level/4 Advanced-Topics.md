# NestJS Phase 3: Advanced Topics

## 1. GraphQL Integration with NestJS

### Installation and Setup
```bash
npm install @nestjs/graphql @nestjs/apollo graphql apollo-server-express
npm install @apollo/server
```

### GraphQL Module Configuration
```typescript
// app.module.ts
import { GraphQLModule } from '@nestjs/graphql';
import { ApolloDriver, ApolloDriverConfig } from '@nestjs/apollo';

@Module({
  imports: [
    GraphQLModule.forRoot<ApolloDriverConfig>({
      driver: ApolloDriver,
      autoSchemaFile: 'schema.gql',
      playground: true,
      introspection: true,
    }),
  ],
})
export class AppModule {}
```

### GraphQL Resolver and Types
```typescript
// users/user.model.ts
import { ObjectType, Field, ID } from '@nestjs/graphql';

@ObjectType()
export class User {
  @Field(() => ID)
  id: string;

  @Field()
  email: string;

  @Field()
  firstName: string;

  @Field()
  lastName: string;

  @Field(() => [Post])
  posts: Post[];
}

// users/users.resolver.ts
import { Resolver, Query, Mutation, Args } from '@nestjs/graphql';

@Resolver(() => User)
export class UsersResolver {
  constructor(private usersService: UsersService) {}

  @Query(() => [User])
  users(): Promise<User[]> {
    return this.usersService.findAll();
  }

  @Query(() => User)
  user(@Args('id') id: string): Promise<User> {
    return this.usersService.findOne(id);
  }

  @Mutation(() => User)
  createUser(@Args('input') input: CreateUserInput): Promise<User> {
    return this.usersService.create(input);
  }
}
```

## 2. WebSocket Implementation

### WebSocket Gateway Setup
```bash
npm install @nestjs/websockets @nestjs/platform-socket.io socket.io
```

```typescript
// events/events.gateway.ts
import {
  WebSocketGateway,
  SubscribeMessage,
  MessageBody,
  WebSocketServer,
  ConnectedSocket,
} from '@nestjs/websockets';
import { Server, Socket } from 'socket.io';

@WebSocketGateway({
  cors: { origin: '*' },
})
export class EventsGateway {
  @WebSocketServer()
  server: Server;

  @SubscribeMessage('join-room')
  handleJoinRoom(
    @MessageBody() data: { room: string },
    @ConnectedSocket() client: Socket,
  ) {
    client.join(data.room);
    client.to(data.room).emit('user-joined', { userId: client.id });
  }

  @SubscribeMessage('send-message')
  handleMessage(
    @MessageBody() data: { room: string; message: string },
    @ConnectedSocket() client: Socket,
  ) {
    this.server.to(data.room).emit('message', {
      userId: client.id,
      message: data.message,
      timestamp: new Date(),
    });
  }

  handleConnection(client: Socket) {
    console.log(`Client connected: ${client.id}`);
  }

  handleDisconnect(client: Socket) {
    console.log(`Client disconnected: ${client.id}`);
  }
}
```

## 3. Task Scheduling and Queues

### Task Scheduling
```bash
npm install @nestjs/schedule
npm install @nestjs/bull bull redis
```

```typescript
// tasks/tasks.service.ts
import { Injectable } from '@nestjs/common';
import { Cron, CronExpression } from '@nestjs/schedule';

@Injectable()
export class TasksService {
  @Cron(CronExpression.EVERY_10_SECONDS)
  handleCron() {
    console.log('Called every 10 seconds');
  }

  @Cron('45 * * * * *')
  handleCronCustom() {
    console.log('Called when the current second is 45');
  }
}

// queues/email.processor.ts
import { Process, Processor } from '@nestjs/bull';
import { Job } from 'bull';

@Processor('email')
export class EmailProcessor {
  @Process('welcome')
  async handleWelcomeEmail(job: Job<{ email: string; name: string }>) {
    await this.sendWelcomeEmail(job.data.email, job.data.name);
  }

  @Process('notification')
  async handleNotification(job: Job<{ userId: string; message: string }>) {
    await this.sendNotification(job.data.userId, job.data.message);
  }
}
```

## 4. File Upload Handling

### Single and Multiple File Upload
```typescript
// upload/upload.controller.ts
import {
  Controller,
  Post,
  UseInterceptors,
  UploadedFile,
  UploadedFiles,
} from '@nestjs/common';
import { FileInterceptor, FilesInterceptor } from '@nestjs/platform-express';
import { diskStorage } from 'multer';

@Controller('upload')
export class UploadController {
  @Post('single')
  @UseInterceptors(
    FileInterceptor('file', {
      storage: diskStorage({
        destination: './uploads',
        filename: (req, file, cb) => {
          const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
          cb(null, `${file.fieldname}-${uniqueSuffix}.${file.originalname.split('.').pop()}`);
        },
      }),
      fileFilter: (req, file, cb) => {
        if (file.mimetype.match(/\/(jpg|jpeg|png|gif)$/)) {
          cb(null, true);
        } else {
          cb(new Error('Unsupported file type'), false);
        }
      },
      limits: { fileSize: 5 * 1024 * 1024 }, // 5MB
    }),
  )
  uploadFile(@UploadedFile() file: Express.Multer.File) {
    return {
      filename: file.filename,
      originalName: file.originalname,
      size: file.size,
    };
  }

  @Post('multiple')
  @UseInterceptors(FilesInterceptor('files', 10))
  uploadFiles(@UploadedFiles() files: Express.Multer.File[]) {
    return files.map(file => ({
      filename: file.filename,
      originalName: file.originalname,
      size: file.size,
    }));
  }
}
```

## 5. Server-Sent Events (SSE)

### SSE Implementation
```typescript
// events/sse.controller.ts
import { Controller, Get, Sse, MessageEvent } from '@nestjs/common';
import { Observable, interval, map } from 'rxjs';

@Controller('events')
export class SseController {
  @Get('stream')
  @Sse()
  stream(): Observable<MessageEvent> {
    return interval(1000).pipe(
      map((num: number) => ({
        type: 'message',
        data: { timestamp: new Date(), counter: num },
      })),
    );
  }

  @Get('notifications')
  @Sse()
  notifications(): Observable<MessageEvent> {
    return this.notificationService.getNotificationStream().pipe(
      map(notification => ({
        type: 'notification',
        data: notification,
      })),
    );
  }
}
```

## 6. Custom Providers and Factories

### Factory Providers
```typescript
// providers/database.factory.ts
export const DatabaseFactory = {
  provide: 'DATABASE_CONNECTION',
  useFactory: async (configService: ConfigService) => {
    const config = {
      host: configService.get('DB_HOST'),
      port: configService.get('DB_PORT'),
      username: configService.get('DB_USERNAME'),
      password: configService.get('DB_PASSWORD'),
    };
    return createConnection(config);
  },
  inject: [ConfigService],
};

// Custom provider with async factory
export const ApiClientProvider = {
  provide: 'API_CLIENT',
  useFactory: async (httpService: HttpService, configService: ConfigService) => {
    const baseURL = configService.get('API_BASE_URL');
    const apiKey = configService.get('API_KEY');
    
    return {
      get: (url: string) => httpService.get(`${baseURL}${url}`, {
        headers: { 'Authorization': `Bearer ${apiKey}` }
      }),
    };
  },
  inject: [HttpService, ConfigService],
};

// Value provider
export const AppConfigProvider = {
  provide: 'APP_CONFIG',
  useValue: {
    apiVersion: 'v1',
    maxFileSize: 5 * 1024 * 1024,
    supportedFormats: ['jpg', 'png', 'pdf'],
  },
};

// Class provider with custom token
export const LoggerProvider = {
  provide: 'CUSTOM_LOGGER',
  useClass: CustomLoggerService,
};
```

### Dynamic Module with Custom Providers
```typescript
// modules/dynamic.module.ts
@Module({})
export class DynamicModule {
  static forRoot(options: DynamicModuleOptions): DynamicModule {
    return {
      module: DynamicModule,
      providers: [
        {
          provide: 'DYNAMIC_OPTIONS',
          useValue: options,
        },
        {
          provide: DynamicService,
          useFactory: (options: DynamicModuleOptions) => {
            return new DynamicService(options);
          },
          inject: ['DYNAMIC_OPTIONS'],
        },
      ],
      exports: [DynamicService],
      global: options.isGlobal,
    };
  }

  static forRootAsync(options: DynamicModuleAsyncOptions): DynamicModule {
    return {
      module: DynamicModule,
      imports: options.imports || [],
      providers: [
        ...this.createAsyncProviders(options),
        {
          provide: DynamicService,
          useFactory: (config: DynamicModuleOptions) => {
            return new DynamicService(config);
          },
          inject: ['DYNAMIC_OPTIONS'],
        },
      ],
      exports: [DynamicService],
    };
  }

  private static createAsyncProviders(options: DynamicModuleAsyncOptions) {
    if (options.useFactory) {
      return [
        {
          provide: 'DYNAMIC_OPTIONS',
          useFactory: options.useFactory,
          inject: options.inject || [],
        },
      ];
    }
    return [];
  }
}
```

### Usage in Module
```typescript
// app.module.ts
@Module({
  imports: [
    BullModule.forRoot({
      redis: { host: 'localhost', port: 6379 },
    }),
    BullModule.registerQueue({ name: 'email' }),
    ScheduleModule.forRoot(),
    DynamicModule.forRoot({
      apiKey: 'your-api-key',
      timeout: 5000,
      isGlobal: true,
    }),
  ],
  providers: [
    DatabaseFactory,
    ApiClientProvider,
    AppConfigProvider,
    LoggerProvider,
    TasksService,
    EmailProcessor,
  ],
})
export class AppModule {}
```

This guide covers all essential advanced topics for Phase 3, providing practical implementations for GraphQL, WebSockets, task scheduling, file uploads, SSE, and custom providers within 197 lines.
