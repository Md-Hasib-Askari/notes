# Phase 4: Production Readiness

## Overview
This guide covers essential topics for deploying and maintaining NestJS applications in production environments.

## 1. Logging and Monitoring Setup

### Winston Logger Integration
```typescript
// logger.module.ts
import { Module } from '@nestjs/common';
import { WinstonModule } from 'nest-winston';
import * as winston from 'winston';

@Module({
  imports: [
    WinstonModule.forRoot({
      transports: [
        new winston.transports.Console({
          format: winston.format.combine(
            winston.format.timestamp(),
            winston.format.colorize(),
            winston.format.simple(),
          ),
        }),
        new winston.transports.File({
          filename: 'logs/error.log',
          level: 'error',
          format: winston.format.combine(
            winston.format.timestamp(),
            winston.format.json(),
          ),
        }),
        new winston.transports.File({
          filename: 'logs/app.log',
          format: winston.format.combine(
            winston.format.timestamp(),
            winston.format.json(),
          ),
        }),
      ],
    }),
  ],
})
export class LoggerModule {}
```

### Custom Logger Service
```typescript
// logger.service.ts
import { Injectable, LoggerService } from '@nestjs/common';
import { createLogger, Logger, format, transports } from 'winston';

@Injectable()
export class CustomLoggerService implements LoggerService {
  private logger: Logger;

  constructor() {
    this.logger = createLogger({
      level: process.env.LOG_LEVEL || 'info',
      format: format.combine(
        format.timestamp(),
        format.errors({ stack: true }),
        format.json(),
      ),
      defaultMeta: { service: 'nestjs-app' },
      transports: [
        new transports.Console(),
        new transports.File({ filename: 'logs/app.log' }),
      ],
    });
  }

  log(message: string, context?: string) {
    this.logger.info(message, { context });
  }

  error(message: string, trace?: string, context?: string) {
    this.logger.error(message, { trace, context });
  }

  warn(message: string, context?: string) {
    this.logger.warn(message, { context });
  }

  debug(message: string, context?: string) {
    this.logger.debug(message, { context });
  }

  verbose(message: string, context?: string) {
    this.logger.verbose(message, { context });
  }
}
```

### Request Logging Middleware
```typescript
// request-logger.middleware.ts
import { Injectable, NestMiddleware } from '@nestjs/common';
import { Request, Response, NextFunction } from 'express';
import { Logger } from '@nestjs/common';

@Injectable()
export class RequestLoggerMiddleware implements NestMiddleware {
  private logger = new Logger('RequestLogger');

  use(req: Request, res: Response, next: NextFunction) {
    const startTime = Date.now();
    const { method, originalUrl, ip } = req;
    const userAgent = req.get('User-Agent') || '';

    res.on('finish', () => {
      const { statusCode } = res;
      const responseTime = Date.now() - startTime;

      this.logger.log(
        `${method} ${originalUrl} ${statusCode} ${responseTime}ms - ${userAgent} ${ip}`,
      );
    });

    next();
  }
}
```

## 2. Health Checks Implementation

### Health Check Setup
```typescript
// health.module.ts
import { Module } from '@nestjs/common';
import { TerminusModule } from '@nestjs/terminus';
import { TypeOrmModule } from '@nestjs/typeorm';
import { HealthController } from './health.controller';
import { HealthService } from './health.service';

@Module({
  imports: [TerminusModule, TypeOrmModule],
  controllers: [HealthController],
  providers: [HealthService],
})
export class HealthModule {}
```

### Health Controller
```typescript
// health.controller.ts
import { Controller, Get } from '@nestjs/common';
import {
  HealthCheckService,
  TypeOrmHealthIndicator,
  MemoryHealthIndicator,
  DiskHealthIndicator,
  HttpHealthIndicator,
  HealthCheck,
} from '@nestjs/terminus';

@Controller('health')
export class HealthController {
  constructor(
    private health: HealthCheckService,
    private db: TypeOrmHealthIndicator,
    private memory: MemoryHealthIndicator,
    private disk: DiskHealthIndicator,
    private http: HttpHealthIndicator,
  ) {}

  @Get()
  @HealthCheck()
  check() {
    return this.health.check([
      () => this.db.pingCheck('database'),
      () => this.memory.checkHeap('memory_heap', 150 * 1024 * 1024),
      () => this.memory.checkRSS('memory_rss', 150 * 1024 * 1024),
      () =>
        this.disk.checkStorage('storage', {
          path: '/',
          thresholdPercent: 0.9,
        }),
    ]);
  }

  @Get('external')
  @HealthCheck()
  checkExternal() {
    return this.health.check([
      () => this.http.pingCheck('api', 'https://api.example.com/health'),
    ]);
  }
}
```

## 3. Graceful Shutdown Handling

### Graceful Shutdown Service
```typescript
// graceful-shutdown.service.ts
import { Injectable, OnModuleDestroy } from '@nestjs/common';
import { Subject } from 'rxjs';

@Injectable()
export class GracefulShutdownService implements OnModuleDestroy {
  private shutdownListener$: Subject<void> = new Subject();

  subscribeToShutdown(shutdownFn: () => void): void {
    this.shutdownListener$.subscribe(() => shutdownFn());
  }

  shutdown(): void {
    this.shutdownListener$.next();
  }

  onModuleDestroy(): void {
    this.shutdown();
  }
}
```

### Application Shutdown Setup
```typescript
// main.ts
import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { GracefulShutdownService } from './graceful-shutdown.service';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  
  // Enable graceful shutdown
  app.enableShutdownHooks();
  
  const gracefulShutdownService = app.get(GracefulShutdownService);
  
  // Handle SIGTERM and SIGINT
  process.on('SIGTERM', async () => {
    console.log('SIGTERM received, shutting down gracefully');
    gracefulShutdownService.shutdown();
    await app.close();
    process.exit(0);
  });

  process.on('SIGINT', async () => {
    console.log('SIGINT received, shutting down gracefully');
    gracefulShutdownService.shutdown();
    await app.close();
    process.exit(0);
  });

  await app.listen(3000);
}
bootstrap();
```

## 4. Docker Containerization

### Multi-stage Dockerfile
```dockerfile
# Dockerfile
# Build stage
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY yarn.lock ./

# Install dependencies
RUN yarn install --frozen-lockfile

# Copy source code
COPY . .

# Build the application
RUN yarn build

# Production stage
FROM node:18-alpine AS production

WORKDIR /app

# Create non-root user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nestjs -u 1001

# Copy package files
COPY package*.json ./
COPY yarn.lock ./

# Install only production dependencies
RUN yarn install --frozen-lockfile --production && yarn cache clean

# Copy built application
COPY --from=builder /app/dist ./dist

# Change ownership to non-root user
RUN chown -R nestjs:nodejs /app
USER nestjs

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

# Start the application
CMD ["node", "dist/main"]
```

### Docker Compose for Development
```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgresql://user:password@db:5432/nestjs_db
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    restart: unless-stopped

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: nestjs_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

volumes:
  postgres_data:
```

## 5. CI/CD Pipeline Setup

### GitHub Actions Workflow
```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'yarn'
    
    - name: Install dependencies
      run: yarn install --frozen-lockfile
    
    - name: Run linting
      run: yarn lint
    
    - name: Run tests
      run: yarn test:cov
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: docker build -t nestjs-app:${{ github.sha }} .
    
    - name: Deploy to production
      run: |
        # Add your deployment script here
        echo "Deploying to production..."
```

## 6. Performance Monitoring and APM

### Prometheus Metrics
```typescript
// metrics.module.ts
import { Module } from '@nestjs/common';
import { PrometheusModule } from '@willsoto/nestjs-prometheus';

@Module({
  imports: [
    PrometheusModule.register({
      path: '/metrics',
      config: {
        customLabels: {
          app: 'nestjs-app',
        },
      },
    }),
  ],
})
export class MetricsModule {}
```

### Custom Metrics Service
```typescript
// metrics.service.ts
import { Injectable } from '@nestjs/common';
import { Counter, Histogram, Gauge } from 'prom-client';
import { InjectMetric } from '@willsoto/nestjs-prometheus';

@Injectable()
export class MetricsService {
  constructor(
    @InjectMetric('http_requests_total')
    private readonly httpRequestsTotal: Counter<string>,
    @InjectMetric('http_request_duration_ms')
    private readonly httpRequestDuration: Histogram<string>,
    @InjectMetric('active_connections')
    private readonly activeConnections: Gauge<string>,
  ) {}

  incrementHttpRequests(method: string, route: string, statusCode: string) {
    this.httpRequestsTotal.inc({ method, route, status_code: statusCode });
  }

  observeHttpDuration(method: string, route: string, duration: number) {
    this.httpRequestDuration.observe({ method, route }, duration);
  }

  setActiveConnections(count: number) {
    this.activeConnections.set(count);
  }
}
```

## Best Practices

### Security Headers
```typescript
// security.ts
import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import helmet from 'helmet';
import * as compression from 'compression';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  
  // Security headers
  app.use(helmet());
  
  // Compression
  app.use(compression());
  
  // CORS
  app.enableCors({
    origin: process.env.ALLOWED_ORIGINS?.split(',') || 'http://localhost:3000',
    credentials: true,
  });
  
  await app.listen(3000);
}
```

### Environment Configuration
```typescript
// config/production.ts
export default () => ({
  port: parseInt(process.env.PORT, 10) || 3000,
  database: {
    host: process.env.DB_HOST,
    port: parseInt(process.env.DB_PORT, 10) || 5432,
    username: process.env.DB_USERNAME,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_NAME,
    ssl: process.env.DB_SSL === 'true',
    synchronize: false,
    logging: process.env.DB_LOGGING === 'true',
  },
  redis: {
    host: process.env.REDIS_HOST,
    port: parseInt(process.env.REDIS_PORT, 10) || 6379,
    password: process.env.REDIS_PASSWORD,
  },
  jwt: {
    secret: process.env.JWT_SECRET,
    expiresIn: process.env.JWT_EXPIRES_IN || '1h',
  },
  logging: {
    level: process.env.LOG_LEVEL || 'info',
  },
});
```

This comprehensive guide covers all essential aspects of making your NestJS application production-ready, from monitoring to deployment strategies.
