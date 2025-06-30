# NestJS Phase 3: Performance & Scalability

## 1. Caching Strategies (Redis Integration)

### Redis Setup and Configuration
```bash
npm install @nestjs/cache-manager cache-manager
npm install cache-manager-redis-store redis
```

```typescript
// cache.module.ts
import { CacheModule } from '@nestjs/cache-manager';
import { redisStore } from 'cache-manager-redis-store';

@Module({
  imports: [
    CacheModule.registerAsync({
      useFactory: async () => ({
        store: redisStore,
        host: process.env.REDIS_HOST || 'localhost',
        port: process.env.REDIS_PORT || 6379,
        password: process.env.REDIS_PASSWORD,
        ttl: 60 * 60, // 1 hour default TTL
        max: 1000, // Maximum number of items in cache
      }),
    }),
  ],
  exports: [CacheModule],
})
export class CacheConfigModule {}
```

### Multi-Level Caching Strategy
```typescript
@Injectable()
export class CachingService {
  constructor(
    @Inject(CACHE_MANAGER) private cacheManager: Cache,
  ) {}

  // L1 Cache: In-memory (fastest)
  private memoryCache = new Map<string, { data: any; expiry: number }>();

  async get<T>(key: string): Promise<T | null> {
    // Check L1 cache first
    const memoryData = this.memoryCache.get(key);
    if (memoryData && memoryData.expiry > Date.now()) {
      return memoryData.data;
    }

    // Check L2 cache (Redis)
    const redisData = await this.cacheManager.get<T>(key);
    if (redisData) {
      // Store in L1 cache for faster access
      this.memoryCache.set(key, {
        data: redisData,
        expiry: Date.now() + 60000, // 1 minute in L1
      });
      return redisData;
    }

    return null;
  }

  async set<T>(key: string, value: T, ttl = 3600): Promise<void> {
    // Set in both caches
    await this.cacheManager.set(key, value, ttl);
    this.memoryCache.set(key, {
      data: value,
      expiry: Date.now() + Math.min(ttl * 1000, 60000),
    });
  }

  async invalidate(pattern: string): Promise<void> {
    // Clear from memory cache
    for (const key of this.memoryCache.keys()) {
      if (key.includes(pattern)) {
        this.memoryCache.delete(key);
      }
    }

    // Clear from Redis (requires pattern support)
    const keys = await this.getKeysPattern(pattern);
    if (keys.length > 0) {
      await this.cacheManager.store.del(...keys);
    }
  }
}
```

### Cache-Aside Pattern Implementation
```typescript
@Injectable()
export class ProductsService {
  constructor(
    private cachingService: CachingService,
    private productsRepository: Repository<Product>,
  ) {}

  async findOne(id: number): Promise<Product> {
    const cacheKey = `product:${id}`;
    
    // Try cache first
    const cached = await this.cachingService.get<Product>(cacheKey);
    if (cached) {
      return cached;
    }

    // Fetch from database
    const product = await this.productsRepository.findOne({ 
      where: { id },
      relations: ['category', 'reviews'],
    });

    if (product) {
      // Cache for 1 hour
      await this.cachingService.set(cacheKey, product, 3600);
    }

    return product;
  }

  async update(id: number, updateData: UpdateProductDto): Promise<Product> {
    const product = await this.productsRepository.save({ id, ...updateData });
    
    // Invalidate cache
    await this.cachingService.invalidate(`product:${id}`);
    await this.cachingService.invalidate('products:list');
    
    return product;
  }
}
```

### Write-Behind Caching
```typescript
@Injectable()
export class WriteBehindCacheService {
  private writeQueue = new Map<string, { data: any; timestamp: number }>();
  private readonly BATCH_SIZE = 100;
  private readonly FLUSH_INTERVAL = 5000; // 5 seconds

  constructor(
    private cachingService: CachingService,
    private dataSource: DataSource,
  ) {
    this.startFlushTimer();
  }

  async write(key: string, data: any): Promise<void> {
    // Write to cache immediately
    await this.cachingService.set(key, data);
    
    // Queue for database write
    this.writeQueue.set(key, { data, timestamp: Date.now() });
    
    if (this.writeQueue.size >= this.BATCH_SIZE) {
      await this.flushToDatabase();
    }
  }

  private async flushToDatabase(): Promise<void> {
    if (this.writeQueue.size === 0) return;

    const batch = Array.from(this.writeQueue.entries());
    this.writeQueue.clear();

    try {
      await this.dataSource.transaction(async manager => {
        for (const [key, value] of batch) {
          await this.saveToDatabase(manager, key, value.data);
        }
      });
    } catch (error) {
      // Re-queue failed writes
      for (const [key, value] of batch) {
        this.writeQueue.set(key, value);
      }
      throw error;
    }
  }

  private startFlushTimer(): void {
    setInterval(() => {
      this.flushToDatabase().catch(console.error);
    }, this.FLUSH_INTERVAL);
  }
}
```

## 2. Response Compression

### Compression Middleware
```typescript
// compression.middleware.ts
import * as compression from 'compression';

@Injectable()
export class CompressionMiddleware implements NestMiddleware {
  use(req: Request, res: Response, next: NextFunction) {
    compression({
      level: 6, // Compression level 1-9
      threshold: 1024, // Only compress responses > 1KB
      filter: (req, res) => {
        // Skip compression for images and videos
        const contentType = res.getHeader('content-type') as string;
        if (contentType?.startsWith('image/') || contentType?.startsWith('video/')) {
          return false;
        }
        return compression.filter(req, res);
      },
    })(req, res, next);
  }
}

// Apply globally
export class AppModule implements NestModule {
  configure(consumer: MiddlewareConsumer) {
    consumer.apply(CompressionMiddleware).forRoutes('*');
  }
}
```

### Selective Compression Interceptor
```typescript
@Injectable()
export class SelectiveCompressionInterceptor implements NestInterceptor {
  intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
    const response = context.switchToHttp().getResponse();
    
    return next.handle().pipe(
      map((data) => {
        const dataSize = JSON.stringify(data).length;
        
        // Only compress large responses
        if (dataSize > 5000) {
          response.setHeader('Content-Encoding', 'gzip');
          return this.compressData(data);
        }
        
        return data;
      }),
    );
  }

  private compressData(data: any): any {
    // Custom compression logic for specific data types
    if (Array.isArray(data)) {
      return this.compressArray(data);
    }
    return data;
  }

  private compressArray(array: any[]): any {
    // Remove redundant fields, use abbreviations
    return array.map(item => ({
      id: item.id,
      n: item.name, // Abbreviated field names
      p: item.price,
      // ... other optimizations
    }));
  }
}
```

## 3. Database Query Optimization

### Query Performance Monitoring
```typescript
@Injectable()
export class QueryPerformanceInterceptor implements NestInterceptor {
  private readonly logger = new Logger('QueryPerformance');

  intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
    const start = Date.now();
    
    return next.handle().pipe(
      tap(() => {
        const duration = Date.now() - start;
        const handler = context.getHandler().name;
        
        if (duration > 1000) { // Log slow queries
          this.logger.warn(`Slow query in ${handler}: ${duration}ms`);
        }
      }),
    );
  }
}
```

### Optimized Repository Patterns
```typescript
@Injectable()
export class OptimizedProductsRepository {
  constructor(
    @InjectRepository(Product) private repository: Repository<Product>,
  ) {}

  // Use projections to reduce data transfer
  async findProductSummaries(): Promise<ProductSummary[]> {
    return this.repository
      .createQueryBuilder('product')
      .select(['product.id', 'product.name', 'product.price'])
      .leftJoin('product.category', 'category')
      .addSelect(['category.name'])
      .where('product.isActive = :active', { active: true })
      .orderBy('product.createdAt', 'DESC')
      .getMany();
  }

  // Batch loading to avoid N+1 queries
  async findProductsWithReviews(productIds: number[]): Promise<Product[]> {
    return this.repository
      .createQueryBuilder('product')
      .leftJoinAndSelect('product.reviews', 'review')
      .whereInIds(productIds)
      .getMany();
  }

  // Efficient counting with EXISTS
  async countActiveProducts(): Promise<number> {
    return this.repository
      .createQueryBuilder('product')
      .where('product.isActive = :active', { active: true })
      .getCount();
  }
}
```

## 4. Memory Management

### Memory Monitoring Service
```typescript
@Injectable()
export class MemoryMonitoringService {
  private readonly logger = new Logger('MemoryMonitor');
  private readonly maxMemoryUsage = 0.8; // 80% threshold

  @Cron('*/30 * * * * *') // Every 30 seconds
  checkMemoryUsage(): void {
    const usage = process.memoryUsage();
    const usedMB = usage.heapUsed / 1024 / 1024;
    const totalMB = usage.heapTotal / 1024 / 1024;
    const usagePercent = usedMB / totalMB;

    if (usagePercent > this.maxMemoryUsage) {
      this.logger.warn(`High memory usage: ${usagePercent.toFixed(2)}%`);
      
      // Trigger garbage collection if available
      if (global.gc) {
        global.gc();
        this.logger.log('Forced garbage collection');
      }
    }
  }

  getMemoryStats(): any {
    const usage = process.memoryUsage();
    return {
      rss: Math.round(usage.rss / 1024 / 1024),
      heapTotal: Math.round(usage.heapTotal / 1024 / 1024),
      heapUsed: Math.round(usage.heapUsed / 1024 / 1024),
      external: Math.round(usage.external / 1024 / 1024),
    };
  }
}
```

### Stream Processing for Large Data
```typescript
@Injectable()
export class StreamProcessingService {
  async processLargeDataset(query: string): Promise<void> {
    const stream = this.dataSource
      .createQueryBuilder()
      .select()
      .from(LargeTable, 'table')
      .where(query)
      .stream();

    const transformStream = new Transform({
      objectMode: true,
      transform(chunk, encoding, callback) {
        // Process chunk without loading all into memory
        const processed = this.processChunk(chunk);
        callback(null, processed);
      },
    });

    const writeStream = fs.createWriteStream('output.json');

    return new Promise((resolve, reject) => {
      pipeline(stream, transformStream, writeStream, (error) => {
        if (error) reject(error);
        else resolve();
      });
    });
  }
}
```

## 5. Horizontal Scaling Considerations

### Load Balancing Configuration
```typescript
// cluster.service.ts
import * as cluster from 'cluster';
import * as os from 'os';

@Injectable()
export class ClusterService {
  static clusterize(callback: Function): void {
    if (cluster.isMaster) {
      console.log(`Master ${process.pid} is running`);

      const numCPUs = os.cpus().length;
      for (let i = 0; i < numCPUs; i++) {
        cluster.fork();
      }

      cluster.on('exit', (worker, code, signal) => {
        console.log(`Worker ${worker.process.pid} died`);
        cluster.fork(); // Restart worker
      });
    } else {
      console.log(`Worker ${process.pid} started`);
      callback();
    }
  }
}

// main.ts
async function bootstrap() {
  if (process.env.NODE_ENV === 'production') {
    ClusterService.clusterize(createApp);
  } else {
    await createApp();
  }
}

async function createApp() {
  const app = await NestFactory.create(AppModule);
  await app.listen(3000);
}
```

### Session Management for Scaling
```typescript
// session.module.ts
import * as session from 'express-session';
import * as connectRedis from 'connect-redis';

@Module({})
export class SessionModule {
  static forRoot(): DynamicModule {
    const RedisStore = connectRedis(session);
    
    return {
      module: SessionModule,
      providers: [
        {
          provide: 'SESSION_CONFIG',
          useFactory: () => ({
            store: new RedisStore({
              host: process.env.REDIS_HOST,
              port: process.env.REDIS_PORT,
            }),
            secret: process.env.SESSION_SECRET,
            resave: false,
            saveUninitialized: false,
            cookie: {
              maxAge: 24 * 60 * 60 * 1000, // 24 hours
              httpOnly: true,
              secure: process.env.NODE_ENV === 'production',
            },
          }),
        },
      ],
      exports: ['SESSION_CONFIG'],
    };
  }
}
```

### Health Checks for Load Balancers
```typescript
@Controller('health')
export class HealthController {
  constructor(
    private memoryService: MemoryMonitoringService,
    private databaseService: DatabaseHealthService,
  ) {}

  @Get()
  async checkHealth() {
    const memory = this.memoryService.getMemoryStats();
    const database = await this.databaseService.checkHealth();
    
    const isHealthy = memory.heapUsed < 1000 && database.status === 'healthy';
    
    return {
      status: isHealthy ? 'ok' : 'error',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      memory,
      database,
      environment: process.env.NODE_ENV,
    };
  }

  @Get('ready')
  checkReadiness() {
    // Quick check for load balancer
    return { status: 'ready' };
  }
}
```
