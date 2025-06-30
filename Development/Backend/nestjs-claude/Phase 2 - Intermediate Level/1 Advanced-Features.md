# NestJS Phase 2: Advanced Features

## 1. Middleware Implementation and Usage

Middleware functions execute during the request-response cycle and have access to request and response objects.

### Creating Middleware
```typescript
// middleware/logger.middleware.ts
@Injectable()
export class LoggerMiddleware implements NestMiddleware {
  use(req: Request, res: Response, next: NextFunction) {
    const { method, originalUrl } = req;
    const timestamp = new Date().toISOString();
    
    console.log(`[${timestamp}] ${method} ${originalUrl}`);
    
    // Measure response time
    const start = Date.now();
    res.on('finish', () => {
      const duration = Date.now() - start;
      console.log(`[${timestamp}] ${method} ${originalUrl} - ${res.statusCode} (${duration}ms)`);
    });
    
    next();
  }
}

// Functional middleware
export function corsMiddleware(req: Request, res: Response, next: NextFunction) {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE');
  res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  next();
}
```

### Applying Middleware
```typescript
// app.module.ts
export class AppModule implements NestModule {
  configure(consumer: MiddlewareConsumer) {
    consumer
      .apply(LoggerMiddleware)
      .forRoutes('*'); // Apply to all routes
    
    consumer
      .apply(corsMiddleware)
      .exclude({ path: 'health', method: RequestMethod.GET })
      .forRoutes(UsersController);
    
    // Route-specific middleware
    consumer
      .apply(AuthMiddleware)
      .forRoutes(
        { path: 'users', method: RequestMethod.POST },
        { path: 'users/*', method: RequestMethod.PUT }
      );
  }
}
```

### Authentication Middleware
```typescript
@Injectable()
export class AuthMiddleware implements NestMiddleware {
  constructor(private jwtService: JwtService) {}

  use(req: Request, res: Response, next: NextFunction) {
    const token = req.headers.authorization?.replace('Bearer ', '');
    
    if (!token) {
      throw new UnauthorizedException('Token not found');
    }
    
    try {
      const payload = this.jwtService.verify(token);
      req['user'] = payload;
      next();
    } catch (error) {
      throw new UnauthorizedException('Invalid token');
    }
  }
}
```

## 2. Guards for Authentication and Authorization

Guards determine whether a request should be handled by the route handler.

### Authentication Guard
```typescript
@Injectable()
export class JwtAuthGuard implements CanActivate {
  constructor(private jwtService: JwtService) {}

  canActivate(context: ExecutionContext): boolean | Promise<boolean> {
    const request = context.switchToHttp().getRequest();
    const token = this.extractTokenFromHeader(request);
    
    if (!token) {
      throw new UnauthorizedException('Access token required');
    }
    
    try {
      const payload = this.jwtService.verify(token);
      request.user = payload;
      return true;
    } catch (error) {
      throw new UnauthorizedException('Invalid or expired token');
    }
  }

  private extractTokenFromHeader(request: Request): string | undefined {
    const [type, token] = request.headers.authorization?.split(' ') ?? [];
    return type === 'Bearer' ? token : undefined;
  }
}
```

### Role-Based Authorization Guard
```typescript
@Injectable()
export class RolesGuard implements CanActivate {
  constructor(private reflector: Reflector) {}

  canActivate(context: ExecutionContext): boolean {
    const requiredRoles = this.reflector.getAllAndOverride<Role[]>('roles', [
      context.getHandler(),
      context.getClass(),
    ]);
    
    if (!requiredRoles) {
      return true;
    }
    
    const { user } = context.switchToHttp().getRequest();
    return requiredRoles.some((role) => user.roles?.includes(role));
  }
}

// Custom decorator for roles
export const Roles = (...roles: Role[]) => SetMetadata('roles', roles);

// Usage in controller
@Controller('admin')
@UseGuards(JwtAuthGuard, RolesGuard)
export class AdminController {
  @Get('users')
  @Roles(Role.Admin, Role.SuperAdmin)
  getUsers() {
    return this.usersService.findAll();
  }
}
```

### Resource Ownership Guard
```typescript
@Injectable()
export class OwnershipGuard implements CanActivate {
  constructor(private usersService: UsersService) {}

  async canActivate(context: ExecutionContext): Promise<boolean> {
    const request = context.switchToHttp().getRequest();
    const userId = request.params.id;
    const currentUser = request.user;
    
    // Allow admin access
    if (currentUser.roles.includes(Role.Admin)) {
      return true;
    }
    
    // Check if user owns the resource
    return currentUser.sub === userId;
  }
}
```

## 3. Interceptors for Request/Response Transformation

Interceptors have access to request/response and can transform the result.

### Logging Interceptor
```typescript
@Injectable()
export class LoggingInterceptor implements NestInterceptor {
  intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
    const request = context.switchToHttp().getRequest();
    const { method, url } = request;
    const now = Date.now();
    
    console.log(`Incoming Request: ${method} ${url}`);
    
    return next.handle().pipe(
      tap(() => {
        console.log(`Request completed: ${method} ${url} - ${Date.now() - now}ms`);
      }),
      catchError((error) => {
        console.error(`Request failed: ${method} ${url} - ${error.message}`);
        throw error;
      })
    );
  }
}
```

### Transform Response Interceptor
```typescript
@Injectable()
export class TransformInterceptor<T> implements NestInterceptor<T, Response<T>> {
  intercept(context: ExecutionContext, next: CallHandler): Observable<Response<T>> {
    return next.handle().pipe(
      map(data => ({
        success: true,
        timestamp: new Date().toISOString(),
        path: context.switchToHttp().getRequest().url,
        data,
      }))
    );
  }
}

interface Response<T> {
  success: boolean;
  timestamp: string;
  path: string;
  data: T;
}
```

### Cache Interceptor
```typescript
@Injectable()
export class CacheInterceptor implements NestInterceptor {
  constructor(private cacheService: CacheService) {}

  intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
    const request = context.switchToHttp().getRequest();
    const cacheKey = `${request.method}:${request.url}`;
    
    // Only cache GET requests
    if (request.method !== 'GET') {
      return next.handle();
    }
    
    const cachedResult = this.cacheService.get(cacheKey);
    if (cachedResult) {
      return of(cachedResult);
    }
    
    return next.handle().pipe(
      tap(result => {
        this.cacheService.set(cacheKey, result, 300); // Cache for 5 minutes
      })
    );
  }
}
```

## 4. Pipes for Validation and Transformation

Pipes transform input data and validate it before it reaches the route handler.

### Validation Pipe with DTOs
```typescript
// dto/create-user.dto.ts
export class CreateUserDto {
  @IsString()
  @IsNotEmpty()
  @Length(2, 50)
  firstName: string;

  @IsString()
  @IsNotEmpty()
  @Length(2, 50)
  lastName: string;

  @IsEmail()
  email: string;

  @IsString()
  @MinLength(8)
  @Matches(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/, {
    message: 'Password must contain uppercase, lowercase and number'
  })
  password: string;

  @IsOptional()
  @IsArray()
  @IsString({ each: true })
  roles?: string[];
}

// Custom validation pipe
@Injectable()
export class CustomValidationPipe implements PipeTransform<any> {
  async transform(value: any, { metatype }: ArgumentMetadata) {
    if (!metatype || !this.toValidate(metatype)) {
      return value;
    }
    
    const object = plainToClass(metatype, value);
    const errors = await validate(object);
    
    if (errors.length > 0) {
      const errorMessages = errors.map(error => ({
        field: error.property,
        errors: Object.values(error.constraints || {}),
      }));
      throw new BadRequestException({
        message: 'Validation failed',
        errors: errorMessages,
      });
    }
    
    return object;
  }

  private toValidate(metatype: Function): boolean {
    const types: Function[] = [String, Boolean, Number, Array, Object];
    return !types.includes(metatype);
  }
}
```

### Transform Pipe
```typescript
@Injectable()
export class ParseIntPipe implements PipeTransform<string, number> {
  transform(value: string, metadata: ArgumentMetadata): number {
    const val = parseInt(value, 10);
    if (isNaN(val)) {
      throw new BadRequestException('Invalid number format');
    }
    return val;
  }
}

// Trim pipe for strings
@Injectable()
export class TrimPipe implements PipeTransform {
  transform(value: any) {
    if (typeof value === 'string') {
      return value.trim();
    }
    if (typeof value === 'object' && value !== null) {
      Object.keys(value).forEach(key => {
        if (typeof value[key] === 'string') {
          value[key] = value[key].trim();
        }
      });
    }
    return value;
  }
}
```

## 5. Exception Filters for Error Handling

Exception filters handle all unhandled exceptions across the application.

### Global Exception Filter
```typescript
@Catch()
export class AllExceptionsFilter implements ExceptionFilter {
  private readonly logger = new Logger(AllExceptionsFilter.name);

  catch(exception: unknown, host: ArgumentsHost) {
    const ctx = host.switchToHttp();
    const response = ctx.getResponse<Response>();
    const request = ctx.getRequest<Request>();

    let status: number;
    let message: string | object;

    if (exception instanceof HttpException) {
      status = exception.getStatus();
      message = exception.getResponse();
    } else if (exception instanceof Error) {
      status = HttpStatus.INTERNAL_SERVER_ERROR;
      message = 'Internal server error';
      this.logger.error(exception.message, exception.stack);
    } else {
      status = HttpStatus.INTERNAL_SERVER_ERROR;
      message = 'Unknown error occurred';
    }

    const errorResponse = {
      statusCode: status,
      timestamp: new Date().toISOString(),
      path: request.url,
      method: request.method,
      message,
    };

    response.status(status).json(errorResponse);
  }
}
```

### Specific Exception Filter
```typescript
@Catch(TypeOrmError)
export class DatabaseExceptionFilter implements ExceptionFilter {
  catch(exception: TypeOrmError, host: ArgumentsHost) {
    const ctx = host.switchToHttp();
    const response = ctx.getResponse<Response>();
    const request = ctx.getRequest<Request>();

    let status = HttpStatus.INTERNAL_SERVER_ERROR;
    let message = 'Database error occurred';

    // Handle specific database errors
    if (exception.message.includes('duplicate key')) {
      status = HttpStatus.CONFLICT;
      message = 'Resource already exists';
    } else if (exception.message.includes('foreign key')) {
      status = HttpStatus.BAD_REQUEST;
      message = 'Invalid reference to related resource';
    }

    response.status(status).json({
      statusCode: status,
      timestamp: new Date().toISOString(),
      path: request.url,
      message,
    });
  }
}
```

## 6. Custom Decorators Creation

Custom decorators enhance code reusability and expressiveness.

### User Decorator
```typescript
// decorators/user.decorator.ts
export const CurrentUser = createParamDecorator(
  (data: string, ctx: ExecutionContext) => {
    const request = ctx.switchToHttp().getRequest();
    const user = request.user;
    
    return data ? user?.[data] : user;
  },
);

// Usage in controller
@Controller('profile')
export class ProfileController {
  @Get()
  @UseGuards(JwtAuthGuard)
  getProfile(@CurrentUser() user: any) {
    return this.profileService.getProfile(user.id);
  }

  @Get('email')
  @UseGuards(JwtAuthGuard)
  getEmail(@CurrentUser('email') email: string) {
    return { email };
  }
}
```

### API Response Decorator
```typescript
export function ApiSuccessResponse(description: string, type?: any) {
  return applyDecorators(
    ApiOkResponse({
      description,
      type,
      schema: {
        properties: {
          success: { type: 'boolean', default: true },
          timestamp: { type: 'string', format: 'date-time' },
          data: type ? { $ref: getSchemaPath(type) } : { type: 'object' },
        },
      },
    }),
  );
}

// Usage
@Controller('users')
export class UsersController {
  @Get()
  @ApiSuccessResponse('Users retrieved successfully', [User])
  findAll() {
    return this.usersService.findAll();
  }
}
```

### Rate Limiting Decorator
```typescript
export const RateLimit = (limit: number, windowMs: number) => {
  return applyDecorators(
    SetMetadata('rateLimit', { limit, windowMs }),
    UseGuards(RateLimitGuard),
  );
};

@Injectable()
export class RateLimitGuard implements CanActivate {
  private requests = new Map<string, number[]>();

  canActivate(context: ExecutionContext): boolean {
    const request = context.switchToHttp().getRequest();
    const rateLimitConfig = this.reflector.get('rateLimit', context.getHandler());
    
    if (!rateLimitConfig) return true;

    const key = `${request.ip}:${request.route.path}`;
    const now = Date.now();
    const windowStart = now - rateLimitConfig.windowMs;
    
    const requestTimes = this.requests.get(key) || [];
    const validRequests = requestTimes.filter(time => time > windowStart);
    
    if (validRequests.length >= rateLimitConfig.limit) {
      throw new HttpException('Rate limit exceeded', HttpStatus.TOO_MANY_REQUESTS);
    }
    
    validRequests.push(now);
    this.requests.set(key, validRequests);
    
    return true;
  }
}

// Usage
@Controller('api')
export class ApiController {
  @Post('upload')
  @RateLimit(5, 60000) // 5 requests per minute
  uploadFile() {
    // Upload logic
  }
}
```

## Implementation Example

```typescript
// app.module.ts
@Module({
  providers: [
    {
      provide: APP_INTERCEPTOR,
      useClass: LoggingInterceptor,
    },
    {
      provide: APP_PIPE,
      useClass: CustomValidationPipe,
    },
    {
      provide: APP_FILTER,
      useClass: AllExceptionsFilter,
    },
    {
      provide: APP_GUARD,
      useClass: JwtAuthGuard,
    },
  ],
})
export class AppModule implements NestModule {
  configure(consumer: MiddlewareConsumer) {
    consumer.apply(LoggerMiddleware).forRoutes('*');
  }
}
```

This comprehensive guide covers all advanced features in Phase 2, providing practical implementations and real-world usage patterns for building robust NestJS applications.
