# NestJS Phase 1: Key Topics to Master

## 1. Module System and Feature Modules

### Understanding Modules
Modules are the fundamental building blocks in NestJS that organize your application into cohesive blocks of functionality.

**Root Module (app.module.ts):**
```typescript
@Module({
  imports: [UsersModule, ProductsModule, OrdersModule],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
```

### Feature Modules
Feature modules encapsulate related functionality and promote code organization.

**Creating a Feature Module:**
```typescript
// users/users.module.ts
@Module({
  imports: [TypeOrmModule.forFeature([User])],
  controllers: [UsersController],
  providers: [UsersService, UsersRepository],
  exports: [UsersService] // Export for use in other modules
})
export class UsersModule {}
```

**Shared Modules:**
```typescript
// shared/shared.module.ts
@Module({
  providers: [EmailService, LoggerService],
  exports: [EmailService, LoggerService],
})
export class SharedModule {}

// Make it global
@Global()
@Module({
  providers: [ConfigService],
  exports: [ConfigService],
})
export class CoreModule {}
```

**Dynamic Modules:**
```typescript
@Module({})
export class DatabaseModule {
  static forRoot(options: DatabaseOptions): DynamicModule {
    return {
      module: DatabaseModule,
      providers: [
        {
          provide: 'DATABASE_OPTIONS',
          useValue: options,
        },
        DatabaseService,
      ],
      exports: [DatabaseService],
    };
  }
}
```

## 2. Controller Request Handling

### Basic Controller Structure
Controllers handle incoming HTTP requests and return responses to the client.

```typescript
@Controller('api/v1/users')
export class UsersController {
  constructor(private readonly usersService: UsersService) {}

  // GET /api/v1/users
  @Get()
  async findAll(
    @Query('page') page: number = 1,
    @Query('limit') limit: number = 10,
    @Query('search') search?: string
  ) {
    return this.usersService.findAll({ page, limit, search });
  }

  // GET /api/v1/users/:id
  @Get(':id')
  async findOne(@Param('id', ParseIntPipe) id: number) {
    return this.usersService.findOne(id);
  }

  // POST /api/v1/users
  @Post()
  @HttpCode(HttpStatus.CREATED)
  async create(@Body() createUserDto: CreateUserDto) {
    return this.usersService.create(createUserDto);
  }

  // PUT /api/v1/users/:id
  @Put(':id')
  async update(
    @Param('id', ParseIntPipe) id: number,
    @Body() updateUserDto: UpdateUserDto
  ) {
    return this.usersService.update(id, updateUserDto);
  }

  // DELETE /api/v1/users/:id
  @Delete(':id')
  @HttpCode(HttpStatus.NO_CONTENT)
  async remove(@Param('id', ParseIntPipe) id: number) {
    return this.usersService.remove(id);
  }
}
```

### Advanced Request Handling
```typescript
@Controller('users')
export class UsersController {
  // Handle file uploads
  @Post('upload')
  @UseInterceptors(FileInterceptor('file'))
  uploadFile(@UploadedFile() file: Express.Multer.File) {
    return { filename: file.filename, size: file.size };
  }

  // Custom headers and status codes
  @Post('custom')
  @Header('X-Custom-Header', 'Custom Value')
  @HttpCode(HttpStatus.ACCEPTED)
  customResponse(@Body() data: any) {
    return { message: 'Custom response', data };
  }

  // Async operation with proper error handling
  @Get(':id/profile')
  async getUserProfile(@Param('id') id: string) {
    try {
      const user = await this.usersService.findOne(id);
      const profile = await this.usersService.getProfile(id);
      return { user, profile };
    } catch (error) {
      throw new BadRequestException('Failed to fetch user profile');
    }
  }
}
```

## 3. Service Layer and Business Logic Separation

### Service Design Principles
Services contain business logic and should be independent of HTTP concerns.

```typescript
@Injectable()
export class UsersService {
  constructor(
    private readonly usersRepository: UsersRepository,
    private readonly emailService: EmailService,
    private readonly logger: Logger
  ) {}

  async findAll(options: FindUsersOptions): Promise<PaginatedResult<User>> {
    this.logger.log('Fetching users with options', options);
    
    const { users, total } = await this.usersRepository.findAndCount(options);
    
    return {
      data: users,
      total,
      page: options.page,
      limit: options.limit,
      totalPages: Math.ceil(total / options.limit)
    };
  }

  async create(createUserDto: CreateUserDto): Promise<User> {
    // Business logic validation
    await this.validateUniqueEmail(createUserDto.email);
    
    // Create user
    const user = await this.usersRepository.create(createUserDto);
    
    // Side effects (send welcome email)
    await this.emailService.sendWelcomeEmail(user);
    
    this.logger.log(`User created: ${user.id}`);
    return user;
  }

  async update(id: number, updateUserDto: UpdateUserDto): Promise<User> {
    const user = await this.findOne(id);
    
    // Business logic for email change
    if (updateUserDto.email && updateUserDto.email !== user.email) {
      await this.validateUniqueEmail(updateUserDto.email);
      updateUserDto.emailVerified = false;
    }
    
    return this.usersRepository.update(id, updateUserDto);
  }

  private async validateUniqueEmail(email: string): Promise<void> {
    const existingUser = await this.usersRepository.findByEmail(email);
    if (existingUser) {
      throw new ConflictException('Email already exists');
    }
  }
}
```

### Repository Pattern
```typescript
@Injectable()
export class UsersRepository {
  constructor(
    @InjectRepository(User)
    private readonly userRepository: Repository<User>
  ) {}

  async findAndCount(options: FindUsersOptions) {
    const queryBuilder = this.userRepository.createQueryBuilder('user');
    
    if (options.search) {
      queryBuilder.where(
        'user.email ILIKE :search OR user.firstName ILIKE :search',
        { search: `%${options.search}%` }
      );
    }
    
    queryBuilder
      .skip((options.page - 1) * options.limit)
      .take(options.limit)
      .orderBy('user.createdAt', 'DESC');
    
    return queryBuilder.getManyAndCount();
  }

  async findByEmail(email: string): Promise<User | null> {
    return this.userRepository.findOne({ where: { email } });
  }
}
```

## 4. Basic Error Handling

### Built-in HTTP Exceptions
```typescript
@Injectable()
export class UsersService {
  async findOne(id: number): Promise<User> {
    const user = await this.usersRepository.findOne(id);
    
    if (!user) {
      throw new NotFoundException(`User with ID ${id} not found`);
    }
    
    return user;
  }

  async create(createUserDto: CreateUserDto): Promise<User> {
    try {
      return await this.usersRepository.create(createUserDto);
    } catch (error) {
      if (error.code === '23505') { // Unique constraint violation
        throw new ConflictException('Email already exists');
      }
      throw new InternalServerErrorException('Failed to create user');
    }
  }

  async deleteUser(id: number): Promise<void> {
    const user = await this.findOne(id);
    
    if (user.isAdmin) {
      throw new ForbiddenException('Cannot delete admin users');
    }
    
    await this.usersRepository.delete(id);
  }
}
```

### Custom Exception Handling
```typescript
// Custom exception
export class UserNotActiveException extends HttpException {
  constructor() {
    super('User account is not active', HttpStatus.FORBIDDEN);
  }
}

// Global exception filter
@Catch()
export class AllExceptionsFilter implements ExceptionFilter {
  catch(exception: unknown, host: ArgumentsHost) {
    const ctx = host.switchToHttp();
    const response = ctx.getResponse<Response>();
    const request = ctx.getRequest<Request>();

    const status = exception instanceof HttpException
      ? exception.getStatus()
      : HttpStatus.INTERNAL_SERVER_ERROR;

    const message = exception instanceof HttpException
      ? exception.getResponse()
      : 'Internal server error';

    response.status(status).json({
      statusCode: status,
      timestamp: new Date().toISOString(),
      path: request.url,
      message,
    });
  }
}
```

## 5. Environment Configuration

### Configuration Setup
```typescript
// config/configuration.ts
export default () => ({
  port: parseInt(process.env.PORT, 10) || 3000,
  database: {
    host: process.env.DATABASE_HOST || 'localhost',
    port: parseInt(process.env.DATABASE_PORT, 10) || 5432,
    username: process.env.DATABASE_USERNAME || 'postgres',
    password: process.env.DATABASE_PASSWORD || 'password',
    database: process.env.DATABASE_NAME || 'nestjs_app',
  },
  jwt: {
    secret: process.env.JWT_SECRET || 'default-secret',
    expiresIn: process.env.JWT_EXPIRES_IN || '1d',
  },
  email: {
    host: process.env.EMAIL_HOST,
    port: parseInt(process.env.EMAIL_PORT, 10) || 587,
    user: process.env.EMAIL_USER,
    password: process.env.EMAIL_PASSWORD,
  },
});
```

### Module Configuration
```typescript
// app.module.ts
@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
      load: [configuration],
      envFilePath: ['.env.local', '.env'],
      validationSchema: Joi.object({
        NODE_ENV: Joi.string().valid('development', 'production', 'test'),
        PORT: Joi.number().default(3000),
        DATABASE_HOST: Joi.string().required(),
        JWT_SECRET: Joi.string().required(),
      }),
    }),
  ],
})
export class AppModule {}
```

### Using Configuration in Services
```typescript
@Injectable()
export class AppService {
  constructor(private configService: ConfigService) {}

  getDatabaseConfig() {
    return {
      host: this.configService.get<string>('database.host'),
      port: this.configService.get<number>('database.port'),
    };
  }

  getJwtSecret(): string {
    return this.configService.get<string>('jwt.secret');
  }
}
```

### Environment File Example (.env)
```bash
NODE_ENV=development
PORT=3000

# Database
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_USERNAME=postgres
DATABASE_PASSWORD=password
DATABASE_NAME=nestjs_app

# JWT
JWT_SECRET=your-super-secret-key-here
JWT_EXPIRES_IN=24h

# Email
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USER=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
```

## Best Practices Summary

1. **Modules**: Keep modules focused and cohesive
2. **Controllers**: Handle only HTTP concerns, delegate business logic to services
3. **Services**: Implement business logic, maintain single responsibility
4. **Error Handling**: Use appropriate HTTP exceptions and global filters
5. **Configuration**: Use environment variables for all configurable values
6. **Testing**: Write unit tests for services and integration tests for controllers
7. **Documentation**: Use Swagger/OpenAPI for API documentation
