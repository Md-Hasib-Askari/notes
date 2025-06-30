# NestJS Phase 1: Beginner Level Notes

## 1. NestJS Philosophy and Architecture

NestJS is a progressive Node.js framework built with TypeScript that uses decorators and follows the modular architecture pattern inspired by Angular.

**Core Philosophy:**
- **Modular**: Everything is organized into modules
- **Scalable**: Built for enterprise-level applications
- **Testable**: Dependency injection makes testing easier
- **Platform-agnostic**: Can work with Express or Fastify

**Architecture Overview:**
```
Application
├── Modules (Feature organization)
├── Controllers (Handle HTTP requests)
├── Services (Business logic)
└── Providers (Dependency injection)
```

## 2. Development Environment Setup

**Prerequisites:**
- Node.js (v16+)
- npm or yarn
- TypeScript knowledge

**Installation:**
```bash
npm i -g @nestjs/cli
nest new project-name
cd project-name
npm run start:dev
```

## 3. CLI Basics (@nestjs/cli)

**Essential Commands:**
```bash
nest new <app-name>           # Create new application
nest generate module <name>   # Generate module
nest g controller <name>      # Generate controller
nest g service <name>         # Generate service
nest g guard <name>           # Generate guard
nest build                    # Build application
nest start                    # Start application
```

## 4. Project Structure

```
src/
├── app.controller.ts         # Root controller
├── app.module.ts            # Root module
├── app.service.ts           # Root service
├── main.ts                  # Entry point
└── modules/                 # Feature modules
    ├── users/
    │   ├── users.module.ts
    │   ├── users.controller.ts
    │   └── users.service.ts
    └── products/
```

## 5. Modules

Modules are classes annotated with `@Module()` decorator that organize related components.

```typescript
@Module({
  imports: [],      // Other modules to import
  controllers: [],  // Controllers in this module
  providers: [],    // Services and providers
  exports: []       // What to export to other modules
})
export class UsersModule {}
```

**Feature Module Example:**
```typescript
@Module({
  controllers: [UsersController],
  providers: [UsersService],
  exports: [UsersService]
})
export class UsersModule {}
```

## 6. Controllers

Controllers handle incoming HTTP requests and return responses.

```typescript
@Controller('users')
export class UsersController {
  constructor(private usersService: UsersService) {}

  @Get()
  findAll(): User[] {
    return this.usersService.findAll();
  }

  @Get(':id')
  findOne(@Param('id') id: string): User {
    return this.usersService.findOne(id);
  }

  @Post()
  create(@Body() createUserDto: CreateUserDto): User {
    return this.usersService.create(createUserDto);
  }

  @Put(':id')
  update(@Param('id') id: string, @Body() updateUserDto: UpdateUserDto): User {
    return this.usersService.update(id, updateUserDto);
  }

  @Delete(':id')
  remove(@Param('id') id: string): void {
    return this.usersService.remove(id);
  }
}
```

## 7. Services

Services contain business logic and are injectable providers.

```typescript
@Injectable()
export class UsersService {
  private users: User[] = [];

  findAll(): User[] {
    return this.users;
  }

  findOne(id: string): User {
    return this.users.find(user => user.id === id);
  }

  create(createUserDto: CreateUserDto): User {
    const user = { id: Date.now().toString(), ...createUserDto };
    this.users.push(user);
    return user;
  }

  update(id: string, updateUserDto: UpdateUserDto): User {
    const userIndex = this.users.findIndex(user => user.id === id);
    this.users[userIndex] = { ...this.users[userIndex], ...updateUserDto };
    return this.users[userIndex];
  }

  remove(id: string): void {
    this.users = this.users.filter(user => user.id !== id);
  }
}
```

## 8. Dependency Injection

NestJS uses dependency injection to manage object creation and dependencies.

```typescript
// Service injection into controller
@Controller('users')
export class UsersController {
  constructor(private readonly usersService: UsersService) {}
}

// Multiple service injection
@Injectable()
export class UsersService {
  constructor(
    private readonly emailService: EmailService,
    private readonly databaseService: DatabaseService
  ) {}
}
```

## 9. Essential Decorators

**Controller Decorators:**
- `@Controller('path')` - Define controller route prefix
- `@Get()`, `@Post()`, `@Put()`, `@Delete()` - HTTP methods
- `@Param('id')` - Route parameters
- `@Query('search')` - Query parameters
- `@Body()` - Request body
- `@Headers()` - Request headers

**Class Decorators:**
- `@Injectable()` - Mark class as injectable provider
- `@Module()` - Define module

## 10. Request/Response Handling

```typescript
@Controller('api/users')
export class UsersController {
  @Get()
  findAll(@Query('page') page: number, @Query('limit') limit: number) {
    return this.usersService.findAll(page, limit);
  }

  @Get(':id')
  findOne(@Param('id') id: string) {
    return this.usersService.findOne(id);
  }

  @Post()
  create(@Body() createUserDto: CreateUserDto, @Res() res: Response) {
    const user = this.usersService.create(createUserDto);
    return res.status(201).json(user);
  }
}
```

## 11. Environment Configuration

```typescript
// app.module.ts
@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
      envFilePath: '.env'
    })
  ]
})
export class AppModule {}

// Using configuration
@Injectable()
export class AppService {
  constructor(private configService: ConfigService) {}

  getPort(): number {
    return this.configService.get<number>('PORT', 3000);
  }
}
```

## 12. Basic Error Handling

```typescript
@Injectable()
export class UsersService {
  findOne(id: string): User {
    const user = this.users.find(user => user.id === id);
    if (!user) {
      throw new NotFoundException(`User with ID ${id} not found`);
    }
    return user;
  }
}

// Built-in HTTP exceptions
throw new BadRequestException('Invalid input');
throw new UnauthorizedException('Access denied');
throw new ForbiddenException('Insufficient permissions');
throw new NotFoundException('Resource not found');
throw new InternalServerErrorException('Server error');
```

## 13. Best Practices for Beginners

1. **One responsibility per service**
2. **Use DTOs for data validation**
3. **Follow naming conventions**
4. **Keep controllers thin**
5. **Use dependency injection properly**
6. **Handle errors gracefully**
7. **Structure your modules logically**

## Quick Start Checklist

- [ ] Install NestJS CLI
- [ ] Create new project
- [ ] Understand project structure
- [ ] Create first module
- [ ] Implement CRUD controller
- [ ] Add service with business logic
- [ ] Test API endpoints
- [ ] Add basic error handling
- [ ] Configure environment variables
