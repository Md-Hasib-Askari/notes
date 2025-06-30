# NestJS Phase 2: Testing

## 1. Unit Testing with Jest

### Test Setup and Configuration
```bash
# Jest and testing utilities are included by default in NestJS
# Additional packages for testing
npm install -D @nestjs/testing supertest

# For database testing
npm install -D @types/supertest
```

### Jest Configuration (jest.config.js)
```javascript
module.exports = {
  moduleFileExtensions: ['js', 'json', 'ts'],
  rootDir: 'src',
  testRegex: '.*\\.spec\\.ts$',
  transform: {
    '^.+\\.(t|j)s$': 'ts-jest',
  },
  collectCoverageFrom: [
    '**/*.(t|j)s',
    '!**/*.spec.ts',
    '!**/*.interface.ts',
    '!**/dto/**',
    '!**/entities/**',
    '!main.ts',
  ],
  coverageDirectory: '../coverage',
  testEnvironment: 'node',
  coverageReporters: ['text', 'lcov', 'html'],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80,
    },
  },
};
```

### Service Unit Testing
```typescript
// users/users.service.spec.ts
import { Test, TestingModule } from '@nestjs/testing';
import { getRepositoryToken } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { NotFoundException, ConflictException } from '@nestjs/common';

import { UsersService } from './users.service';
import { User } from './entities/user.entity';
import { CreateUserDto } from './dto/create-user.dto';

describe('UsersService', () => {
  let service: UsersService;
  let repository: Repository<User>;

  const mockRepository = {
    find: jest.fn(),
    findOne: jest.fn(),
    create: jest.fn(),
    save: jest.fn(),
    update: jest.fn(),
    delete: jest.fn(),
    findAndCount: jest.fn(),
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        UsersService,
        {
          provide: getRepositoryToken(User),
          useValue: mockRepository,
        },
      ],
    }).compile();

    service = module.get<UsersService>(UsersService);
    repository = module.get<Repository<User>>(getRepositoryToken(User));
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('findAll', () => {
    it('should return an array of users', async () => {
      const users = [
        { id: 1, email: 'test1@example.com', firstName: 'John', lastName: 'Doe' },
        { id: 2, email: 'test2@example.com', firstName: 'Jane', lastName: 'Smith' },
      ];

      mockRepository.find.mockResolvedValue(users);

      const result = await service.findAll();

      expect(result).toEqual(users);
      expect(mockRepository.find).toHaveBeenCalledWith();
    });

    it('should return empty array when no users exist', async () => {
      mockRepository.find.mockResolvedValue([]);

      const result = await service.findAll();

      expect(result).toEqual([]);
    });
  });

  describe('findOne', () => {
    it('should return a user when found', async () => {
      const user = { id: 1, email: 'test@example.com', firstName: 'John', lastName: 'Doe' };
      mockRepository.findOne.mockResolvedValue(user);

      const result = await service.findOne(1);

      expect(result).toEqual(user);
      expect(mockRepository.findOne).toHaveBeenCalledWith({ where: { id: 1 } });
    });

    it('should throw NotFoundException when user not found', async () => {
      mockRepository.findOne.mockResolvedValue(null);

      await expect(service.findOne(999)).rejects.toThrow(NotFoundException);
      await expect(service.findOne(999)).rejects.toThrow('User with ID 999 not found');
    });
  });

  describe('create', () => {
    it('should create and return a new user', async () => {
      const createUserDto: CreateUserDto = {
        email: 'test@example.com',
        firstName: 'John',
        lastName: 'Doe',
        password: 'password123',
      };

      const savedUser = { id: 1, ...createUserDto };

      mockRepository.create.mockReturnValue(createUserDto);
      mockRepository.save.mockResolvedValue(savedUser);

      const result = await service.create(createUserDto);

      expect(result).toEqual(savedUser);
      expect(mockRepository.create).toHaveBeenCalledWith(createUserDto);
      expect(mockRepository.save).toHaveBeenCalledWith(createUserDto);
    });

    it('should throw ConflictException for duplicate email', async () => {
      const createUserDto: CreateUserDto = {
        email: 'test@example.com',
        firstName: 'John',
        lastName: 'Doe',
        password: 'password123',
      };

      mockRepository.create.mockReturnValue(createUserDto);
      mockRepository.save.mockRejectedValue({ code: '23505' }); // Unique constraint violation

      await expect(service.create(createUserDto)).rejects.toThrow(ConflictException);
    });
  });

  describe('update', () => {
    it('should update and return the user', async () => {
      const updateUserDto = { firstName: 'Updated Name' };
      const existingUser = { id: 1, email: 'test@example.com', firstName: 'John', lastName: 'Doe' };
      const updatedUser = { ...existingUser, ...updateUserDto };

      mockRepository.findOne.mockResolvedValue(existingUser);
      mockRepository.update.mockResolvedValue({ affected: 1 });
      mockRepository.findOne.mockResolvedValueOnce(existingUser).mockResolvedValueOnce(updatedUser);

      const result = await service.update(1, updateUserDto);

      expect(result).toEqual(updatedUser);
      expect(mockRepository.update).toHaveBeenCalledWith(1, updateUserDto);
    });
  });

  describe('remove', () => {
    it('should delete the user', async () => {
      mockRepository.delete.mockResolvedValue({ affected: 1 });

      await service.remove(1);

      expect(mockRepository.delete).toHaveBeenCalledWith(1);
    });

    it('should throw NotFoundException when user to delete not found', async () => {
      mockRepository.delete.mockResolvedValue({ affected: 0 });

      await expect(service.remove(999)).rejects.toThrow(NotFoundException);
    });
  });
});
```

### Controller Unit Testing
```typescript
// users/users.controller.spec.ts
import { Test, TestingModule } from '@nestjs/testing';
import { UsersController } from './users.controller';
import { UsersService } from './users.service';
import { CreateUserDto } from './dto/create-user.dto';
import { UpdateUserDto } from './dto/update-user.dto';

describe('UsersController', () => {
  let controller: UsersController;
  let service: UsersService;

  const mockUsersService = {
    findAll: jest.fn(),
    findOne: jest.fn(),
    create: jest.fn(),
    update: jest.fn(),
    remove: jest.fn(),
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [UsersController],
      providers: [
        {
          provide: UsersService,
          useValue: mockUsersService,
        },
      ],
    }).compile();

    controller = module.get<UsersController>(UsersController);
    service = module.get<UsersService>(UsersService);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('findAll', () => {
    it('should return an array of users', async () => {
      const users = [{ id: 1, email: 'test@example.com' }];
      mockUsersService.findAll.mockResolvedValue(users);

      const result = await controller.findAll();

      expect(result).toEqual(users);
      expect(service.findAll).toHaveBeenCalled();
    });
  });

  describe('findOne', () => {
    it('should return a user', async () => {
      const user = { id: 1, email: 'test@example.com' };
      mockUsersService.findOne.mockResolvedValue(user);

      const result = await controller.findOne('1');

      expect(result).toEqual(user);
      expect(service.findOne).toHaveBeenCalledWith(1);
    });
  });

  describe('create', () => {
    it('should create a user', async () => {
      const createUserDto: CreateUserDto = {
        email: 'test@example.com',
        firstName: 'John',
        lastName: 'Doe',
        password: 'password123',
      };
      const createdUser = { id: 1, ...createUserDto };

      mockUsersService.create.mockResolvedValue(createdUser);

      const result = await controller.create(createUserDto);

      expect(result).toEqual(createdUser);
      expect(service.create).toHaveBeenCalledWith(createUserDto);
    });
  });
});
```

### Testing Custom Providers
```typescript
// auth/auth.service.spec.ts
import { Test, TestingModule } from '@nestjs/testing';
import { JwtService } from '@nestjs/jwt';
import { ConfigService } from '@nestjs/config';
import * as bcrypt from 'bcrypt';

import { AuthService } from './auth.service';
import { UsersService } from '../users/users.service';

jest.mock('bcrypt');

describe('AuthService', () => {
  let service: AuthService;
  let usersService: UsersService;
  let jwtService: JwtService;

  const mockUsersService = {
    findByEmail: jest.fn(),
    create: jest.fn(),
  };

  const mockJwtService = {
    sign: jest.fn(),
    verify: jest.fn(),
  };

  const mockConfigService = {
    get: jest.fn(),
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        AuthService,
        { provide: UsersService, useValue: mockUsersService },
        { provide: JwtService, useValue: mockJwtService },
        { provide: ConfigService, useValue: mockConfigService },
      ],
    }).compile();

    service = module.get<AuthService>(AuthService);
    usersService = module.get<UsersService>(UsersService);
    jwtService = module.get<JwtService>(JwtService);
  });

  describe('validateUser', () => {
    it('should return user without password when credentials are valid', async () => {
      const user = {
        id: 1,
        email: 'test@example.com',
        password: 'hashedPassword',
        firstName: 'John',
        lastName: 'Doe',
      };

      mockUsersService.findByEmail.mockResolvedValue(user);
      (bcrypt.compare as jest.Mock).mockResolvedValue(true);

      const result = await service.validateUser('test@example.com', 'password');

      expect(result).toEqual({
        id: 1,
        email: 'test@example.com',
        firstName: 'John',
        lastName: 'Doe',
      });
    });

    it('should return null when user not found', async () => {
      mockUsersService.findByEmail.mockResolvedValue(null);

      const result = await service.validateUser('test@example.com', 'password');

      expect(result).toBeNull();
    });

    it('should return null when password is invalid', async () => {
      const user = { id: 1, email: 'test@example.com', password: 'hashedPassword' };
      mockUsersService.findByEmail.mockResolvedValue(user);
      (bcrypt.compare as jest.Mock).mockResolvedValue(false);

      const result = await service.validateUser('test@example.com', 'wrongPassword');

      expect(result).toBeNull();
    });
  });

  describe('login', () => {
    it('should return access token and user info', async () => {
      const user = { id: 1, email: 'test@example.com', roles: ['user'] };
      const token = 'jwt-token';

      mockJwtService.sign.mockReturnValue(token);

      const result = await service.login(user);

      expect(result).toEqual({
        access_token: token,
        user: {
          id: 1,
          email: 'test@example.com',
          roles: ['user'],
        },
      });
    });
  });
});
```

## 2. Integration Testing

### Test Database Setup
```typescript
// test/test-database.ts
import { TypeOrmModule } from '@nestjs/typeorm';
import { ConfigModule, ConfigService } from '@nestjs/config';

export const TestDatabaseModule = TypeOrmModule.forRootAsync({
  imports: [ConfigModule],
  inject: [ConfigService],
  useFactory: (configService: ConfigService) => ({
    type: 'postgres',
    host: configService.get('TEST_DATABASE_HOST', 'localhost'),
    port: configService.get('TEST_DATABASE_PORT', 5433),
    username: configService.get('TEST_DATABASE_USERNAME', 'postgres'),
    password: configService.get('TEST_DATABASE_PASSWORD', 'password'),
    database: configService.get('TEST_DATABASE_NAME', 'nestjs_test'),
    entities: [__dirname + '/../src/**/*.entity{.ts,.js}'],
    synchronize: true,
    dropSchema: true,
  }),
});
```

### Integration Test Example
```typescript
// users/users.integration.spec.ts
import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { getRepositoryToken } from '@nestjs/typeorm';

import { UsersModule } from './users.module';
import { User } from './entities/user.entity';
import { TestDatabaseModule } from '../test/test-database';

describe('UsersModule (Integration)', () => {
  let app: INestApplication;
  let userRepository: Repository<User>;

  beforeAll(async () => {
    const moduleRef: TestingModule = await Test.createTestingModule({
      imports: [
        TestDatabaseModule,
        TypeOrmModule.forFeature([User]),
        UsersModule,
      ],
    }).compile();

    app = moduleRef.createNestApplication();
    userRepository = moduleRef.get<Repository<User>>(getRepositoryToken(User));
    
    await app.init();
  });

  afterAll(async () => {
    await app.close();
  });

  beforeEach(async () => {
    await userRepository.clear();
  });

  describe('User CRUD Operations', () => {
    it('should create and retrieve a user', async () => {
      const userData = {
        email: 'test@example.com',
        firstName: 'John',
        lastName: 'Doe',
        password: 'password123',
      };

      // Create user
      const createdUser = userRepository.create(userData);
      const savedUser = await userRepository.save(createdUser);

      expect(savedUser.id).toBeDefined();
      expect(savedUser.email).toBe(userData.email);

      // Retrieve user
      const foundUser = await userRepository.findOne({ where: { id: savedUser.id } });
      expect(foundUser).toBeDefined();
      expect(foundUser.email).toBe(userData.email);
    });

    it('should update a user', async () => {
      // Create user
      const user = userRepository.create({
        email: 'test@example.com',
        firstName: 'John',
        lastName: 'Doe',
        password: 'password123',
      });
      const savedUser = await userRepository.save(user);

      // Update user
      await userRepository.update(savedUser.id, { firstName: 'Jane' });
      
      const updatedUser = await userRepository.findOne({ where: { id: savedUser.id } });
      expect(updatedUser.firstName).toBe('Jane');
    });

    it('should delete a user', async () => {
      // Create user
      const user = userRepository.create({
        email: 'test@example.com',
        firstName: 'John',
        lastName: 'Doe',
        password: 'password123',
      });
      const savedUser = await userRepository.save(user);

      // Delete user
      await userRepository.delete(savedUser.id);
      
      const deletedUser = await userRepository.findOne({ where: { id: savedUser.id } });
      expect(deletedUser).toBeNull();
    });
  });

  describe('User Relationships', () => {
    it('should handle user with orders relationship', async () => {
      // Create user with orders
      const user = userRepository.create({
        email: 'test@example.com',
        firstName: 'John',
        lastName: 'Doe',
        password: 'password123',
        orders: [
          { total: 100, status: 'pending' },
          { total: 200, status: 'completed' },
        ],
      });

      const savedUser = await userRepository.save(user);
      
      const userWithOrders = await userRepository.findOne({
        where: { id: savedUser.id },
        relations: ['orders'],
      });

      expect(userWithOrders.orders).toHaveLength(2);
      expect(userWithOrders.orders[0].total).toBe(100);
    });
  });
});
```

## 3. E2E Testing Setup

### E2E Test Configuration
```typescript
// test/app.e2e-spec.ts
import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication, ValidationPipe } from '@nestjs/common';
import * as request from 'supertest';
import { Repository } from 'typeorm';
import { getRepositoryToken } from '@nestjs/typeorm';

import { AppModule } from '../src/app.module';
import { User } from '../src/users/entities/user.entity';
import { AuthService } from '../src/auth/auth.service';

describe('AppController (e2e)', () => {
  let app: INestApplication;
  let userRepository: Repository<User>;
  let authService: AuthService;

  beforeAll(async () => {
    const moduleFixture: TestingModule = await Test.createTestingModule({
      imports: [AppModule],
    }).compile();

    app = moduleFixture.createNestApplication();
    app.useGlobalPipes(new ValidationPipe());
    
    userRepository = moduleFixture.get<Repository<User>>(getRepositoryToken(User));
    authService = moduleFixture.get<AuthService>(AuthService);
    
    await app.init();
  });

  afterAll(async () => {
    await app.close();
  });

  beforeEach(async () => {
    await userRepository.clear();
  });

  describe('Authentication Flow', () => {
    const userData = {
      email: 'test@example.com',
      firstName: 'John',
      lastName: 'Doe',
      password: 'Password123!',
    };

    it('/auth/register (POST)', () => {
      return request(app.getHttpServer())
        .post('/auth/register')
        .send(userData)
        .expect(201)
        .expect((res) => {
          expect(res.body.access_token).toBeDefined();
          expect(res.body.user.email).toBe(userData.email);
          expect(res.body.user.password).toBeUndefined();
        });
    });

    it('/auth/login (POST)', async () => {
      // First register a user
      await request(app.getHttpServer())
        .post('/auth/register')
        .send(userData);

      return request(app.getHttpServer())
        .post('/auth/login')
        .send({ email: userData.email, password: userData.password })
        .expect(200)
        .expect((res) => {
          expect(res.body.access_token).toBeDefined();
          expect(res.body.user.email).toBe(userData.email);
        });
    });

    it('/auth/login (POST) - invalid credentials', () => {
      return request(app.getHttpServer())
        .post('/auth/login')
        .send({ email: 'wrong@example.com', password: 'wrongpassword' })
        .expect(401);
    });
  });

  describe('Protected Routes', () => {
    let authToken: string;

    beforeEach(async () => {
      const userData = {
        email: 'test@example.com',
        firstName: 'John',
        lastName: 'Doe',
        password: 'Password123!',
      };

      const response = await request(app.getHttpServer())
        .post('/auth/register')
        .send(userData);

      authToken = response.body.access_token;
    });

    it('/auth/profile (GET) - with valid token', () => {
      return request(app.getHttpServer())
        .get('/auth/profile')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200)
        .expect((res) => {
          expect(res.body.email).toBe('test@example.com');
        });
    });

    it('/auth/profile (GET) - without token', () => {
      return request(app.getHttpServer())
        .get('/auth/profile')
        .expect(401);
    });

    it('/auth/profile (GET) - with invalid token', () => {
      return request(app.getHttpServer())
        .get('/auth/profile')
        .set('Authorization', 'Bearer invalid-token')
        .expect(401);
    });
  });

  describe('Users CRUD', () => {
    let authToken: string;

    beforeEach(async () => {
      const adminUser = {
        email: 'admin@example.com',
        firstName: 'Admin',
        lastName: 'User',
        password: 'Password123!',
        roles: ['admin'],
      };

      const response = await request(app.getHttpServer())
        .post('/auth/register')
        .send(adminUser);

      authToken = response.body.access_token;
    });

    it('/users (GET) - should return users list', () => {
      return request(app.getHttpServer())
        .get('/users')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200)
        .expect((res) => {
          expect(Array.isArray(res.body)).toBe(true);
        });
    });

    it('/users (POST) - should create user', () => {
      const newUser = {
        email: 'newuser@example.com',
        firstName: 'New',
        lastName: 'User',
        password: 'Password123!',
      };

      return request(app.getHttpServer())
        .post('/users')
        .set('Authorization', `Bearer ${authToken}`)
        .send(newUser)
        .expect(201)
        .expect((res) => {
          expect(res.body.email).toBe(newUser.email);
          expect(res.body.id).toBeDefined();
        });
    });

    it('/users/:id (GET) - should return specific user', async () => {
      const user = await userRepository.findOne({ where: { email: 'admin@example.com' } });

      return request(app.getHttpServer())
        .get(`/users/${user.id}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200)
        .expect((res) => {
          expect(res.body.id).toBe(user.id);
          expect(res.body.email).toBe(user.email);
        });
    });
  });

  describe('Validation', () => {
    it('/auth/register (POST) - should validate required fields', () => {
      return request(app.getHttpServer())
        .post('/auth/register')
        .send({})
        .expect(400)
        .expect((res) => {
          expect(res.body.message).toContain('validation failed');
        });
    });

    it('/auth/register (POST) - should validate email format', () => {
      return request(app.getHttpServer())
        .post('/auth/register')
        .send({
          email: 'invalid-email',
          firstName: 'John',
          lastName: 'Doe',
          password: 'Password123!',
        })
        .expect(400);
    });

    it('/auth/register (POST) - should validate password strength', () => {
      return request(app.getHttpServer())
        .post('/auth/register')
        .send({
          email: 'test@example.com',
          firstName: 'John',
          lastName: 'Doe',
          password: 'weak',
        })
        .expect(400);
    });
  });
});
```

## 4. Mocking Services and Dependencies

### Mocking External Services
```typescript
// payment/payment.service.spec.ts
import { Test, TestingModule } from '@nestjs/testing';
import { HttpService } from '@nestjs/axios';
import { of, throwError } from 'rxjs';
import { AxiosResponse } from 'axios';

import { PaymentService } from './payment.service';
import { ConfigService } from '@nestjs/config';

describe('PaymentService', () => {
  let service: PaymentService;
  let httpService: HttpService;

  const mockHttpService = {
    post: jest.fn(),
    get: jest.fn(),
  };

  const mockConfigService = {
    get: jest.fn((key: string) => {
      const config = {
        STRIPE_SECRET_KEY: 'sk_test_123',
        STRIPE_API_URL: 'https://api.stripe.com',
      };
      return config[key];
    }),
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        PaymentService,
        { provide: HttpService, useValue: mockHttpService },
        { provide: ConfigService, useValue: mockConfigService },
      ],
    }).compile();

    service = module.get<PaymentService>(PaymentService);
    httpService = module.get<HttpService>(HttpService);
  });

  describe('processPayment', () => {
    it('should process payment successfully', async () => {
      const paymentData = {
        amount: 1000,
        currency: 'usd',
        paymentMethodId: 'pm_123',
      };

      const mockResponse: AxiosResponse = {
        data: {
          id: 'pi_123',
          status: 'succeeded',
          amount: 1000,
        },
        status: 200,
        statusText: 'OK',
        headers: {},
        config: {},
      };

      mockHttpService.post.mockReturnValue(of(mockResponse));

      const result = await service.processPayment(paymentData);

      expect(result).toEqual(mockResponse.data);
      expect(httpService.post).toHaveBeenCalledWith(
        'https://api.stripe.com/v1/payment_intents',
        expect.any(Object),
        expect.any(Object),
      );
    });

    it('should handle payment failure', async () => {
      const paymentData = {
        amount: 1000,
        currency: 'usd',
        paymentMethodId: 'pm_123',
      };

      mockHttpService.post.mockReturnValue(
        throwError({
          response: {
            data: { error: { message: 'Payment failed' } },
            status: 400,
          },
        }),
      );

      await expect(service.processPayment(paymentData)).rejects.toThrow('Payment failed');
    });
  });
});
```

### Mocking Database Operations
```typescript
// Custom repository mock
const createMockRepository = () => ({
  find: jest.fn(),
  findOne: jest.fn(),
  create: jest.fn(),
  save: jest.fn(),
  update: jest.fn(),
  delete: jest.fn(),
  findAndCount: jest.fn(),
  createQueryBuilder: jest.fn(() => ({
    where: jest.fn().mockReturnThis(),
    leftJoinAndSelect: jest.fn().mockReturnThis(),
    orderBy: jest.fn().mockReturnThis(),
    getMany: jest.fn(),
    getOne: jest.fn(),
  })),
});

// Usage in test
const mockRepository = createMockRepository();

beforeEach(async () => {
  const module: TestingModule = await Test.createTestingModule({
    providers: [
      OrdersService,
      {
        provide: getRepositoryToken(Order),
        useValue: mockRepository,
      },
    ],
  }).compile();
});
```

### Mocking Custom Providers
```typescript
// email/email.service.spec.ts
import { Test, TestingModule } from '@nestjs/testing';
import { EmailService } from './email.service';
import { MAILER_SERVICE } from './email.constants';

describe('EmailService', () => {
  let service: EmailService;
  let mailerService: any;

  const mockMailerService = {
    sendEmail: jest.fn(),
    verifyConnection: jest.fn(),
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        EmailService,
        {
          provide: MAILER_SERVICE,
          useValue: mockMailerService,
        },
      ],
    }).compile();

    service = module.get<EmailService>(EmailService);
    mailerService = module.get(MAILER_SERVICE);
  });

  it('should send welcome email', async () => {
    const userEmail = 'test@example.com';
    const userName = 'John Doe';

    mockMailerService.sendEmail.mockResolvedValue({ messageId: '123' });

    await service.sendWelcomeEmail(userEmail, userName);

    expect(mailerService.sendEmail).toHaveBeenCalledWith({
      to: userEmail,
      subject: 'Welcome!',
      template: 'welcome',
      context: { name: userName },
    });
  });
});
```

## 5. Test Coverage Analysis

### Coverage Configuration
```json
// package.json
{
  "scripts": {
    "test": "jest",
    "test:watch": "jest --watch",
    "test:cov": "jest --coverage",
    "test:debug": "node --inspect-brk -r tsconfig-paths/register -r ts-node/register node_modules/.bin/jest --runInBand",
    "test:e2e": "jest --config ./test/jest-e2e.json",
    "test:e2e:cov": "jest --config ./test/jest-e2e.json --coverage"
  }
}
```

### Coverage Reports and Analysis
```bash
# Run tests with coverage
npm run test:cov

# Run E2E tests with coverage
npm run test:e2e:cov

# Generate HTML coverage report
npm run test:cov -- --coverageReporters=html

# Set coverage thresholds
npm run test:cov -- --coverageThreshold='{"global":{"branches":80,"functions":80,"lines":80,"statements":80}}'
```

### Testing Best Practices
```typescript
// Test utilities
export class TestUtils {
  static createMockUser(overrides: Partial<User> = {}): User {
    return {
      id: 1,
      email: 'test@example.com',
      firstName: 'John',
      lastName: 'Doe',
      password: 'hashedPassword',
      isActive: true,
      roles: ['user'],
      createdAt: new Date(),
      updatedAt: new Date(),
      ...overrides,
    };
  }

  static createMockOrder(overrides: Partial<Order> = {}): Order {
    return {
      id: 1,
      total: 100,
      status: 'pending',
      userId: 1,
      createdAt: new Date(),
      ...overrides,
    };
  }
}

// Custom matchers
expect.extend({
  toBeValidEmail(received) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    const pass = emailRegex.test(received);
    
    return {
      message: () => `Expected ${received} ${pass ? 'not ' : ''}to be a valid email`,
      pass,
    };
  },
});

// Usage
it('should have valid email format', () => {
  const user = TestUtils.createMockUser();
  expect(user.email).toBeValidEmail();
});
```

### GitHub Actions CI/CD for Testing
```yaml
# .github/workflows/test.yml
name: Test

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
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v2
    
    - name: Setup Node.js
      uses: actions/setup-node@v2
      with:
        node-version: '16'
        cache: 'npm'
    
    - name: Install dependencies
      run: npm ci
    
    - name: Run unit tests
      run: npm run test:cov
    
    - name: Run e2e tests
      run: npm run test:e2e
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v2
```

This comprehensive guide covers all essential testing topics for Phase 2, providing practical examples and best practices for implementing robust testing strategies in NestJS applications.
