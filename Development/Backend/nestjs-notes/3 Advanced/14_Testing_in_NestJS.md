
## 🔵 **14. Testing in NestJS**

Testing ensures your application behaves as expected. NestJS is built on top of Jest, a powerful testing framework, and supports unit testing, integration testing, and end-to-end (E2E) testing.

---

### ✅ 1. **Setup**

Jest is installed by default in NestJS projects. If missing:

```bash
npm install --save-dev jest @types/jest ts-jest
```

Nest also includes `@nestjs/testing` for easily creating testing modules.

---

### ✅ 2. **Unit Testing Services**

Unit tests isolate logic without depending on the framework or other classes.

#### Example: `auth.service.spec.ts`

```ts
describe('AuthService', () => {
  let service: AuthService;
  let userService: Partial<UserService>;

  beforeEach(async () => {
    userService = {
      findUser: jest.fn().mockResolvedValue({ id: 1, email: 'test@test.com' }),
    };

    const module = await Test.createTestingModule({
      providers: [
        AuthService,
        { provide: UserService, useValue: userService },
      ],
    }).compile();

    service = module.get<AuthService>(AuthService);
  });

  it('should validate user credentials', async () => {
    const user = await service.validateUser('test@test.com', 'password');
    expect(user.email).toBe('test@test.com');
  });
});
```

---

### ✅ 3. **Unit Testing Controllers**

Inject mocked services into controller tests.

```ts
describe('AuthController', () => {
  let controller: AuthController;

  beforeEach(async () => {
    const module = await Test.createTestingModule({
      controllers: [AuthController],
      providers: [
        {
          provide: AuthService,
          useValue: { login: jest.fn().mockResolvedValue('mock-token') },
        },
      ],
    }).compile();

    controller = module.get<AuthController>(AuthController);
  });

  it('should return a token', async () => {
    const token = await controller.login({ email: 'test', password: '1234' });
    expect(token).toBe('mock-token');
  });
});
```

---

### ✅ 4. **End-to-End (E2E) Testing**

E2E tests test the app as a black box. They typically live in `test/app.e2e-spec.ts`.

#### 📦 Install SuperTest:

```bash
npm install --save-dev supertest
```

#### Example:

```ts
import * as request from 'supertest';
import { Test } from '@nestjs/testing';
import { AppModule } from '../src/app.module';
import { INestApplication } from '@nestjs/common';

describe('AppController (e2e)', () => {
  let app: INestApplication;

  beforeAll(async () => {
    const moduleFixture = await Test.createTestingModule({
      imports: [AppModule],
    }).compile();

    app = moduleFixture.createNestApplication();
    await app.init();
  });

  it('/ (GET)', () => {
    return request(app.getHttpServer())
      .get('/')
      .expect(200)
      .expect('Hello World!');
  });
});
```

---

### ✅ 5. **Mocking Dependencies**

Mocking external dependencies is crucial for isolating tests.

```ts
const mockUserService = {
  findUser: jest.fn().mockReturnValue({ id: 1, name: 'Mock' }),
};
```

---

### 💪 Exercise

✅ **Write unit and E2E tests for your Todo or Auth module**

1. Unit test service logic and controller responses.
2. Mock all external dependencies.
3. Create an E2E test that registers a user and logs them in.
4. Use SuperTest to test real HTTP responses.

