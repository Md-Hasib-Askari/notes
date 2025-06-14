

## 🟡 **9. Authentication with Passport in NestJS**

NestJS integrates tightly with [Passport.js](http://www.passportjs.org/), a popular authentication middleware. The most used strategies are **JWT** and **Local**.

---

### ✅ **1. Install Required Packages**

```bash
npm install @nestjs/passport passport passport-local passport-jwt
npm install @nestjs/jwt
npm install --save-dev @types/passport-local @types/passport-jwt
```

---

### ✅ **2. Define User Entity / Service**

Assume you have a `UserService` with `findByUsername` and `validatePassword`.

---

### ✅ **3. Local Strategy (Username/Password Login)**

#### Create `local.strategy.ts`

```ts
@Injectable()
export class LocalStrategy extends PassportStrategy(Strategy) {
  constructor(private authService: AuthService) {
    super();
  }

  async validate(username: string, password: string): Promise<any> {
    const user = await this.authService.validateUser(username, password);
    if (!user) {
      throw new UnauthorizedException();
    }
    return user;
  }
}
```

#### Enable in `AuthModule`

```ts
providers: [AuthService, LocalStrategy],
```

---

### ✅ **4. Auth Guard for Local Strategy**

#### `local-auth.guard.ts`

```ts
@Injectable()
export class LocalAuthGuard extends AuthGuard('local') {}
```

---

### ✅ **5. JWT Strategy**

#### `jwt.strategy.ts`

```ts
@Injectable()
export class JwtStrategy extends PassportStrategy(Strategy) {
  constructor() {
    super({
      jwtFromRequest: ExtractJwt.fromAuthHeaderAsBearerToken(),
      secretOrKey: 'jwtSecret', // or use ConfigService
    });
  }

  async validate(payload: any) {
    return { userId: payload.sub, username: payload.username };
  }
}
```

#### Enable in `AuthModule`

```ts
providers: [AuthService, JwtStrategy],
```

---

### ✅ **6. Auth Service**

#### `auth.service.ts`

```ts
@Injectable()
export class AuthService {
  constructor(
    private usersService: UsersService,
    private jwtService: JwtService,
  ) {}

  async validateUser(username: string, pass: string): Promise<any> {
    const user = await this.usersService.findByUsername(username);
    if (user && user.password === pass) {
      const { password, ...result } = user;
      return result;
    }
    return null;
  }

  async login(user: any) {
    const payload = { username: user.username, sub: user.userId };
    return {
      access_token: this.jwtService.sign(payload),
    };
  }
}
```

---

### ✅ **7. Auth Controller**

```ts
@Controller('auth')
export class AuthController {
  constructor(private authService: AuthService) {}

  @UseGuards(LocalAuthGuard)
  @Post('login')
  async login(@Request() req) {
    return this.authService.login(req.user);
  }
}
```

---

### ✅ **8. Protected Routes with JWT**

#### Create JWT Guard

```ts
@Injectable()
export class JwtAuthGuard extends AuthGuard('jwt') {}
```

#### Use it:

```ts
@UseGuards(JwtAuthGuard)
@Get('profile')
getProfile(@Request() req) {
  return req.user;
}
```

---

### ✅ **9. Optional: Refresh Tokens**

* Store refresh tokens in DB or client-side cookie.
* Create a `/refresh` route to issue new access tokens.

---

### 💪 Exercise

✅ **Implement secure login/register with JWT**

1. Use LocalStrategy for login.
2. Use JWT strategy for protecting routes.
3. Add a mock user database or Mongo/Postgres.

