# NestJS Phase 2: Authentication & Security

## 1. JWT Implementation

### Installation and Setup
```bash
# Install required packages
npm install @nestjs/jwt @nestjs/passport passport passport-jwt
npm install -D @types/passport-jwt

# For bcrypt password hashing
npm install bcrypt
npm install -D @types/bcrypt
```

### JWT Module Configuration
```typescript
// auth/auth.module.ts
import { JwtModule } from '@nestjs/jwt';
import { ConfigModule, ConfigService } from '@nestjs/config';

@Module({
  imports: [
    ConfigModule,
    UsersModule,
    JwtModule.registerAsync({
      inject: [ConfigService],
      useFactory: (configService: ConfigService) => ({
        secret: configService.get<string>('JWT_SECRET'),
        signOptions: {
          expiresIn: configService.get<string>('JWT_EXPIRES_IN', '1d'),
        },
      }),
    }),
  ],
  providers: [AuthService, LocalStrategy, JwtStrategy],
  controllers: [AuthController],
  exports: [AuthService],
})
export class AuthModule {}
```

### Auth Service Implementation
```typescript
// auth/auth.service.ts
@Injectable()
export class AuthService {
  constructor(
    private usersService: UsersService,
    private jwtService: JwtService,
  ) {}

  async validateUser(email: string, password: string): Promise<any> {
    const user = await this.usersService.findByEmail(email);
    
    if (user && await bcrypt.compare(password, user.password)) {
      const { password, ...result } = user;
      return result;
    }
    
    return null;
  }

  async login(user: any) {
    const payload = { 
      email: user.email, 
      sub: user.id, 
      roles: user.roles 
    };
    
    return {
      access_token: this.jwtService.sign(payload),
      refresh_token: this.jwtService.sign(payload, { expiresIn: '7d' }),
      user: {
        id: user.id,
        email: user.email,
        firstName: user.firstName,
        lastName: user.lastName,
        roles: user.roles,
      },
    };
  }

  async register(createUserDto: CreateUserDto) {
    const hashedPassword = await bcrypt.hash(createUserDto.password, 10);
    
    const user = await this.usersService.create({
      ...createUserDto,
      password: hashedPassword,
    });

    return this.login(user);
  }

  async refreshToken(refreshToken: string) {
    try {
      const payload = this.jwtService.verify(refreshToken);
      const user = await this.usersService.findOne(payload.sub);
      
      return this.login(user);
    } catch (error) {
      throw new UnauthorizedException('Invalid refresh token');
    }
  }

  async forgotPassword(email: string) {
    const user = await this.usersService.findByEmail(email);
    if (!user) {
      throw new NotFoundException('User not found');
    }

    const resetToken = this.jwtService.sign(
      { sub: user.id, type: 'reset' },
      { expiresIn: '1h' }
    );

    // Send email with reset token
    await this.emailService.sendPasswordResetEmail(user.email, resetToken);
    
    return { message: 'Password reset email sent' };
  }

  async resetPassword(token: string, newPassword: string) {
    try {
      const payload = this.jwtService.verify(token);
      
      if (payload.type !== 'reset') {
        throw new UnauthorizedException('Invalid token type');
      }

      const hashedPassword = await bcrypt.hash(newPassword, 10);
      await this.usersService.updatePassword(payload.sub, hashedPassword);
      
      return { message: 'Password reset successfully' };
    } catch (error) {
      throw new UnauthorizedException('Invalid or expired reset token');
    }
  }
}
```

### Auth Controller
```typescript
// auth/auth.controller.ts
@Controller('auth')
export class AuthController {
  constructor(private authService: AuthService) {}

  @Post('register')
  @HttpCode(HttpStatus.CREATED)
  async register(@Body() createUserDto: CreateUserDto) {
    return this.authService.register(createUserDto);
  }

  @UseGuards(LocalAuthGuard)
  @Post('login')
  @HttpCode(HttpStatus.OK)
  async login(@Request() req) {
    return this.authService.login(req.user);
  }

  @Post('refresh')
  @HttpCode(HttpStatus.OK)
  async refresh(@Body('refresh_token') refreshToken: string) {
    return this.authService.refreshToken(refreshToken);
  }

  @Post('forgot-password')
  @HttpCode(HttpStatus.OK)
  async forgotPassword(@Body('email') email: string) {
    return this.authService.forgotPassword(email);
  }

  @Post('reset-password')
  @HttpCode(HttpStatus.OK)
  async resetPassword(
    @Body('token') token: string,
    @Body('password') password: string,
  ) {
    return this.authService.resetPassword(token, password);
  }

  @UseGuards(JwtAuthGuard)
  @Get('profile')
  getProfile(@Request() req) {
    return req.user;
  }

  @UseGuards(JwtAuthGuard)
  @Post('logout')
  @HttpCode(HttpStatus.OK)
  async logout(@Request() req) {
    // Implement token blacklisting if needed
    return { message: 'Logged out successfully' };
  }
}
```

## 2. Passport.js Integration

### Local Strategy (Username/Password)
```typescript
// auth/strategies/local.strategy.ts
import { Strategy } from 'passport-local';
import { PassportStrategy } from '@nestjs/passport';

@Injectable()
export class LocalStrategy extends PassportStrategy(Strategy) {
  constructor(private authService: AuthService) {
    super({
      usernameField: 'email', // Use email instead of username
    });
  }

  async validate(email: string, password: string): Promise<any> {
    const user = await this.authService.validateUser(email, password);
    
    if (!user) {
      throw new UnauthorizedException('Invalid credentials');
    }
    
    return user;
  }
}
```

### JWT Strategy
```typescript
// auth/strategies/jwt.strategy.ts
import { ExtractJwt, Strategy } from 'passport-jwt';
import { PassportStrategy } from '@nestjs/passport';

@Injectable()
export class JwtStrategy extends PassportStrategy(Strategy) {
  constructor(private configService: ConfigService) {
    super({
      jwtFromRequest: ExtractJwt.fromAuthHeaderAsBearerToken(),
      ignoreExpiration: false,
      secretOrKey: configService.get<string>('JWT_SECRET'),
    });
  }

  async validate(payload: any) {
    return {
      id: payload.sub,
      email: payload.email,
      roles: payload.roles,
    };
  }
}
```

### Google OAuth Strategy
```typescript
// auth/strategies/google.strategy.ts
import { Strategy, VerifyCallback } from 'passport-google-oauth20';
import { PassportStrategy } from '@nestjs/passport';

@Injectable()
export class GoogleStrategy extends PassportStrategy(Strategy, 'google') {
  constructor(
    private authService: AuthService,
    private configService: ConfigService,
  ) {
    super({
      clientID: configService.get<string>('GOOGLE_CLIENT_ID'),
      clientSecret: configService.get<string>('GOOGLE_CLIENT_SECRET'),
      callbackURL: configService.get<string>('GOOGLE_CALLBACK_URL'),
      scope: ['email', 'profile'],
    });
  }

  async validate(
    accessToken: string,
    refreshToken: string,
    profile: any,
    done: VerifyCallback,
  ): Promise<any> {
    const { name, emails, photos } = profile;
    
    const user = {
      email: emails[0].value,
      firstName: name.givenName,
      lastName: name.familyName,
      picture: photos[0].value,
      accessToken,
    };

    const existingUser = await this.authService.findOrCreateGoogleUser(user);
    done(null, existingUser);
  }
}
```

### Auth Guards
```typescript
// auth/guards/local-auth.guard.ts
@Injectable()
export class LocalAuthGuard extends AuthGuard('local') {}

// auth/guards/jwt-auth.guard.ts
@Injectable()
export class JwtAuthGuard extends AuthGuard('jwt') {
  canActivate(context: ExecutionContext) {
    return super.canActivate(context);
  }

  handleRequest(err, user, info) {
    if (err || !user) {
      throw err || new UnauthorizedException();
    }
    return user;
  }
}

// auth/guards/google-auth.guard.ts
@Injectable()
export class GoogleAuthGuard extends AuthGuard('google') {}
```

## 3. Role-Based Access Control (RBAC)

### Role and Permission Entities
```typescript
// entities/role.entity.ts
@Entity('roles')
export class Role {
  @PrimaryGeneratedColumn()
  id: number;

  @Column({ unique: true })
  name: string;

  @Column()
  description: string;

  @ManyToMany(() => Permission, permission => permission.roles)
  @JoinTable()
  permissions: Permission[];

  @ManyToMany(() => User, user => user.roles)
  users: User[];
}

// entities/permission.entity.ts
@Entity('permissions')
export class Permission {
  @PrimaryGeneratedColumn()
  id: number;

  @Column({ unique: true })
  name: string;

  @Column()
  resource: string;

  @Column()
  action: string;

  @ManyToMany(() => Role, role => role.permissions)
  roles: Role[];
}
```

### Roles Guard
```typescript
// auth/guards/roles.guard.ts
@Injectable()
export class RolesGuard implements CanActivate {
  constructor(private reflector: Reflector) {}

  canActivate(context: ExecutionContext): boolean {
    const requiredRoles = this.reflector.getAllAndOverride<string[]>('roles', [
      context.getHandler(),
      context.getClass(),
    ]);

    if (!requiredRoles) {
      return true;
    }

    const { user } = context.switchToHttp().getRequest();
    
    return requiredRoles.some((role) => 
      user.roles?.some((userRole) => userRole.name === role)
    );
  }
}
```

### Permission Guard
```typescript
// auth/guards/permissions.guard.ts
@Injectable()
export class PermissionsGuard implements CanActivate {
  constructor(private reflector: Reflector) {}

  canActivate(context: ExecutionContext): boolean {
    const requiredPermissions = this.reflector.getAllAndOverride<string[]>('permissions', [
      context.getHandler(),
      context.getClass(),
    ]);

    if (!requiredPermissions) {
      return true;
    }

    const { user } = context.switchToHttp().getRequest();
    
    const userPermissions = user.roles?.flatMap(role => 
      role.permissions?.map(permission => permission.name) || []
    ) || [];

    return requiredPermissions.every((permission) =>
      userPermissions.includes(permission)
    );
  }
}
```

### Custom Decorators for RBAC
```typescript
// decorators/roles.decorator.ts
export const Roles = (...roles: string[]) => SetMetadata('roles', roles);

// decorators/permissions.decorator.ts
export const Permissions = (...permissions: string[]) => 
  SetMetadata('permissions', permissions);

// decorators/public.decorator.ts
export const Public = () => SetMetadata('isPublic', true);

// Usage in controllers
@Controller('admin')
@UseGuards(JwtAuthGuard, RolesGuard)
export class AdminController {
  @Get('users')
  @Roles('admin', 'super-admin')
  getUsers() {
    return this.usersService.findAll();
  }

  @Delete('users/:id')
  @Permissions('users:delete')
  deleteUser(@Param('id') id: string) {
    return this.usersService.remove(id);
  }
}
```

## 4. Rate Limiting and Security Headers

### Rate Limiting Setup
```bash
npm install @nestjs/throttler
```

```typescript
// app.module.ts
import { ThrottlerModule, ThrottlerGuard } from '@nestjs/throttler';

@Module({
  imports: [
    ThrottlerModule.forRoot({
      ttl: 60, // Time window in seconds
      limit: 10, // Max requests per time window
    }),
  ],
  providers: [
    {
      provide: APP_GUARD,
      useClass: ThrottlerGuard,
    },
  ],
})
export class AppModule {}
```

### Custom Rate Limiting
```typescript
// Custom throttler decorator
export const ApiRateLimit = (limit: number, ttl: number) => 
  applyDecorators(
    Throttle(limit, ttl),
    ApiTooManyRequestsResponse({ description: 'Rate limit exceeded' })
  );

// Usage
@Controller('auth')
export class AuthController {
  @Post('login')
  @ApiRateLimit(5, 60) // 5 requests per minute
  async login(@Body() loginDto: LoginDto) {
    return this.authService.login(loginDto);
  }

  @Post('register')
  @ApiRateLimit(3, 3600) // 3 registrations per hour
  async register(@Body() registerDto: RegisterDto) {
    return this.authService.register(registerDto);
  }
}
```

### Security Headers Middleware
```typescript
// middleware/security.middleware.ts
@Injectable()
export class SecurityMiddleware implements NestMiddleware {
  use(req: Request, res: Response, next: NextFunction) {
    // Security headers
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('X-Frame-Options', 'DENY');
    res.setHeader('X-XSS-Protection', '1; mode=block');
    res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
    res.setHeader('Permissions-Policy', 'camera=(), microphone=(), geolocation=()');
    
    // HSTS (HTTP Strict Transport Security)
    if (req.secure || req.headers['x-forwarded-proto'] === 'https') {
      res.setHeader('Strict-Transport-Security', 'max-age=31536000; includeSubDomains');
    }
    
    next();
  }
}

// Apply globally
export class AppModule implements NestModule {
  configure(consumer: MiddlewareConsumer) {
    consumer.apply(SecurityMiddleware).forRoutes('*');
  }
}
```

### Helmet Integration
```bash
npm install helmet
```

```typescript
// main.ts
import * as helmet from 'helmet';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  
  app.use(helmet({
    contentSecurityPolicy: {
      directives: {
        defaultSrc: ["'self'"],
        styleSrc: ["'self'", "'unsafe-inline'"],
        scriptSrc: ["'self'"],
        imgSrc: ["'self'", "data:", "https:"],
      },
    },
  }));
  
  await app.listen(3000);
}
```

## 5. Input Validation with class-validator

### Installation and Setup
```bash
npm install class-validator class-transformer
```

### DTOs with Validation
```typescript
// dto/create-user.dto.ts
import { 
  IsEmail, 
  IsString, 
  MinLength, 
  MaxLength, 
  Matches, 
  IsOptional,
  IsArray,
  ValidateNested,
  IsPhoneNumber,
  IsDateString
} from 'class-validator';
import { Transform, Type } from 'class-transformer';

export class CreateUserDto {
  @IsEmail({}, { message: 'Please provide a valid email address' })
  @Transform(({ value }) => value.toLowerCase().trim())
  email: string;

  @IsString()
  @MinLength(2, { message: 'First name must be at least 2 characters long' })
  @MaxLength(50, { message: 'First name must not exceed 50 characters' })
  @Transform(({ value }) => value.trim())
  firstName: string;

  @IsString()
  @MinLength(2)
  @MaxLength(50)
  @Transform(({ value }) => value.trim())
  lastName: string;

  @IsString()
  @MinLength(8, { message: 'Password must be at least 8 characters long' })
  @Matches(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/, {
    message: 'Password must contain uppercase, lowercase, number and special character'
  })
  password: string;

  @IsOptional()
  @IsPhoneNumber(null, { message: 'Please provide a valid phone number' })
  phoneNumber?: string;

  @IsOptional()
  @IsDateString({}, { message: 'Please provide a valid date' })
  dateOfBirth?: string;

  @IsOptional()
  @IsArray()
  @IsString({ each: true })
  interests?: string[];
}

// dto/address.dto.ts
export class AddressDto {
  @IsString()
  @MinLength(5)
  street: string;

  @IsString()
  city: string;

  @IsString()
  @Matches(/^\d{5}(-\d{4})?$/, { message: 'Invalid postal code format' })
  postalCode: string;

  @IsString()
  country: string;
}

// dto/user-with-address.dto.ts
export class UserWithAddressDto extends CreateUserDto {
  @ValidateNested()
  @Type(() => AddressDto)
  address: AddressDto;
}
```

### Custom Validators
```typescript
// validators/is-unique.validator.ts
@ValidatorConstraint({ name: 'IsUnique', async: true })
@Injectable()
export class IsUniqueConstraint implements ValidatorConstraintInterface {
  constructor(private dataSource: DataSource) {}

  async validate(value: any, args: ValidationArguments) {
    const [entityClass, property] = args.constraints;
    const repository = this.dataSource.getRepository(entityClass);
    
    const entity = await repository.findOne({ where: { [property]: value } });
    return !entity;
  }

  defaultMessage(args: ValidationArguments) {
    const [, property] = args.constraints;
    return `${property} must be unique`;
  }
}

// Custom decorator
export function IsUnique(entityClass: any, property: string, validationOptions?: ValidationOptions) {
  return function (object: Object, propertyName: string) {
    registerDecorator({
      target: object.constructor,
      propertyName: propertyName,
      options: validationOptions,
      constraints: [entityClass, property],
      validator: IsUniqueConstraint,
    });
  };
}

// Usage
export class CreateUserDto {
  @IsEmail()
  @IsUnique(User, 'email', { message: 'Email is already taken' })
  email: string;
}
```

### Global Validation Pipe
```typescript
// main.ts
import { ValidationPipe } from '@nestjs/common';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  
  app.useGlobalPipes(
    new ValidationPipe({
      whitelist: true, // Strip properties not in DTO
      forbidNonWhitelisted: true, // Throw error for extra properties
      transform: true, // Automatically transform payloads
      transformOptions: {
        enableImplicitConversion: true, // Convert types automatically
      },
      exceptionFactory: (errors) => {
        const result = errors.map((error) => ({
          property: error.property,
          value: error.value,
          constraints: error.constraints,
        }));
        return new BadRequestException(result);
      },
    }),
  );
  
  await app.listen(3000);
}
```

### File Upload Validation
```typescript
// validators/file-validation.ts
export const imageFileFilter = (req, file, callback) => {
  if (!file.originalname.match(/\.(jpg|jpeg|png|gif)$/)) {
    return callback(new Error('Only image files are allowed!'), false);
  }
  callback(null, true);
};

// dto/upload-file.dto.ts
export class UploadFileDto {
  @IsOptional()
  @IsString()
  description?: string;
}

// Controller usage
@Controller('upload')
export class UploadController {
  @Post('image')
  @UseInterceptors(
    FileInterceptor('file', {
      limits: { fileSize: 5 * 1024 * 1024 }, // 5MB
      fileFilter: imageFileFilter,
    })
  )
  uploadImage(
    @UploadedFile() file: Express.Multer.File,
    @Body() uploadDto: UploadFileDto,
  ) {
    if (!file) {
      throw new BadRequestException('File is required');
    }
    
    return {
      filename: file.filename,
      originalName: file.originalname,
      size: file.size,
      description: uploadDto.description,
    };
  }
}
```

This comprehensive guide covers all essential authentication and security topics for Phase 2, providing production-ready implementations and best practices for securing NestJS applications.
