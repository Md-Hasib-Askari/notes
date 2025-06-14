
## 🟡 **7. Configuration and Environment Variables in NestJS**

Managing environment-specific settings (like DB credentials, API keys, modes) is crucial for real-world apps.

---

### ✅ **1. Installing @nestjs/config**

```bash
npm install @nestjs/config
```

---

### ✅ **2. Basic Setup**

#### Add `.env` in the root:

```
PORT=3000
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASS=password
```

#### Update `AppModule`:

```ts
import { ConfigModule } from '@nestjs/config';

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true, // Makes config available app-wide
    }),
  ],
})
export class AppModule {}
```

---

### ✅ **3. Accessing Environment Variables**

You can use the `ConfigService` anywhere.

#### Example:

```ts
import { ConfigService } from '@nestjs/config';

@Injectable()
export class AppService {
  constructor(private configService: ConfigService) {}

  getPort(): string {
    return this.configService.get<string>('PORT');
  }
}
```

---

### ✅ **4. Validating .env Variables**

#### Install `joi`:

```bash
npm install joi
```

#### Use `validationSchema`:

```ts
import * as Joi from 'joi';

ConfigModule.forRoot({
  isGlobal: true,
  validationSchema: Joi.object({
    PORT: Joi.number().default(3000),
    DB_HOST: Joi.string().required(),
    DB_PORT: Joi.number().required(),
    DB_USER: Joi.string().required(),
    DB_PASS: Joi.string().required(),
  }),
});
```

---

### ✅ **5. Environment-Specific Configs**

Structure:

```
.env
.env.development
.env.production
```

Set the environment before running:

```bash
NODE_ENV=production nest start
```

Update `ConfigModule`:

```ts
ConfigModule.forRoot({
  isGlobal: true,
  envFilePath: `.env.${process.env.NODE_ENV || 'development'}`,
});
```

---

### ✅ **6. Custom Configuration Files**

You can separate config logic for readability:

```ts
// config/database.config.ts
export default () => ({
  database: {
    host: process.env.DB_HOST,
    port: parseInt(process.env.DB_PORT),
    user: process.env.DB_USER,
    pass: process.env.DB_PASS,
  },
});
```

Use in `AppModule`:

```ts
ConfigModule.forRoot({
  isGlobal: true,
  load: [databaseConfig],
});
```

---

### 💪 Exercise

✅ **Set up different configurations for dev and prod:**

1. Create `.env.development` and `.env.production`.
2. Use `NODE_ENV` to load the correct config.
3. Validate required keys with `joi`.
