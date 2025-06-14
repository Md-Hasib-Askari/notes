

## 🟣 **16. Logging and Monitoring in NestJS**

Robust logging and monitoring are essential for maintaining observability, identifying issues quickly, and scaling confidently.

---

### ✅ 1. Logging Options

NestJS has a built-in `Logger`, but production apps need more powerful solutions like:

#### 🛠 Built-in Logger

```ts
import { Logger } from '@nestjs/common';

const logger = new Logger('AppService');
logger.log('App started');     // General log
logger.warn('Low memory');     // Warning
logger.error('Crash!', error); // Error with stack
```

Customize globally:

```ts
async function bootstrap() {
  const app = await NestFactory.create(AppModule, {
    logger: ['log', 'warn', 'error'], // control verbosity
  });
}
```

---

### ✅ 2. Using Winston or Pino

More advanced loggers that support:

* Log levels
* JSON format
* File & stream transports
* External log services (e.g., Loggly, Datadog)

#### Example with `nestjs-pino` (Recommended for performance)

```bash
npm install nestjs-pino pino-pretty
```

```ts
// main.ts
import { Logger } from 'nestjs-pino';

const app = await NestFactory.create(AppModule, {
  bufferLogs: true,
});
app.useLogger(app.get(Logger));
```

```ts
// app.module.ts
import { LoggerModule } from 'nestjs-pino';

@Module({
  imports: [
    LoggerModule.forRoot({
      pinoHttp: {
        transport: {
          target: 'pino-pretty',
          options: {
            colorize: true,
            translateTime: true,
          },
        },
      },
    }),
  ],
})
export class AppModule {}
```

---

### ✅ 3. Monitoring with Metrics

#### Prometheus + Grafana (Most popular stack)

* Track CPU, memory, request latency, error rate, etc.
* Use `@willsoto/nestjs-prometheus`

```bash
npm install @willsoto/nestjs-prometheus prom-client
```

```ts
import { PrometheusModule } from '@willsoto/nestjs-prometheus';

@Module({
  imports: [PrometheusModule.register()],
})
export class AppModule {}
```

Then expose a `/metrics` route and view metrics with Prometheus + Grafana.

---

### ✅ 4. Monitoring Errors with Sentry

Catch, log, and analyze uncaught errors and exceptions.

```bash
npm install @ntegral/nestjs-sentry @sentry/node
```

```ts
import { SentryModule } from '@ntegral/nestjs-sentry';

@Module({
  imports: [
    SentryModule.forRoot({
      dsn: 'https://examplePublicKey@o0.ingest.sentry.io/0',
      debug: true,
      environment: 'production',
    }),
  ],
})
export class AppModule {}
```

You can then capture exceptions manually:

```ts
import * as Sentry from '@sentry/node';
Sentry.captureException(error);
```

---

### 🧪 Exercise

✅ **Log all failed requests with timestamps and error stack**

1. Use `nestjs-pino` to set up logging middleware.
2. Capture logs only for `status >= 400`.
3. Save to a rotating file or forward to a log aggregation tool (e.g., Logstash, CloudWatch).

