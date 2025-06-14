
## 🔵 **13. Task Scheduling and Queues in NestJS**

Efficient background processing and scheduled tasks are critical for building scalable systems. NestJS supports both via `@nestjs/schedule` for cron jobs and libraries like Bull for job queues.

---

### ✅ 1. **Task Scheduling with `@nestjs/schedule`**

#### 📦 Install:

```bash
npm install @nestjs/schedule
```

#### 🧠 Setup:

In `AppModule`:

```ts
import { ScheduleModule } from '@nestjs/schedule';

@Module({
  imports: [ScheduleModule.forRoot()],
})
export class AppModule {}
```

#### ⏱ Example:

```ts
import { Injectable } from '@nestjs/common';
import { Cron, Interval, Timeout } from '@nestjs/schedule';

@Injectable()
export class TasksService {
  @Cron('45 * * * * *')
  handleCron() {
    console.log('Cron job every minute at 45 seconds');
  }

  @Interval(10000)
  handleInterval() {
    console.log('Runs every 10 seconds');
  }

  @Timeout(5000)
  handleTimeout() {
    console.log('Runs once after 5 seconds');
  }
}
```

---

### ✅ 2. **Queues with Bull**

Great for offloading time-consuming tasks like email or image processing.

#### 📦 Install Bull & Redis:

```bash
npm install @nestjs/bull bull ioredis
```

> Redis must be installed and running on your system.

---

### 🔧 Setup

```ts
import { BullModule } from '@nestjs/bull';

@Module({
  imports: [
    BullModule.forRoot({
      redis: {
        host: 'localhost',
        port: 6379,
      },
    }),
    BullModule.registerQueue({
      name: 'email',
    }),
  ],
})
export class AppModule {}
```

---

### 📨 Adding Jobs to Queue

```ts
import { InjectQueue } from '@nestjs/bull';
import { Queue } from 'bull';

@Injectable()
export class EmailService {
  constructor(@InjectQueue('email') private emailQueue: Queue) {}

  async sendEmail(emailData: any) {
    await this.emailQueue.add('send', emailData);
  }
}
```

---

### 🛠️ Queue Processor

```ts
import { Process, Processor } from '@nestjs/bull';

@Processor('email')
export class EmailProcessor {
  @Process('send')
  handleSend(job: Job<any>) {
    console.log('Sending email:', job.data);
    // simulate sending email
  }
}
```

---

### 💪 Exercise

✅ **Create a job queue for sending emails after registration**

1. Use Bull to register an `email` queue.
2. Add email data to the queue after user registration.
3. Process the queue to send emails in the background.
4. (Bonus) Use `BullBoard` or `Arena` to monitor the queue visually.

