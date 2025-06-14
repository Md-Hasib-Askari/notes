
### ✅ **5. Data Persistence with TypeORM / Prisma / Mongoose**

---

### 🔹 **Overview**

NestJS supports three main ORMs:

* **TypeORM** – best for SQL databases (PostgreSQL, MySQL).
* **Prisma** – type-safe, developer-friendly ORM.
* **Mongoose** – MongoDB ODM.

We'll use **TypeORM** + **PostgreSQL** as an example.

---

### ✅ **Step-by-Step Setup with TypeORM and PostgreSQL**

#### 1. **Install Dependencies**

```bash
npm install --save @nestjs/typeorm typeorm pg
```

#### 2. **Configure TypeORM**

In `app.module.ts`:

```ts
import { TypeOrmModule } from '@nestjs/typeorm';

@Module({
  imports: [
    TypeOrmModule.forRoot({
      type: 'postgres',
      host: 'localhost',
      port: 5432,
      username: 'postgres',
      password: 'password',
      database: 'todo_db',
      autoLoadEntities: true,
      synchronize: true, // disable in production
    }),
  ],
})
export class AppModule {}
```

---

### ✅ **Entities and Repositories**

#### Create an Entity

```ts
import { Entity, Column, PrimaryGeneratedColumn } from 'typeorm';

@Entity()
export class Todo {
  @PrimaryGeneratedColumn()
  id: number;

  @Column()
  title: string;

  @Column({ default: false })
  completed: boolean;
}
```

#### Register the Entity

```ts
TypeOrmModule.forFeature([Todo])
```

---

### ✅ **Create a Service Using Repository**

```ts
import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Todo } from './todo.entity';

@Injectable()
export class TodoService {
  constructor(
    @InjectRepository(Todo)
    private todoRepo: Repository<Todo>,
  ) {}

  create(title: string) {
    const todo = this.todoRepo.create({ title });
    return this.todoRepo.save(todo);
  }

  findAll() {
    return this.todoRepo.find();
  }

  markDone(id: number) {
    return this.todoRepo.update(id, { completed: true });
  }
}
```

---

### ✅ **DTOs and Validation**

#### Install Validation Packages

```bash
npm install class-validator class-transformer
```

#### Create DTO

```ts
import { IsString, IsNotEmpty } from 'class-validator';

export class CreateTodoDto {
  @IsString()
  @IsNotEmpty()
  title: string;
}
```

#### Use DTO in Controller

```ts
import { Body, Controller, Post, Get } from '@nestjs/common';
import { CreateTodoDto } from './dto/create-todo.dto';
import { TodoService } from './todo.service';

@Controller('todos')
export class TodoController {
  constructor(private todoService: TodoService) {}

  @Post()
  create(@Body() dto: CreateTodoDto) {
    return this.todoService.create(dto.title);
  }

  @Get()
  findAll() {
    return this.todoService.findAll();
  }
}
```

---

### ✅ **Migrations (Optional, for TypeORM CLI)**

#### Install CLI

```bash
npm install -D typeorm ts-node
```

#### Add `ormconfig.js` or extend config in `data-source.ts`.

Then run:

```bash
npx typeorm migration:create src/migrations/Init
npx typeorm migration:run
```

---

### ✅ **Exercise**

🔧 **Build a Todo App:**

* Connect to PostgreSQL.
* Create `Todo` entity.
* Add DTO validation.
* Create endpoints: `GET /todos`, `POST /todos`, `PUT /todos/:id/done`.

