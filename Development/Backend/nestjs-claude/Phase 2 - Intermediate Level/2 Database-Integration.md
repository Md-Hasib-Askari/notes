# NestJS Phase 2: Database Integration

## 1. TypeORM Integration and Setup

### Installation and Configuration
```bash
# Install TypeORM and database driver
npm install @nestjs/typeorm typeorm pg
npm install -D @types/pg

# For MySQL
npm install mysql2
# For SQLite
npm install sqlite3
```

### Database Module Configuration
```typescript
// app.module.ts
import { TypeOrmModule } from '@nestjs/typeorm';
import { ConfigModule, ConfigService } from '@nestjs/config';

@Module({
  imports: [
    ConfigModule.forRoot({ isGlobal: true }),
    TypeOrmModule.forRootAsync({
      inject: [ConfigService],
      useFactory: (configService: ConfigService) => ({
        type: 'postgres',
        host: configService.get('DATABASE_HOST'),
        port: configService.get('DATABASE_PORT'),
        username: configService.get('DATABASE_USERNAME'),
        password: configService.get('DATABASE_PASSWORD'),
        database: configService.get('DATABASE_NAME'),
        entities: [__dirname + '/**/*.entity{.ts,.js}'],
        synchronize: configService.get('NODE_ENV') !== 'production',
        logging: configService.get('NODE_ENV') === 'development',
        migrations: [__dirname + '/migrations/*{.ts,.js}'],
        migrationsRun: true,
      }),
    }),
  ],
})
export class AppModule {}
```

### Environment Configuration
```bash
# .env
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_USERNAME=postgres
DATABASE_PASSWORD=password
DATABASE_NAME=nestjs_app
NODE_ENV=development
```

### Alternative Configuration Methods
```typescript
// Static configuration
TypeOrmModule.forRoot({
  type: 'postgres',
  host: 'localhost',
  port: 5432,
  username: 'postgres',
  password: 'password',
  database: 'nestjs_app',
  entities: [User, Product, Order],
  synchronize: true,
})

// Multiple databases
TypeOrmModule.forRoot({
  name: 'usersConnection',
  type: 'postgres',
  host: 'localhost',
  port: 5432,
  username: 'postgres',
  password: 'password',
  database: 'users_db',
  entities: [User],
})

TypeOrmModule.forRoot({
  name: 'ordersConnection',
  type: 'mysql',
  host: 'localhost',
  port: 3306,
  username: 'root',
  password: 'password',
  database: 'orders_db',
  entities: [Order],
})
```

## 2. Entity Definition and Relationships

### Basic Entity Definition
```typescript
// entities/user.entity.ts
import { Entity, PrimaryGeneratedColumn, Column, CreateDateColumn, UpdateDateColumn } from 'typeorm';

@Entity('users')
export class User {
  @PrimaryGeneratedColumn()
  id: number;

  @Column({ unique: true })
  email: string;

  @Column()
  firstName: string;

  @Column()
  lastName: string;

  @Column()
  password: string;

  @Column({ default: true })
  isActive: boolean;

  @Column('text', { array: true, default: [] })
  roles: string[];

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;

  @Column({ nullable: true })
  lastLoginAt: Date;
}
```

### Advanced Column Types
```typescript
@Entity('products')
export class Product {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ length: 100 })
  name: string;

  @Column('text')
  description: string;

  @Column('decimal', { precision: 10, scale: 2 })
  price: number;

  @Column('int', { default: 0 })
  stock: number;

  @Column('json', { nullable: true })
  specifications: Record<string, any>;

  @Column('enum', { enum: ['draft', 'published', 'archived'] })
  status: string;

  @Column('simple-array')
  tags: string[];

  @Column('boolean', { default: false })
  featured: boolean;
}
```

### One-to-Many Relationship
```typescript
// entities/user.entity.ts
@Entity('users')
export class User {
  // ...other columns

  @OneToMany(() => Order, order => order.user)
  orders: Order[];

  @OneToMany(() => Post, post => post.author)
  posts: Post[];
}

// entities/order.entity.ts
@Entity('orders')
export class Order {
  @PrimaryGeneratedColumn()
  id: number;

  @Column('decimal', { precision: 10, scale: 2 })
  total: number;

  @Column('enum', { enum: ['pending', 'shipped', 'delivered', 'cancelled'] })
  status: string;

  @ManyToOne(() => User, user => user.orders)
  @JoinColumn({ name: 'userId' })
  user: User;

  @Column()
  userId: number;

  @OneToMany(() => OrderItem, orderItem => orderItem.order, { cascade: true })
  items: OrderItem[];

  @CreateDateColumn()
  createdAt: Date;
}
```

### Many-to-Many Relationship
```typescript
// entities/user.entity.ts
@Entity('users')
export class User {
  // ...other columns

  @ManyToMany(() => Role, role => role.users)
  @JoinTable({
    name: 'user_roles',
    joinColumn: { name: 'userId', referencedColumnName: 'id' },
    inverseJoinColumn: { name: 'roleId', referencedColumnName: 'id' }
  })
  roles: Role[];
}

// entities/role.entity.ts
@Entity('roles')
export class Role {
  @PrimaryGeneratedColumn()
  id: number;

  @Column({ unique: true })
  name: string;

  @Column()
  description: string;

  @ManyToMany(() => User, user => user.roles)
  users: User[];

  @ManyToMany(() => Permission, permission => permission.roles)
  @JoinTable()
  permissions: Permission[];
}
```

### Self-Referencing Relationship
```typescript
@Entity('categories')
export class Category {
  @PrimaryGeneratedColumn()
  id: number;

  @Column()
  name: string;

  @ManyToOne(() => Category, category => category.children)
  @JoinColumn({ name: 'parentId' })
  parent: Category;

  @OneToMany(() => Category, category => category.parent)
  children: Category[];

  @Column({ nullable: true })
  parentId: number;
}
```

## 3. Repository Pattern

### Basic Repository Usage
```typescript
// users/users.service.ts
@Injectable()
export class UsersService {
  constructor(
    @InjectRepository(User)
    private usersRepository: Repository<User>,
  ) {}

  async findAll(): Promise<User[]> {
    return this.usersRepository.find();
  }

  async findOne(id: number): Promise<User> {
    const user = await this.usersRepository.findOne({ where: { id } });
    if (!user) {
      throw new NotFoundException(`User with ID ${id} not found`);
    }
    return user;
  }

  async create(createUserDto: CreateUserDto): Promise<User> {
    const user = this.usersRepository.create(createUserDto);
    return this.usersRepository.save(user);
  }

  async update(id: number, updateUserDto: UpdateUserDto): Promise<User> {
    await this.usersRepository.update(id, updateUserDto);
    return this.findOne(id);
  }

  async remove(id: number): Promise<void> {
    const result = await this.usersRepository.delete(id);
    if (result.affected === 0) {
      throw new NotFoundException(`User with ID ${id} not found`);
    }
  }
}
```

### Custom Repository
```typescript
// repositories/users.repository.ts
@Injectable()
export class UsersRepository extends Repository<User> {
  constructor(
    @InjectDataSource()
    private dataSource: DataSource,
  ) {
    super(User, dataSource.createEntityManager());
  }

  async findByEmail(email: string): Promise<User | null> {
    return this.findOne({ where: { email } });
  }

  async findActiveUsers(): Promise<User[]> {
    return this.find({ where: { isActive: true } });
  }

  async findUsersWithOrders(): Promise<User[]> {
    return this.find({
      relations: ['orders'],
      where: { isActive: true },
    });
  }

  async searchUsers(searchTerm: string): Promise<User[]> {
    return this.createQueryBuilder('user')
      .where('user.firstName ILIKE :searchTerm OR user.lastName ILIKE :searchTerm OR user.email ILIKE :searchTerm', {
        searchTerm: `%${searchTerm}%`,
      })
      .getMany();
  }

  async getUserStats(): Promise<{ total: number; active: number; inactive: number }> {
    const [total, active] = await Promise.all([
      this.count(),
      this.count({ where: { isActive: true } }),
    ]);

    return {
      total,
      active,
      inactive: total - active,
    };
  }
}
```

### Query Builder Examples
```typescript
@Injectable()
export class OrdersService {
  constructor(
    @InjectRepository(Order)
    private ordersRepository: Repository<Order>,
  ) {}

  async findOrdersWithItems(userId: number): Promise<Order[]> {
    return this.ordersRepository
      .createQueryBuilder('order')
      .leftJoinAndSelect('order.items', 'item')
      .leftJoinAndSelect('item.product', 'product')
      .where('order.userId = :userId', { userId })
      .orderBy('order.createdAt', 'DESC')
      .getMany();
  }

  async getOrdersByDateRange(startDate: Date, endDate: Date): Promise<Order[]> {
    return this.ordersRepository
      .createQueryBuilder('order')
      .where('order.createdAt BETWEEN :startDate AND :endDate', {
        startDate,
        endDate,
      })
      .leftJoinAndSelect('order.user', 'user')
      .getMany();
  }

  async getOrderStatistics() {
    return this.ordersRepository
      .createQueryBuilder('order')
      .select([
        'COUNT(*) as totalOrders',
        'SUM(order.total) as totalRevenue',
        'AVG(order.total) as averageOrderValue',
      ])
      .where('order.status = :status', { status: 'delivered' })
      .getRawOne();
  }
}
```

## 4. Database Migrations

### Migration Configuration
```typescript
// ormconfig.ts or data-source.ts
import { DataSource } from 'typeorm';

export const AppDataSource = new DataSource({
  type: 'postgres',
  host: 'localhost',
  port: 5432,
  username: 'postgres',
  password: 'password',
  database: 'nestjs_app',
  synchronize: false,
  logging: false,
  entities: ['src/**/*.entity.ts'],
  migrations: ['src/migrations/*.ts'],
  subscribers: ['src/subscribers/*.ts'],
});
```

### Creating Migrations
```bash
# Generate migration
npm run typeorm migration:generate src/migrations/CreateUserTable

# Create empty migration
npm run typeorm migration:create src/migrations/AddUserRoles

# Run migrations
npm run typeorm migration:run

# Revert migration
npm run typeorm migration:revert
```

### Migration Examples
```typescript
// migrations/1640000000000-CreateUserTable.ts
import { MigrationInterface, QueryRunner, Table } from 'typeorm';

export class CreateUserTable1640000000000 implements MigrationInterface {
  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.createTable(
      new Table({
        name: 'users',
        columns: [
          {
            name: 'id',
            type: 'int',
            isPrimary: true,
            isGenerated: true,
            generationStrategy: 'increment',
          },
          {
            name: 'email',
            type: 'varchar',
            isUnique: true,
          },
          {
            name: 'firstName',
            type: 'varchar',
          },
          {
            name: 'lastName',
            type: 'varchar',
          },
          {
            name: 'password',
            type: 'varchar',
          },
          {
            name: 'isActive',
            type: 'boolean',
            default: true,
          },
          {
            name: 'createdAt',
            type: 'timestamp',
            default: 'CURRENT_TIMESTAMP',
          },
          {
            name: 'updatedAt',
            type: 'timestamp',
            default: 'CURRENT_TIMESTAMP',
            onUpdate: 'CURRENT_TIMESTAMP',
          },
        ],
      }),
      true,
    );
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.dropTable('users');
  }
}

// migrations/1640000000001-AddUserRoles.ts
export class AddUserRoles1640000000001 implements MigrationInterface {
  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.addColumn(
      'users',
      new TableColumn({
        name: 'roles',
        type: 'text',
        isArray: true,
        default: "'{}'",
      }),
    );
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.dropColumn('users', 'roles');
  }
}
```

## 5. Query Optimization Basics

### Efficient Querying
```typescript
@Injectable()
export class UsersService {
  // Use select to fetch only needed columns
  async findUserEmails(): Promise<Pick<User, 'id' | 'email'>[]> {
    return this.usersRepository.find({
      select: ['id', 'email'],
    });
  }

  // Use pagination for large datasets
  async findUsers(page: number, limit: number): Promise<{ users: User[]; total: number }> {
    const [users, total] = await this.usersRepository.findAndCount({
      skip: (page - 1) * limit,
      take: limit,
      order: { createdAt: 'DESC' },
    });

    return { users, total };
  }

  // Eager loading with relations
  async findUsersWithOrders(): Promise<User[]> {
    return this.usersRepository.find({
      relations: ['orders', 'orders.items'],
      where: { isActive: true },
    });
  }

  // Lazy loading alternative
  async findUserWithOrdersLazy(id: number): Promise<User> {
    const user = await this.usersRepository.findOne({ where: { id } });
    if (user) {
      user.orders = await this.ordersRepository.find({ where: { userId: id } });
    }
    return user;
  }
}
```

### Query Builder Optimization
```typescript
@Injectable()
export class ReportsService {
  // Use indexes effectively
  async findRecentActiveUsers(): Promise<User[]> {
    return this.usersRepository
      .createQueryBuilder('user')
      .where('user.isActive = :isActive', { isActive: true })
      .andWhere('user.lastLoginAt > :date', { 
        date: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000) 
      })
      .orderBy('user.lastLoginAt', 'DESC')
      .limit(100)
      .getMany();
  }

  // Aggregate queries
  async getUserOrderSummary(userId: number) {
    return this.ordersRepository
      .createQueryBuilder('order')
      .select([
        'COUNT(order.id) as orderCount',
        'SUM(order.total) as totalSpent',
        'AVG(order.total) as averageOrder',
        'MAX(order.createdAt) as lastOrderDate',
      ])
      .where('order.userId = :userId', { userId })
      .getRawOne();
  }

  // Join optimization
  async findOrdersWithProductInfo(): Promise<any[]> {
    return this.ordersRepository
      .createQueryBuilder('order')
      .innerJoin('order.items', 'item')
      .innerJoin('item.product', 'product')
      .select([
        'order.id',
        'order.total',
        'order.createdAt',
        'product.name',
        'product.price',
        'item.quantity',
      ])
      .where('order.status = :status', { status: 'delivered' })
      .getMany();
  }
}
```

### Database Indexing
```typescript
// Add indexes to entities
@Entity('users')
@Index(['email']) // Single column index
@Index(['firstName', 'lastName']) // Composite index
@Index(['isActive', 'createdAt']) // Multi-column index
export class User {
  @Column({ unique: true })
  @Index() // Explicit index
  email: string;

  @Column()
  firstName: string;

  @Column()
  lastName: string;

  @Column()
  @Index() // Index for frequent queries
  isActive: boolean;
}
```

### Module Integration
```typescript
// users/users.module.ts
@Module({
  imports: [TypeOrmModule.forFeature([User])],
  controllers: [UsersController],
  providers: [UsersService, UsersRepository],
  exports: [UsersService],
})
export class UsersModule {}
```

This comprehensive guide covers all essential database integration topics for Phase 2, providing practical examples and best practices for working with TypeORM in NestJS applications.
