# NestJS Phase 3: Advanced Database Patterns

## 1. Complex Database Relationships

### Many-to-Many with Join Tables and Additional Properties
```typescript
// entities/user.entity.ts
@Entity('users')
export class User {
  @PrimaryGeneratedColumn()
  id: number;

  @Column()
  email: string;

  @Column()
  firstName: string;

  @Column()
  lastName: string;

  @OneToMany(() => UserRole, userRole => userRole.user)
  userRoles: UserRole[];

  @OneToMany(() => ProjectMember, projectMember => projectMember.user)
  projectMemberships: ProjectMember[];

  @OneToMany(() => Order, order => order.customer)
  orders: Order[];

  @OneToMany(() => Review, review => review.author)
  reviews: Review[];
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

  @OneToMany(() => UserRole, userRole => userRole.role)
  userRoles: UserRole[];

  @OneToMany(() => Permission, permission => permission.role)
  permissions: Permission[];
}

// entities/user-role.entity.ts (Join table with additional properties)
@Entity('user_roles')
export class UserRole {
  @PrimaryGeneratedColumn()
  id: number;

  @ManyToOne(() => User, user => user.userRoles)
  @JoinColumn({ name: 'userId' })
  user: User;

  @Column()
  userId: number;

  @ManyToOne(() => Role, role => role.userRoles)
  @JoinColumn({ name: 'roleId' })
  role: Role;

  @Column()
  roleId: number;

  @Column({ type: 'timestamp', default: () => 'CURRENT_TIMESTAMP' })
  assignedAt: Date;

  @Column({ type: 'timestamp', nullable: true })
  expiresAt: Date;

  @Column({ default: true })
  isActive: boolean;

  @ManyToOne(() => User)
  @JoinColumn({ name: 'assignedBy' })
  assignedBy: User;
}
```

### Polymorphic Relationships
```typescript
// entities/comment.entity.ts
@Entity('comments')
export class Comment {
  @PrimaryGeneratedColumn()
  id: number;

  @Column()
  content: string;

  @Column()
  commentableType: string; // 'Post' | 'Product' | 'User'

  @Column()
  commentableId: number;

  @ManyToOne(() => User)
  @JoinColumn({ name: 'authorId' })
  author: User;

  @Column()
  authorId: number;

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;

  // Virtual properties for polymorphic relations
  post?: Post;
  product?: Product;
  user?: User;
}

// Custom repository to handle polymorphic relationships
@Injectable()
export class CommentsRepository extends Repository<Comment> {
  constructor(
    @InjectDataSource() private dataSource: DataSource,
    @InjectRepository(Post) private postRepository: Repository<Post>,
    @InjectRepository(Product) private productRepository: Repository<Product>,
    @InjectRepository(User) private userRepository: Repository<User>,
  ) {
    super(Comment, dataSource.createEntityManager());
  }

  async findWithCommentable(id: number): Promise<Comment | null> {
    const comment = await this.findOne({ where: { id }, relations: ['author'] });
    
    if (!comment) return null;

    // Load the polymorphic relation
    switch (comment.commentableType) {
      case 'Post':
        comment.post = await this.postRepository.findOne({ where: { id: comment.commentableId } });
        break;
      case 'Product':
        comment.product = await this.productRepository.findOne({ where: { id: comment.commentableId } });
        break;
      case 'User':
        comment.user = await this.userRepository.findOne({ where: { id: comment.commentableId } });
        break;
    }

    return comment;
  }

  async findByCommentable(type: string, id: number): Promise<Comment[]> {
    return this.find({
      where: { commentableType: type, commentableId: id },
      relations: ['author'],
      order: { createdAt: 'DESC' },
    });
  }
}
```

### Self-Referencing with Tree Structures
```typescript
// entities/category.entity.ts
@Entity('categories')
export class Category {
  @PrimaryGeneratedColumn()
  id: number;

  @Column()
  name: string;

  @Column()
  slug: string;

  @Column({ nullable: true })
  description: string;

  @TreeParent()
  parent: Category;

  @TreeChildren()
  children: Category[];

  @Column({ nullable: true })
  parentId: number;

  @Column({ default: 0 })
  level: number;

  @Column({ nullable: true })
  path: string; // Store full path like "1.2.5"

  @OneToMany(() => Product, product => product.category)
  products: Product[];

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;

  @BeforeInsert()
  @BeforeUpdate()
  async updatePath() {
    if (this.parent) {
      this.level = this.parent.level + 1;
      this.path = this.parent.path ? `${this.parent.path}.${this.id}` : `${this.id}`;
    } else {
      this.level = 0;
      this.path = `${this.id}`;
    }
  }
}

// Category service with tree operations
@Injectable()
export class CategoriesService {
  constructor(
    @InjectRepository(Category)
    private categoriesRepository: TreeRepository<Category>,
  ) {}

  async findTree(): Promise<Category[]> {
    return this.categoriesRepository.findTrees();
  }

  async findDescendants(category: Category): Promise<Category[]> {
    return this.categoriesRepository.findDescendants(category);
  }

  async findAncestors(category: Category): Promise<Category[]> {
    return this.categoriesRepository.findAncestors(category);
  }

  async findWithPath(path: string): Promise<Category[]> {
    return this.categoriesRepository
      .createQueryBuilder('category')
      .where('category.path LIKE :path', { path: `${path}%` })
      .orderBy('category.level', 'ASC')
      .getMany();
  }

  async moveCategory(categoryId: number, newParentId: number | null): Promise<void> {
    const category = await this.categoriesRepository.findOne({ where: { id: categoryId } });
    const newParent = newParentId ? await this.categoriesRepository.findOne({ where: { id: newParentId } }) : null;

    // Update all descendants
    const descendants = await this.findDescendants(category);
    
    await this.categoriesRepository.manager.transaction(async manager => {
      // Update parent
      category.parent = newParent;
      category.parentId = newParentId;
      await manager.save(category);

      // Update all descendants' paths and levels
      for (const descendant of descendants) {
        await descendant.updatePath();
        await manager.save(descendant);
      }
    });
  }
}
```

### Complex Aggregations with Window Functions
```typescript
// Complex analytics service
@Injectable()
export class AnalyticsService {
  constructor(
    @InjectDataSource() private dataSource: DataSource,
  ) {}

  async getSalesAnalytics(startDate: Date, endDate: Date) {
    return this.dataSource.query(`
      WITH monthly_sales AS (
        SELECT 
          DATE_TRUNC('month', o.created_at) as month,
          SUM(o.total) as monthly_total,
          COUNT(*) as order_count,
          COUNT(DISTINCT o.customer_id) as unique_customers
        FROM orders o
        WHERE o.created_at BETWEEN $1 AND $2
          AND o.status = 'completed'
        GROUP BY DATE_TRUNC('month', o.created_at)
      ),
      sales_with_growth AS (
        SELECT 
          month,
          monthly_total,
          order_count,
          unique_customers,
          LAG(monthly_total) OVER (ORDER BY month) as prev_month_total,
          (monthly_total - LAG(monthly_total) OVER (ORDER BY month)) / 
            NULLIF(LAG(monthly_total) OVER (ORDER BY month), 0) * 100 as growth_rate,
          SUM(monthly_total) OVER (ORDER BY month) as cumulative_total,
          ROW_NUMBER() OVER (ORDER BY monthly_total DESC) as rank_by_revenue
        FROM monthly_sales
      )
      SELECT 
        month,
        monthly_total,
        order_count,
        unique_customers,
        COALESCE(growth_rate, 0) as growth_rate,
        cumulative_total,
        rank_by_revenue
      FROM sales_with_growth
      ORDER BY month;
    `, [startDate, endDate]);
  }

  async getCustomerLifetimeValue() {
    return this.dataSource.query(`
      WITH customer_metrics AS (
        SELECT 
          c.id,
          c.email,
          c.created_at as customer_since,
          COUNT(o.id) as total_orders,
          SUM(o.total) as lifetime_value,
          AVG(o.total) as avg_order_value,
          MAX(o.created_at) as last_order_date,
          EXTRACT(DAYS FROM NOW() - MAX(o.created_at)) as days_since_last_order,
          EXTRACT(DAYS FROM MAX(o.created_at) - MIN(o.created_at)) as customer_lifespan_days
        FROM customers c
        LEFT JOIN orders o ON c.id = o.customer_id AND o.status = 'completed'
        GROUP BY c.id, c.email, c.created_at
      ),
      customer_segments AS (
        SELECT 
          *,
          CASE 
            WHEN lifetime_value >= 1000 AND total_orders >= 5 THEN 'VIP'
            WHEN lifetime_value >= 500 AND total_orders >= 3 THEN 'High Value'
            WHEN lifetime_value >= 100 AND total_orders >= 2 THEN 'Regular'
            WHEN total_orders = 1 THEN 'One-time'
            ELSE 'Inactive'
          END as customer_segment,
          NTILE(10) OVER (ORDER BY lifetime_value DESC) as value_decile
        FROM customer_metrics
      )
      SELECT 
        customer_segment,
        COUNT(*) as customer_count,
        AVG(lifetime_value) as avg_lifetime_value,
        AVG(total_orders) as avg_orders,
        AVG(avg_order_value) as avg_order_value,
        AVG(customer_lifespan_days) as avg_lifespan_days
      FROM customer_segments
      GROUP BY customer_segment
      ORDER BY avg_lifetime_value DESC;
    `);
  }
}
```

## 2. Query Optimization and Indexing

### Strategic Index Design
```typescript
// entities/product.entity.ts
@Entity('products')
@Index(['status', 'createdAt']) // Composite index for filtering and sorting
@Index(['categoryId', 'featured']) // For category pages with featured items
@Index(['price', 'status']) // For price range queries
@Index('IDX_PRODUCT_SEARCH', ['name', 'description']) // Full-text search
export class Product {
  @PrimaryGeneratedColumn()
  id: number;

  @Column()
  @Index() // Frequently searched
  name: string;

  @Column('text')
  description: string;

  @Column('decimal', { precision: 10, scale: 2 })
  price: number;

  @Column()
  @Index() // Foreign key index
  categoryId: number;

  @Column({ default: 'active' })
  status: 'active' | 'inactive' | 'discontinued';

  @Column({ default: false })
  featured: boolean;

  @Column({ default: 0 })
  @Index() // Frequently used for sorting
  viewCount: number;

  @Column({ default: 0 })
  stockQuantity: number;

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;

  // Virtual calculated fields
  @AfterLoad()
  calculatePopularityScore() {
    // Complex calculation that might need optimization
    this.popularityScore = this.viewCount * 0.3 + (this.orders?.length || 0) * 0.7;
  }

  popularityScore?: number;

  @ManyToOne(() => Category)
  @JoinColumn({ name: 'categoryId' })
  category: Category;

  @OneToMany(() => OrderItem, orderItem => orderItem.product)
  orders: OrderItem[];
}
```

### Advanced Query Optimization
```typescript
@Injectable()
export class ProductsService {
  constructor(
    @InjectRepository(Product)
    private productsRepository: Repository<Product>,
    @InjectDataSource()
    private dataSource: DataSource,
  ) {}

  // Optimized query with proper indexing
  async findProductsOptimized(filters: ProductFilters): Promise<PaginatedResult<Product>> {
    const queryBuilder = this.productsRepository.createQueryBuilder('product');
    
    // Use index-friendly conditions
    queryBuilder.where('product.status = :status', { status: 'active' });

    if (filters.categoryId) {
      // This uses the categoryId index
      queryBuilder.andWhere('product.categoryId = :categoryId', { categoryId: filters.categoryId });
    }

    if (filters.priceMin || filters.priceMax) {
      // This uses the price index
      if (filters.priceMin) {
        queryBuilder.andWhere('product.price >= :priceMin', { priceMin: filters.priceMin });
      }
      if (filters.priceMax) {
        queryBuilder.andWhere('product.price <= :priceMax', { priceMax: filters.priceMax });
      }
    }

    if (filters.featured) {
      // This uses the composite index categoryId + featured
      queryBuilder.andWhere('product.featured = :featured', { featured: true });
    }

    if (filters.search) {
      // Use full-text search for better performance
      queryBuilder.andWhere(
        `to_tsvector('english', product.name || ' ' || product.description) @@ plainto_tsquery('english', :search)`,
        { search: filters.search }
      );
    }

    // Use the composite index for sorting
    queryBuilder.orderBy('product.createdAt', 'DESC');

    // Efficient pagination
    const offset = (filters.page - 1) * filters.limit;
    queryBuilder.skip(offset).take(filters.limit);

    const [products, total] = await queryBuilder.getManyAndCount();

    return {
      data: products,
      total,
      page: filters.page,
      limit: filters.limit,
      totalPages: Math.ceil(total / filters.limit),
    };
  }

  // Query with materialized aggregations
  async getProductStatistics(): Promise<any> {
    return this.dataSource.query(`
      SELECT 
        p.category_id,
        c.name as category_name,
        COUNT(*) as product_count,
        AVG(p.price) as avg_price,
        MIN(p.price) as min_price,
        MAX(p.price) as max_price,
        SUM(p.stock_quantity) as total_stock,
        COUNT(CASE WHEN p.featured THEN 1 END) as featured_count
      FROM products p
      INNER JOIN categories c ON p.category_id = c.id
      WHERE p.status = 'active'
      GROUP BY p.category_id, c.name
      ORDER BY product_count DESC;
    `);
  }

  // Bulk operations for performance
  async updateViewCounts(productViews: Array<{ productId: number; views: number }>): Promise<void> {
    const values = productViews.map(pv => `(${pv.productId}, ${pv.views})`).join(',');
    
    await this.dataSource.query(`
      UPDATE products 
      SET view_count = view_count + v.views,
          updated_at = NOW()
      FROM (VALUES ${values}) AS v(id, views)
      WHERE products.id = v.id;
    `);
  }

  // Query with proper JOIN optimization
  async findProductsWithSalesData(limit: number = 50): Promise<any[]> {
    return this.dataSource.query(`
      SELECT 
        p.id,
        p.name,
        p.price,
        COALESCE(sales.total_sold, 0) as total_sold,
        COALESCE(sales.revenue, 0) as revenue,
        COALESCE(reviews.avg_rating, 0) as avg_rating,
        COALESCE(reviews.review_count, 0) as review_count
      FROM products p
      LEFT JOIN (
        SELECT 
          oi.product_id,
          SUM(oi.quantity) as total_sold,
          SUM(oi.quantity * oi.price) as revenue
        FROM order_items oi
        INNER JOIN orders o ON oi.order_id = o.id
        WHERE o.status = 'completed'
        GROUP BY oi.product_id
      ) sales ON p.id = sales.product_id
      LEFT JOIN (
        SELECT 
          product_id,
          AVG(rating) as avg_rating,
          COUNT(*) as review_count
        FROM reviews
        GROUP BY product_id
      ) reviews ON p.id = reviews.product_id
      WHERE p.status = 'active'
      ORDER BY COALESCE(sales.revenue, 0) DESC
      LIMIT $1;
    `, [limit]);
  }
}
```

### Database Performance Monitoring
```typescript
@Injectable()
export class DatabaseMonitoringService {
  constructor(@InjectDataSource() private dataSource: DataSource) {}

  async getSlowQueries(limit: number = 10): Promise<any[]> {
    // PostgreSQL specific - monitor slow queries
    return this.dataSource.query(`
      SELECT 
        query,
        calls,
        total_time,
        mean_time,
        rows,
        100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
      FROM pg_stat_statements
      WHERE query NOT LIKE '%pg_stat_statements%'
      ORDER BY total_time DESC
      LIMIT $1;
    `, [limit]);
  }

  async getIndexUsage(): Promise<any[]> {
    return this.dataSource.query(`
      SELECT 
        schemaname,
        tablename,
        indexname,
        idx_scan as index_scans,
        idx_tup_read as tuples_read,
        idx_tup_fetch as tuples_fetched
      FROM pg_stat_user_indexes
      ORDER BY idx_scan DESC;
    `);
  }

  async getTableSizes(): Promise<any[]> {
    return this.dataSource.query(`
      SELECT 
        schemaname,
        tablename,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
        pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
      FROM pg_tables
      WHERE schemaname = 'public'
      ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
    `);
  }
}
```

## 3. Database Transactions

### Advanced Transaction Management
```typescript
@Injectable()
export class OrdersService {
  constructor(
    @InjectRepository(Order) private ordersRepository: Repository<Order>,
    @InjectRepository(OrderItem) private orderItemsRepository: Repository<OrderItem>,
    @InjectRepository(Product) private productsRepository: Repository<Product>,
    @InjectRepository(User) private usersRepository: Repository<User>,
    @InjectDataSource() private dataSource: DataSource,
  ) {}

  // Complex transaction with multiple entities
  async createOrderWithInventoryUpdate(createOrderDto: CreateOrderDto): Promise<Order> {
    return this.dataSource.transaction(async manager => {
      // 1. Validate user exists and has sufficient balance
      const user = await manager.findOne(User, { 
        where: { id: createOrderDto.customerId },
        lock: { mode: 'pessimistic_write' } // Lock user record
      });

      if (!user) {
        throw new NotFoundException('User not found');
      }

      if (user.balance < createOrderDto.total) {
        throw new BadRequestException('Insufficient balance');
      }

      // 2. Check and reserve inventory
      const orderItems: OrderItem[] = [];
      let calculatedTotal = 0;

      for (const item of createOrderDto.items) {
        const product = await manager.findOne(Product, {
          where: { id: item.productId },
          lock: { mode: 'pessimistic_write' }
        });

        if (!product) {
          throw new NotFoundException(`Product ${item.productId} not found`);
        }

        if (product.stockQuantity < item.quantity) {
          throw new BadRequestException(`Insufficient stock for product ${product.name}`);
        }

        // Update inventory
        product.stockQuantity -= item.quantity;
        await manager.save(product);

        // Create order item
        const orderItem = manager.create(OrderItem, {
          productId: item.productId,
          quantity: item.quantity,
          price: product.price,
          product,
        });
        orderItems.push(orderItem);
        calculatedTotal += product.price * item.quantity;
      }

      // 3. Validate total
      if (Math.abs(calculatedTotal - createOrderDto.total) > 0.01) {
        throw new BadRequestException('Order total mismatch');
      }

      // 4. Create order
      const order = manager.create(Order, {
        customerId: createOrderDto.customerId,
        total: calculatedTotal,
        status: 'pending',
        customer: user,
      });
      const savedOrder = await manager.save(order);

      // 5. Save order items
      for (const orderItem of orderItems) {
        orderItem.orderId = savedOrder.id;
        orderItem.order = savedOrder;
      }
      await manager.save(orderItems);

      // 6. Update user balance
      user.balance -= calculatedTotal;
      await manager.save(user);

      // 7. Create audit log
      await manager.save(AuditLog, {
        entityType: 'Order',
        entityId: savedOrder.id,
        action: 'CREATE',
        userId: createOrderDto.customerId,
        details: { total: calculatedTotal, itemCount: orderItems.length },
      });

      return savedOrder;
    });
  }

  // Transaction with custom isolation level
  async processRefund(orderId: number, refundAmount: number): Promise<void> {
    await this.dataSource.transaction('SERIALIZABLE', async manager => {
      const order = await manager.findOne(Order, {
        where: { id: orderId },
        relations: ['customer'],
        lock: { mode: 'pessimistic_write' }
      });

      if (!order) {
        throw new NotFoundException('Order not found');
      }

      if (order.status !== 'completed') {
        throw new BadRequestException('Cannot refund non-completed order');
      }

      // Check if refund amount is valid
      const existingRefunds = await manager.sum(Refund, 'amount', { orderId });
      const totalRefunded = existingRefunds || 0;
      
      if (totalRefunded + refundAmount > order.total) {
        throw new BadRequestException('Refund amount exceeds order total');
      }

      // Create refund record
      const refund = manager.create(Refund, {
        orderId,
        amount: refundAmount,
        status: 'pending',
        createdAt: new Date(),
      });
      await manager.save(refund);

      // Update customer balance
      order.customer.balance += refundAmount;
      await manager.save(order.customer);

      // If fully refunded, update order status
      if (totalRefunded + refundAmount >= order.total) {
        order.status = 'refunded';
        await manager.save(order);
      }

      // Restore inventory if needed
      const orderItems = await manager.find(OrderItem, { where: { orderId } });
      for (const item of orderItems) {
        const product = await manager.findOne(Product, { where: { id: item.productId } });
        if (product) {
          product.stockQuantity += item.quantity;
          await manager.save(product);
        }
      }
    });
  }

  // Distributed transaction across multiple databases
  async syncOrderToAnalytics(orderId: number): Promise<void> {
    // This would typically use a distributed transaction coordinator
    const queryRunner = this.dataSource.createQueryRunner();
    await queryRunner.connect();
    await queryRunner.startTransaction();

    try {
      // Main database operations
      const order = await queryRunner.manager.findOne(Order, {
        where: { id: orderId },
        relations: ['items', 'customer']
      });

      // Update analytics database (could be a different connection)
      await this.analyticsDataSource.transaction(async analyticsManager => {
        await analyticsManager.save(AnalyticsOrder, {
          orderId: order.id,
          customerId: order.customerId,
          total: order.total,
          itemCount: order.items.length,
          processedAt: new Date(),
        });

        for (const item of order.items) {
          await analyticsManager.save(AnalyticsOrderItem, {
            orderId: order.id,
            productId: item.productId,
            quantity: item.quantity,
            revenue: item.price * item.quantity,
          });
        }
      });

      await queryRunner.commitTransaction();
    } catch (error) {
      await queryRunner.rollbackTransaction();
      throw error;
    } finally {
      await queryRunner.release();
    }
  }
}
```

### Transaction Decorators and Interceptors
```typescript
// decorators/transactional.decorator.ts
export const Transactional = (isolationLevel?: IsolationLevel) =>
  createDecorator('transactional', { isolationLevel });

// interceptors/transaction.interceptor.ts
@Injectable()
export class TransactionInterceptor implements NestInterceptor {
  constructor(@InjectDataSource() private dataSource: DataSource) {}

  intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
    const transactionalMetadata = this.reflector.get('transactional', context.getHandler());
    
    if (!transactionalMetadata) {
      return next.handle();
    }

    return from(
      this.dataSource.transaction(
        transactionalMetadata.isolationLevel,
        async (manager) => {
          // Inject transaction manager into request context
          const request = context.switchToHttp().getRequest();
          request.transactionManager = manager;
          
          return next.handle().toPromise();
        }
      )
    );
  }
}

// Usage in service
@Injectable()
export class UsersService {
  @Transactional('READ_COMMITTED')
  async updateUserProfile(userId: number, updateData: UpdateUserDto): Promise<User> {
    // This method will automatically run in a transaction
    const user = await this.usersRepository.findOne({ where: { id: userId } });
    Object.assign(user, updateData);
    return this.usersRepository.save(user);
  }
}
```

## 4. Connection Pooling

### Advanced Connection Pool Configuration
```typescript
// database/database.config.ts
export const databaseConfig = {
  type: 'postgres' as const,
  host: process.env.DATABASE_HOST,
  port: parseInt(process.env.DATABASE_PORT, 10),
  username: process.env.DATABASE_USERNAME,
  password: process.env.DATABASE_PASSWORD,
  database: process.env.DATABASE_NAME,
  
  // Connection pool settings
  extra: {
    // Connection pool configuration
    max: 20, // Maximum number of connections in the pool
    min: 5,  // Minimum number of connections in the pool
    idle: 10000, // Close connections after 10 seconds of inactivity
    acquire: 60000, // Maximum time to try getting connection
    evict: 1000, // Run eviction every second
    
    // Connection validation
    testOnBorrow: true,
    validationQuery: 'SELECT 1',
    
    // Advanced PostgreSQL settings
    application_name: 'nestjs-app',
    statement_timeout: 30000, // 30 seconds
    idle_in_transaction_session_timeout: 60000, // 1 minute
    
    // SSL configuration
    ssl: process.env.NODE_ENV === 'production' ? {
      rejectUnauthorized: false,
      ca: fs.readFileSync(process.env.SSL_CA_PATH).toString(),
      cert: fs.readFileSync(process.env.SSL_CERT_PATH).toString(),
      key: fs.readFileSync(process.env.SSL_KEY_PATH).toString(),
    } : false,
  },
  
  // Query optimization
  cache: {
    type: 'redis',
    options: {
      host: process.env.REDIS_HOST,
      port: parseInt(process.env.REDIS_PORT, 10),
    },
    duration: 300000, // 5 minutes
  },
  
  // Logging for performance monitoring
  logging: process.env.NODE_ENV === 'development' ? 'all' : ['error', 'warn'],
  logger: 'advanced-console',
  
  // Connection retry
  retryAttempts: 3,
  retryDelay: 3000,
};
```

### Connection Pool Monitoring Service
```typescript
@Injectable()
export class ConnectionPoolService {
  constructor(@InjectDataSource() private dataSource: DataSource) {}

  async getPoolStats(): Promise<any> {
    const driver = this.dataSource.driver as any;
    
    if (driver.master && driver.master.pool) {
      const pool = driver.master.pool;
      
      return {
        totalConnections: pool.totalCount,
        idleConnections: pool.idleCount,
        waitingRequests: pool.waitingCount,
        maxConnections: pool.options.max,
        minConnections: pool.options.min,
        acquiredConnections: pool.totalCount - pool.idleCount,
        poolUtilization: ((pool.totalCount - pool.idleCount) / pool.options.max * 100).toFixed(2) + '%',
      };
    }
    
    return { message: 'Pool stats not available' };
  }

  async healthCheck(): Promise<boolean> {
    try {
      await this.dataSource.query('SELECT 1');
      return true;
    } catch (error) {
      console.error('Database health check failed:', error);
      return false;
    }
  }

  async getActiveConnections(): Promise<any[]> {
    return this.dataSource.query(`
      SELECT 
        pid,
        usename,
        application_name,
        client_addr,
        state,
        query_start,
        state_change,
        query
      FROM pg_stat_activity
      WHERE state = 'active'
        AND pid != pg_backend_pid()
      ORDER BY query_start;
    `);
  }

  async killLongRunningQueries(timeoutMinutes: number = 30): Promise<void> {
    const longRunningQueries = await this.dataSource.query(`
      SELECT pid, query_start, query
      FROM pg_stat_activity
      WHERE state = 'active'
        AND query_start < NOW() - INTERVAL '${timeoutMinutes} minutes'
        AND pid != pg_backend_pid()
        AND query NOT LIKE '%pg_stat_activity%';
    `);

    for (const query of longRunningQueries) {
      console.warn(`Killing long-running query (PID: ${query.pid}):`, query.query);
      await this.dataSource.query(`SELECT pg_terminate_backend(${query.pid})`);
    }
  }
}
```

### Read/Write Splitting
```typescript
// database/database.module.ts
@Module({
  imports: [
    TypeOrmModule.forRootAsync({
      name: 'master',
      useFactory: () => ({
        ...databaseConfig,
        name: 'master',
        // Master database for writes
        replication: {
          master: {
            host: process.env.DB_MASTER_HOST,
            port: parseInt(process.env.DB_MASTER_PORT, 10),
            username: process.env.DB_MASTER_USERNAME,
            password: process.env.DB_MASTER_PASSWORD,
            database: process.env.DB_MASTER_NAME,
          },
          slaves: [
            {
              host: process.env.DB_SLAVE1_HOST,
              port: parseInt(process.env.DB_SLAVE1_PORT, 10),
              username: process.env.DB_SLAVE1_USERNAME,
              password: process.env.DB_SLAVE1_PASSWORD,
              database: process.env.DB_SLAVE1_NAME,
            },
            {
              host: process.env.DB_SLAVE2_HOST,
              port: parseInt(process.env.DB_SLAVE2_PORT, 10),
              username: process.env.DB_SLAVE2_USERNAME,
              password: process.env.DB_SLAVE2_PASSWORD,
              database: process.env.DB_SLAVE2_NAME,
            },
          ],
        },
      }),
    }),
  ],
})
export class DatabaseModule {}

// Service with read/write splitting
@Injectable()
export class UsersService {
  constructor(
    @InjectRepository(User, 'master')
    private usersRepository: Repository<User>,
  ) {}

  // Write operations use master
  async createUser(createUserDto: CreateUserDto): Promise<User> {
    const user = this.usersRepository.create(createUserDto);
    return this.usersRepository.save(user);
  }

  // Read operations can use slaves
  async findUsers(filters: any): Promise<User[]> {
    return this.usersRepository.find({
      where: filters,
      // This will use slave connections for read operations
    });
  }

  // Force master for critical reads
  async findUserForAuth(email: string): Promise<User> {
    return this.usersRepository
      .createQueryBuilder('user')
      .where('user.email = :email', { email })
      .useTransaction(true) // Force master database
      .getOne();
  }
}
```

## 5. Multiple Database Connections

### Multi-Database Setup
```typescript
// app.module.ts
@Module({
  imports: [
    // Primary application database
    TypeOrmModule.forRoot({
      name: 'default',
      type: 'postgres',
      host: process.env.PRIMARY_DB_HOST,
      port: parseInt(process.env.PRIMARY_DB_PORT, 10),
      username: process.env.PRIMARY_DB_USERNAME,
      password: process.env.PRIMARY_DB_PASSWORD,
      database: process.env.PRIMARY_DB_NAME,
      entities: [User, Order, Product, Category],
      synchronize: false,
      migrations: ['src/migrations/primary/*.ts'],
    }),

    // Analytics database
    TypeOrmModule.forRoot({
      name: 'analytics',
      type: 'postgres',
      host: process.env.ANALYTICS_DB_HOST,
      port: parseInt(process.env.ANALYTICS_DB_PORT, 10),
      username: process.env.ANALYTICS_DB_USERNAME,
      password: process.env.ANALYTICS_DB_PASSWORD,
      database: process.env.ANALYTICS_DB_NAME,
      entities: [AnalyticsEvent, UserAnalytics, SalesAnalytics],
      synchronize: false,
      migrations: ['src/migrations/analytics/*.ts'],
    }),

    // Logging database
    TypeOrmModule.forRoot({
      name: 'logging',
      type: 'postgres',
      host: process.env.LOGGING_DB_HOST,
      port: parseInt(process.env.LOGGING_DB_PORT, 10),
      username: process.env.LOGGING_DB_USERNAME,
      password: process.env.LOGGING_DB_PASSWORD,
      database: process.env.LOGGING_DB_NAME,
      entities: [AuditLog, ErrorLog, ActivityLog],
      synchronize: false,
      migrations: ['src/migrations/logging/*.ts'],
    }),

    // Feature-specific modules
    TypeOrmModule.forFeature([User, Order, Product], 'default'),
    TypeOrmModule.forFeature([AnalyticsEvent, UserAnalytics], 'analytics'),
    TypeOrmModule.forFeature([AuditLog, ErrorLog], 'logging'),
  ],
})
export class AppModule {}
```

### Cross-Database Service
```typescript
@Injectable()
export class CrossDatabaseService {
  constructor(
    // Primary database repositories
    @InjectRepository(User, 'default')
    private usersRepository: Repository<User>,
    @InjectRepository(Order, 'default')
    private ordersRepository: Repository<Order>,

    // Analytics database repositories
    @InjectRepository(UserAnalytics, 'analytics')
    private userAnalyticsRepository: Repository<UserAnalytics>,
    @InjectRepository(SalesAnalytics, 'analytics')
    private salesAnalyticsRepository: Repository<SalesAnalytics>,

    // Logging database repositories
    @InjectRepository(AuditLog, 'logging')
    private auditLogRepository: Repository<AuditLog>,

    // Data sources for raw queries
    @InjectDataSource('default')
    private primaryDataSource: DataSource,
    @InjectDataSource('analytics')
    private analyticsDataSource: DataSource,
    @InjectDataSource('logging')
    private loggingDataSource: DataSource,
  ) {}

  async createUserWithAnalytics(createUserDto: CreateUserDto): Promise<User> {
    // Step 1: Create user in primary database
    const user = await this.usersRepository.save(
      this.usersRepository.create(createUserDto)
    );

    try {
      // Step 2: Create analytics profile
      await this.userAnalyticsRepository.save({
        userId: user.id,
        registrationDate: new Date(),
        source: createUserDto.source || 'direct',
        initialSegment: 'new_user',
      });

      // Step 3: Log the action
      await this.auditLogRepository.save({
        entityType: 'User',
        entityId: user.id,
        action: 'CREATE',
        timestamp: new Date(),
        details: { source: createUserDto.source },
      });

      return user;
    } catch (error) {
      // If analytics or logging fails, we might want to compensate
      console.error('Failed to create user analytics/logs:', error);
      // Could implement compensation logic here
      throw error;
    }
  }

  async generateCrossDbReport(): Promise<any> {
    // Complex query across multiple databases
    const userStats = await this.primaryDataSource.query(`
      SELECT 
        COUNT(*) as total_users,
        COUNT(CASE WHEN created_at > NOW() - INTERVAL '30 days' THEN 1 END) as new_users_30d,
        AVG(EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400) as avg_age_days
      FROM users
      WHERE status = 'active';
    `);

    const salesStats = await this.analyticsDataSource.query(`
      SELECT 
        SUM(revenue) as total_revenue,
        COUNT(DISTINCT user_id) as unique_customers,
        AVG(revenue) as avg_revenue_per_sale
      FROM sales_analytics
      WHERE sale_date > NOW() - INTERVAL '30 days';
    `);

    const activityStats = await this.loggingDataSource.query(`
      SELECT 
        COUNT(*) as total_actions,
        COUNT(DISTINCT user_id) as active_users,
        action,
        COUNT(*) as action_count
      FROM audit_logs
      WHERE timestamp > NOW() - INTERVAL '30 days'
      GROUP BY action
      ORDER BY action_count DESC;
    `);

    return {
      period: '30 days',
      users: userStats[0],
      sales: salesStats[0],
      activities: activityStats,
      generatedAt: new Date(),
    };
  }

  // Data synchronization between databases
  async syncUserDataToAnalytics(userId: number): Promise<void> {
    const user = await this.usersRepository.findOne({
      where: { id: userId },
      relations: ['orders'],
    });

    if (!user) {
      throw new NotFoundException('User not found');
    }

    // Calculate user metrics
    const totalOrders = user.orders.length;
    const totalSpent = user.orders.reduce((sum, order) => sum + order.total, 0);
    const lastOrderDate = user.orders.length > 0 
      ? new Date(Math.max(...user.orders.map(order => order.createdAt.getTime())))
      : null;

    // Update or create analytics record
    await this.userAnalyticsRepository.upsert({
      userId: user.id,
      totalOrders,
      totalSpent,
      averageOrderValue: totalOrders > 0 ? totalSpent / totalOrders : 0,
      lastOrderDate,
      lastUpdated: new Date(),
    }, ['userId']);
  }

  // Cleanup old data across databases
  async cleanupOldData(retentionDays: number): Promise<void> {
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - retentionDays);

    // Clean up logs
    const deletedLogs = await this.loggingDataSource.query(`
      DELETE FROM audit_logs 
      WHERE timestamp < $1
    `, [cutoffDate]);

    // Clean up old analytics data
    const deletedAnalytics = await this.analyticsDataSource.query(`
      DELETE FROM analytics_events 
      WHERE event_date < $1
    `, [cutoffDate]);

    console.log(`Cleanup completed: ${deletedLogs.affectedRows} logs, ${deletedAnalytics.affectedRows} analytics records deleted`);
  }
}
```

### Database Sharding Strategy
```typescript
@Injectable()
export class ShardingService {
  private shards = new Map<string, DataSource>();

  constructor() {
    this.initializeShards();
  }

  private initializeShards() {
    // Initialize multiple database shards
    for (let i = 1; i <= 4; i++) {
      const dataSource = new DataSource({
        type: 'postgres',
        host: process.env[`SHARD_${i}_HOST`],
        port: parseInt(process.env[`SHARD_${i}_PORT`], 10),
        username: process.env[`SHARD_${i}_USERNAME`],
        password: process.env[`SHARD_${i}_PASSWORD`],
        database: process.env[`SHARD_${i}_DATABASE`],
        entities: [User, Order, Product],
      });
      
      this.shards.set(`shard_${i}`, dataSource);
    }
  }

  getShardForUser(userId: number): DataSource {
    // Simple modulo sharding
    const shardNumber = (userId % 4) + 1;
    return this.shards.get(`shard_${shardNumber}`);
  }

  getShardForOrder(orderId: number): DataSource {
    // Consistent hashing for orders
    const hash = this.simpleHash(orderId.toString());
    const shardNumber = (hash % 4) + 1;
    return this.shards.get(`shard_${shardNumber}`);
  }

  async findUserAcrossShards(userId: number): Promise<User | null> {
    const shard = this.getShardForUser(userId);
    const repository = shard.getRepository(User);
    return repository.findOne({ where: { id: userId } });
  }

  async broadcastQuery(query: string, parameters: any[] = []): Promise<any[]> {
    const results = await Promise.all(
      Array.from(this.shards.values()).map(shard =>
        shard.query(query, parameters)
      )
    );
    
    // Combine results from all shards
    return results.flat();
  }

  private simpleHash(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }
}
```

This comprehensive guide covers all advanced database patterns for Phase 3, providing enterprise-level database management techniques and optimization strategies for building scalable NestJS applications.
