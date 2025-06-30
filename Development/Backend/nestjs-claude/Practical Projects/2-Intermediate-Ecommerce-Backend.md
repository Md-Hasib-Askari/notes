# Intermediate Project: E-commerce Backend with Authentication, File Uploads, and Payment Integration

## Project Overview
Build a comprehensive e-commerce backend with advanced features including JWT authentication, file uploads, payment processing, and role-based access control.

## Features
- User authentication with JWT
- Role-based access control (Admin, Customer)
- Product management with image uploads
- Shopping cart functionality
- Order management
- Payment integration (Stripe)
- Email notifications
- Rate limiting and security

## Tech Stack
- NestJS with TypeScript
- PostgreSQL with TypeORM
- JWT authentication with Passport
- Multer for file uploads
- Stripe for payments
- Nodemailer for emails
- Redis for caching

## Project Structure
```
src/
├── app.module.ts
├── main.ts
├── auth/
│   ├── auth.module.ts
│   ├── auth.controller.ts
│   ├── auth.service.ts
│   ├── strategies/
│   └── guards/
├── users/
├── products/
├── cart/
├── orders/
├── payments/
├── uploads/
├── mail/
└── common/
    ├── decorators/
    ├── filters/
    ├── guards/
    └── interceptors/
```

## Implementation

### 1. Authentication Module
```typescript
// src/auth/auth.module.ts
import { Module } from '@nestjs/common';
import { JwtModule } from '@nestjs/jwt';
import { PassportModule } from '@nestjs/passport';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { AuthService } from './auth.service';
import { AuthController } from './auth.controller';
import { JwtStrategy } from './strategies/jwt.strategy';
import { UsersModule } from '../users/users.module';

@Module({
  imports: [
    UsersModule,
    PassportModule,
    JwtModule.registerAsync({
      imports: [ConfigModule],
      useFactory: async (configService: ConfigService) => ({
        secret: configService.get<string>('JWT_SECRET'),
        signOptions: { expiresIn: '7d' },
      }),
      inject: [ConfigService],
    }),
  ],
  providers: [AuthService, JwtStrategy],
  controllers: [AuthController],
  exports: [AuthService],
})
export class AuthModule {}
```

```typescript
// src/auth/auth.service.ts
import { Injectable, UnauthorizedException } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import { UsersService } from '../users/users.service';
import * as bcrypt from 'bcryptjs';
import { LoginDto } from './dto/login.dto';
import { RegisterDto } from './dto/register.dto';

@Injectable()
export class AuthService {
  constructor(
    private usersService: UsersService,
    private jwtService: JwtService,
  ) {}

  async register(registerDto: RegisterDto) {
    const user = await this.usersService.create(registerDto);
    const payload = { email: user.email, sub: user.id, role: user.role };
    
    return {
      access_token: this.jwtService.sign(payload),
      user: { id: user.id, email: user.email, name: user.name, role: user.role },
    };
  }

  async login(loginDto: LoginDto) {
    const user = await this.validateUser(loginDto.email, loginDto.password);
    const payload = { email: user.email, sub: user.id, role: user.role };
    
    return {
      access_token: this.jwtService.sign(payload),
      user: { id: user.id, email: user.email, name: user.name, role: user.role },
    };
  }

  async validateUser(email: string, password: string): Promise<any> {
    const user = await this.usersService.findByEmail(email);
    
    if (user && await bcrypt.compare(password, user.password)) {
      const { password, ...result } = user;
      return result;
    }
    
    throw new UnauthorizedException('Invalid credentials');
  }
}
```

### 2. Products Module with File Upload
```typescript
// src/products/entities/product.entity.ts
import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  CreateDateColumn,
  UpdateDateColumn,
  ManyToMany,
} from 'typeorm';
import { Category } from '../../categories/entities/category.entity';

@Entity('products')
export class Product {
  @PrimaryGeneratedColumn()
  id: number;

  @Column()
  name: string;

  @Column('text')
  description: string;

  @Column('decimal', { precision: 10, scale: 2 })
  price: number;

  @Column()
  stock: number;

  @Column('json', { nullable: true })
  images: string[];

  @Column({ default: true })
  isActive: boolean;

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;

  @ManyToMany(() => Category, category => category.products)
  categories: Category[];
}
```

```typescript
// src/products/products.controller.ts
import {
  Controller,
  Get,
  Post,
  Body,
  Patch,
  Param,
  Delete,
  UseGuards,
  UseInterceptors,
  UploadedFiles,
  Query,
} from '@nestjs/common';
import { FilesInterceptor } from '@nestjs/platform-express';
import { ProductsService } from './products.service';
import { CreateProductDto } from './dto/create-product.dto';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { RolesGuard } from '../auth/guards/roles.guard';
import { Roles } from '../common/decorators/roles.decorator';
import { Role } from '../users/enums/role.enum';

@Controller('products')
export class ProductsController {
  constructor(private readonly productsService: ProductsService) {}

  @Post()
  @UseGuards(JwtAuthGuard, RolesGuard)
  @Roles(Role.Admin)
  @UseInterceptors(FilesInterceptor('images', 5))
  async create(
    @Body() createProductDto: CreateProductDto,
    @UploadedFiles() images: Express.Multer.File[],
  ) {
    return this.productsService.create(createProductDto, images);
  }

  @Get()
  findAll(@Query() query: any) {
    return this.productsService.findAll(query);
  }

  @Get(':id')
  findOne(@Param('id') id: string) {
    return this.productsService.findOne(+id);
  }

  @Patch(':id')
  @UseGuards(JwtAuthGuard, RolesGuard)
  @Roles(Role.Admin)
  @UseInterceptors(FilesInterceptor('images', 5))
  update(
    @Param('id') id: string,
    @Body() updateProductDto: any,
    @UploadedFiles() images?: Express.Multer.File[],
  ) {
    return this.productsService.update(+id, updateProductDto, images);
  }

  @Delete(':id')
  @UseGuards(JwtAuthGuard, RolesGuard)
  @Roles(Role.Admin)
  remove(@Param('id') id: string) {
    return this.productsService.remove(+id);
  }
}
```

### 3. Shopping Cart Module
```typescript
// src/cart/entities/cart-item.entity.ts
import { Entity, PrimaryGeneratedColumn, Column, ManyToOne } from 'typeorm';
import { User } from '../../users/entities/user.entity';
import { Product } from '../../products/entities/product.entity';

@Entity('cart_items')
export class CartItem {
  @PrimaryGeneratedColumn()
  id: number;

  @Column()
  quantity: number;

  @ManyToOne(() => User, user => user.cartItems)
  user: User;

  @Column()
  userId: number;

  @ManyToOne(() => Product, { eager: true })
  product: Product;

  @Column()
  productId: number;
}
```

```typescript
// src/cart/cart.service.ts
import { Injectable, NotFoundException } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { CartItem } from './entities/cart-item.entity';
import { AddToCartDto } from './dto/add-to-cart.dto';

@Injectable()
export class CartService {
  constructor(
    @InjectRepository(CartItem)
    private cartRepository: Repository<CartItem>,
  ) {}

  async addToCart(userId: number, addToCartDto: AddToCartDto): Promise<CartItem> {
    const existingItem = await this.cartRepository.findOne({
      where: { userId, productId: addToCartDto.productId },
    });

    if (existingItem) {
      existingItem.quantity += addToCartDto.quantity;
      return this.cartRepository.save(existingItem);
    }

    const cartItem = this.cartRepository.create({
      userId,
      ...addToCartDto,
    });

    return this.cartRepository.save(cartItem);
  }

  async getCart(userId: number): Promise<CartItem[]> {
    return this.cartRepository.find({
      where: { userId },
      relations: ['product'],
    });
  }

  async updateQuantity(userId: number, itemId: number, quantity: number): Promise<CartItem> {
    const item = await this.cartRepository.findOne({
      where: { id: itemId, userId },
    });

    if (!item) {
      throw new NotFoundException('Cart item not found');
    }

    item.quantity = quantity;
    return this.cartRepository.save(item);
  }

  async removeFromCart(userId: number, itemId: number): Promise<void> {
    const result = await this.cartRepository.delete({ id: itemId, userId });
    
    if (result.affected === 0) {
      throw new NotFoundException('Cart item not found');
    }
  }

  async clearCart(userId: number): Promise<void> {
    await this.cartRepository.delete({ userId });
  }

  async getCartTotal(userId: number): Promise<number> {
    const cartItems = await this.getCart(userId);
    
    return cartItems.reduce((total, item) => {
      return total + (item.product.price * item.quantity);
    }, 0);
  }
}
```

### 4. Orders Module
```typescript
// src/orders/entities/order.entity.ts
import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  CreateDateColumn,
  ManyToOne,
  OneToMany,
} from 'typeorm';
import { User } from '../../users/entities/user.entity';
import { OrderItem } from './order-item.entity';

export enum OrderStatus {
  PENDING = 'pending',
  CONFIRMED = 'confirmed',
  SHIPPED = 'shipped',
  DELIVERED = 'delivered',
  CANCELLED = 'cancelled',
}

@Entity('orders')
export class Order {
  @PrimaryGeneratedColumn()
  id: number;

  @Column('decimal', { precision: 10, scale: 2 })
  total: number;

  @Column({
    type: 'enum',
    enum: OrderStatus,
    default: OrderStatus.PENDING,
  })
  status: OrderStatus;

  @Column('json')
  shippingAddress: {
    street: string;
    city: string;
    state: string;
    zipCode: string;
    country: string;
  };

  @CreateDateColumn()
  createdAt: Date;

  @ManyToOne(() => User, user => user.orders)
  user: User;

  @Column()
  userId: number;

  @OneToMany(() => OrderItem, orderItem => orderItem.order, { cascade: true })
  items: OrderItem[];

  @Column({ nullable: true })
  paymentIntentId: string;
}
```

```typescript
// src/orders/orders.service.ts
import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Order } from './entities/order.entity';
import { OrderItem } from './entities/order-item.entity';
import { CartService } from '../cart/cart.service';
import { CreateOrderDto } from './dto/create-order.dto';
import { PaymentsService } from '../payments/payments.service';
import { MailService } from '../mail/mail.service';

@Injectable()
export class OrdersService {
  constructor(
    @InjectRepository(Order)
    private ordersRepository: Repository<Order>,
    @InjectRepository(OrderItem)
    private orderItemsRepository: Repository<OrderItem>,
    private cartService: CartService,
    private paymentsService: PaymentsService,
    private mailService: MailService,
  ) {}

  async create(userId: number, createOrderDto: CreateOrderDto): Promise<Order> {
    const cartItems = await this.cartService.getCart(userId);
    
    if (cartItems.length === 0) {
      throw new Error('Cart is empty');
    }

    const total = await this.cartService.getCartTotal(userId);

    // Create payment intent
    const paymentIntent = await this.paymentsService.createPaymentIntent(
      total * 100, // Convert to cents
      'usd'
    );

    // Create order
    const order = this.ordersRepository.create({
      userId,
      total,
      shippingAddress: createOrderDto.shippingAddress,
      paymentIntentId: paymentIntent.id,
    });

    const savedOrder = await this.ordersRepository.save(order);

    // Create order items
    for (const cartItem of cartItems) {
      const orderItem = this.orderItemsRepository.create({
        orderId: savedOrder.id,
        productId: cartItem.productId,
        quantity: cartItem.quantity,
        price: cartItem.product.price,
      });
      await this.orderItemsRepository.save(orderItem);
    }

    // Clear cart
    await this.cartService.clearCart(userId);

    // Send confirmation email
    await this.mailService.sendOrderConfirmation(savedOrder);

    return this.findOne(savedOrder.id);
  }

  async findAll(userId?: number): Promise<Order[]> {
    const where = userId ? { userId } : {};
    
    return this.ordersRepository.find({
      where,
      relations: ['items', 'items.product'],
      order: { createdAt: 'DESC' },
    });
  }

  async findOne(id: number): Promise<Order> {
    return this.ordersRepository.findOne({
      where: { id },
      relations: ['items', 'items.product', 'user'],
    });
  }

  async updateStatus(id: number, status: string): Promise<Order> {
    await this.ordersRepository.update(id, { status });
    return this.findOne(id);
  }
}
```

### 5. Payment Integration
```typescript
// src/payments/payments.service.ts
import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import Stripe from 'stripe';

@Injectable()
export class PaymentsService {
  private stripe: Stripe;

  constructor(private configService: ConfigService) {
    this.stripe = new Stripe(configService.get<string>('STRIPE_SECRET_KEY'), {
      apiVersion: '2022-11-15',
    });
  }

  async createPaymentIntent(amount: number, currency: string) {
    return this.stripe.paymentIntents.create({
      amount,
      currency,
      automatic_payment_methods: {
        enabled: true,
      },
    });
  }

  async confirmPayment(paymentIntentId: string) {
    return this.stripe.paymentIntents.retrieve(paymentIntentId);
  }

  async createRefund(paymentIntentId: string, amount?: number) {
    return this.stripe.refunds.create({
      payment_intent: paymentIntentId,
      amount,
    });
  }

  async handleWebhook(signature: string, payload: Buffer) {
    const endpointSecret = this.configService.get<string>('STRIPE_WEBHOOK_SECRET');
    
    try {
      const event = this.stripe.webhooks.constructEvent(
        payload,
        signature,
        endpointSecret,
      );

      switch (event.type) {
        case 'payment_intent.succeeded':
          await this.handlePaymentSuccess(event.data.object);
          break;
        case 'payment_intent.payment_failed':
          await this.handlePaymentFailure(event.data.object);
          break;
      }

      return { received: true };
    } catch (err) {
      throw new Error(`Webhook signature verification failed: ${err.message}`);
    }
  }

  private async handlePaymentSuccess(paymentIntent: any) {
    // Update order status, send confirmation email, etc.
    console.log('Payment succeeded:', paymentIntent.id);
  }

  private async handlePaymentFailure(paymentIntent: any) {
    // Handle payment failure
    console.log('Payment failed:', paymentIntent.id);
  }
}
```

### 6. File Upload Service
```typescript
// src/uploads/uploads.service.ts
import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import * as AWS from 'aws-sdk';
import { v4 as uuid } from 'uuid';

@Injectable()
export class UploadsService {
  private s3: AWS.S3;

  constructor(private configService: ConfigService) {
    this.s3 = new AWS.S3({
      accessKeyId: configService.get<string>('AWS_ACCESS_KEY_ID'),
      secretAccessKey: configService.get<string>('AWS_SECRET_ACCESS_KEY'),
      region: configService.get<string>('AWS_REGION'),
    });
  }

  async uploadFile(file: Express.Multer.File): Promise<string> {
    const fileName = `${uuid()}-${file.originalname}`;
    const bucketName = this.configService.get<string>('AWS_S3_BUCKET');

    const uploadResult = await this.s3.upload({
      Bucket: bucketName,
      Key: fileName,
      Body: file.buffer,
      ContentType: file.mimetype,
      ACL: 'public-read',
    }).promise();

    return uploadResult.Location;
  }

  async uploadMultipleFiles(files: Express.Multer.File[]): Promise<string[]> {
    const uploadPromises = files.map(file => this.uploadFile(file));
    return Promise.all(uploadPromises);
  }

  async deleteFile(fileUrl: string): Promise<void> {
    const fileName = fileUrl.split('/').pop();
    const bucketName = this.configService.get<string>('AWS_S3_BUCKET');

    await this.s3.deleteObject({
      Bucket: bucketName,
      Key: fileName,
    }).promise();
  }
}
```

### 7. Rate Limiting and Security
```typescript
// src/main.ts
import { NestFactory } from '@nestjs/core';
import { ValidationPipe } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import * as helmet from 'helmet';
import * as rateLimit from 'express-rate-limit';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  const configService = app.get(ConfigService);

  // Security middleware
  app.use(helmet());

  // Rate limiting
  app.use(
    rateLimit({
      windowMs: 15 * 60 * 1000, // 15 minutes
      max: 100, // limit each IP to 100 requests per windowMs
    }),
  );

  // Global validation
  app.useGlobalPipes(new ValidationPipe({
    whitelist: true,
    forbidNonWhitelisted: true,
    transform: true,
  }));

  // CORS
  app.enableCors({
    origin: configService.get<string>('FRONTEND_URL'),
    credentials: true,
  });

  const port = configService.get<number>('PORT') || 3000;
  await app.listen(port);
}
bootstrap();
```

## Environment Configuration
```env
# Database
DATABASE_URL=postgresql://username:password@localhost:5432/ecommerce_db

# JWT
JWT_SECRET=your-super-secret-jwt-key

# Stripe
STRIPE_SECRET_KEY=sk_test_your_stripe_secret_key
STRIPE_PUBLISHABLE_KEY=pk_test_your_stripe_publishable_key
STRIPE_WEBHOOK_SECRET=whsec_your_webhook_secret

# AWS S3
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
AWS_S3_BUCKET=your-bucket-name

# Email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-email-password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# App
PORT=3000
FRONTEND_URL=http://localhost:3001
```

## Learning Objectives
- Implement JWT authentication and authorization
- Handle file uploads and cloud storage
- Integrate payment processing
- Build complex business logic
- Implement caching strategies
- Handle email notifications
- Apply security best practices
- Structure large applications

This project demonstrates intermediate NestJS concepts and real-world application patterns.
