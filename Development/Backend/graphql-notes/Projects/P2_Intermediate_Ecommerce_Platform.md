# Intermediate Project: E-commerce Platform

## Project Overview
A comprehensive GraphQL-powered e-commerce platform featuring complex data relationships, real-time inventory updates, payment integration, and image uploads. This project demonstrates intermediate GraphQL concepts and production-ready features.

## Core Features

### 1. Complex Data Relationships
- Users, Products, Categories, Orders, Reviews
- Many-to-many relationships
- Nested queries and data fetching

### 2. Real-time Inventory Updates
- Stock management with subscriptions
- Inventory tracking across multiple locations
- Low stock alerts

### 3. Payment Integration
- Stripe payment processing
- Order management
- Payment status tracking

### 4. Image Uploads
- Product image management
- File upload handling
- Image optimization and CDN integration

## Advanced Schema Design

```graphql
type User {
  id: ID!
  email: String!
  firstName: String!
  lastName: String!
  role: UserRole!
  orders: [Order!]!
  reviews: [Review!]!
  cart: Cart
  addresses: [Address!]!
  createdAt: String!
}

enum UserRole {
  CUSTOMER
  ADMIN
  MERCHANT
}

type Product {
  id: ID!
  name: String!
  description: String!
  price: Float!
  category: Category!
  images: [Image!]!
  inventory: Inventory!
  reviews: [Review!]!
  averageRating: Float
  variants: [ProductVariant!]!
  tags: [String!]!
  isActive: Boolean!
  createdAt: String!
  updatedAt: String!
}

type ProductVariant {
  id: ID!
  product: Product!
  name: String!
  price: Float!
  sku: String!
  inventory: Inventory!
  attributes: [VariantAttribute!]!
}

type VariantAttribute {
  name: String!
  value: String!
}

type Category {
  id: ID!
  name: String!
  slug: String!
  description: String
  parent: Category
  children: [Category!]!
  products: [Product!]!
  image: Image
}

type Inventory {
  id: ID!
  quantity: Int!
  reserved: Int!
  available: Int!
  location: String!
  lowStockThreshold: Int!
  isLowStock: Boolean!
  lastUpdated: String!
}

type Order {
  id: ID!
  user: User!
  items: [OrderItem!]!
  status: OrderStatus!
  total: Float!
  subtotal: Float!
  tax: Float!
  shipping: Float!
  shippingAddress: Address!
  billingAddress: Address!
  payment: Payment!
  tracking: OrderTracking
  createdAt: String!
  updatedAt: String!
}

enum OrderStatus {
  PENDING
  CONFIRMED
  PROCESSING
  SHIPPED
  DELIVERED
  CANCELLED
  REFUNDED
}

type OrderItem {
  id: ID!
  order: Order!
  product: Product!
  variant: ProductVariant
  quantity: Int!
  price: Float!
  total: Float!
}

type Payment {
  id: ID!
  order: Order!
  amount: Float!
  currency: String!
  status: PaymentStatus!
  method: PaymentMethod!
  stripePaymentId: String
  transactionId: String
  createdAt: String!
}

enum PaymentStatus {
  PENDING
  PROCESSING
  SUCCEEDED
  FAILED
  CANCELLED
  REFUNDED
}

enum PaymentMethod {
  CREDIT_CARD
  PAYPAL
  BANK_TRANSFER
  CASH_ON_DELIVERY
}

type Cart {
  id: ID!
  user: User!
  items: [CartItem!]!
  total: Float!
  itemCount: Int!
  updatedAt: String!
}

type CartItem {
  id: ID!
  cart: Cart!
  product: Product!
  variant: ProductVariant
  quantity: Int!
  price: Float!
  total: Float!
}

type Review {
  id: ID!
  user: User!
  product: Product!
  rating: Int!
  title: String!
  comment: String!
  verified: Boolean!
  helpful: Int!
  createdAt: String!
}

type Address {
  id: ID!
  user: User!
  type: AddressType!
  firstName: String!
  lastName: String!
  company: String
  street1: String!
  street2: String
  city: String!
  state: String!
  zipCode: String!
  country: String!
  isDefault: Boolean!
}

enum AddressType {
  SHIPPING
  BILLING
  BOTH
}

type Image {
  id: ID!
  url: String!
  alt: String
  width: Int
  height: Int
  size: Int
  mimeType: String!
}

type Query {
  # Products
  products(
    category: ID
    search: String
    priceRange: PriceRangeInput
    sortBy: ProductSortBy
    first: Int
    after: String
  ): ProductConnection!
  
  product(id: ID!): Product
  productBySlug(slug: String!): Product
  
  # Categories
  categories: [Category!]!
  category(id: ID!): Category
  
  # User & Orders
  me: User
  myOrders: [Order!]!
  order(id: ID!): Order
  
  # Cart
  myCart: Cart
  
  # Admin queries
  allOrders(status: OrderStatus, first: Int, after: String): OrderConnection!
  inventoryReport: [InventoryReport!]!
}

type Mutation {
  # Authentication
  register(input: RegisterInput!): AuthPayload!
  login(email: String!, password: String!): AuthPayload!
  
  # Cart management
  addToCart(productId: ID!, variantId: ID, quantity: Int!): Cart!
  updateCartItem(itemId: ID!, quantity: Int!): Cart!
  removeFromCart(itemId: ID!): Cart!
  clearCart: Boolean!
  
  # Orders
  createOrder(input: CreateOrderInput!): Order!
  updateOrderStatus(orderId: ID!, status: OrderStatus!): Order!
  
  # Payment
  createPaymentIntent(orderId: ID!): PaymentIntent!
  confirmPayment(paymentIntentId: String!): Payment!
  
  # Reviews
  createReview(input: CreateReviewInput!): Review!
  
  # File uploads
  uploadImage(file: Upload!): Image!
  
  # Admin mutations
  createProduct(input: CreateProductInput!): Product!
  updateProduct(id: ID!, input: UpdateProductInput!): Product!
  updateInventory(productId: ID!, variantId: ID, quantity: Int!): Inventory!
}

type Subscription {
  # Real-time inventory updates
  inventoryUpdated(productId: ID!): Inventory!
  lowStockAlert: Inventory!
  
  # Order tracking
  orderStatusChanged(orderId: ID!): Order!
  
  # Cart updates
  cartUpdated(userId: ID!): Cart!
}

input RegisterInput {
  email: String!
  password: String!
  firstName: String!
  lastName: String!
}

input CreateOrderInput {
  shippingAddressId: ID!
  billingAddressId: ID!
  paymentMethodId: String!
}

input CreateReviewInput {
  productId: ID!
  rating: Int!
  title: String!
  comment: String!
}

input PriceRangeInput {
  min: Float
  max: Float
}

enum ProductSortBy {
  NAME_ASC
  NAME_DESC
  PRICE_ASC
  PRICE_DESC
  RATING_DESC
  CREATED_DESC
}
```

## Advanced Server Implementation

### DataLoader for N+1 Problem
```javascript
const DataLoader = require('dataloader');

function createLoaders() {
  return {
    productLoader: new DataLoader(async (productIds) => {
      const products = await Product.find({ _id: { $in: productIds } });
      return productIds.map(id => products.find(p => p.id === id.toString()));
    }),
    
    categoryLoader: new DataLoader(async (categoryIds) => {
      const categories = await Category.find({ _id: { $in: categoryIds } });
      return categoryIds.map(id => categories.find(c => c.id === id.toString()));
    }),
    
    inventoryLoader: new DataLoader(async (productIds) => {
      const inventories = await Inventory.find({ product: { $in: productIds } });
      return productIds.map(id => inventories.find(i => i.product.toString() === id.toString()));
    }),
    
    reviewsByProductLoader: new DataLoader(async (productIds) => {
      const reviews = await Review.find({ product: { $in: productIds } });
      return productIds.map(id => reviews.filter(r => r.product.toString() === id.toString()));
    })
  };
}
```

### Real-time Subscriptions
```javascript
const { PubSub } = require('graphql-subscriptions');
const pubsub = new PubSub();

const INVENTORY_UPDATED = 'INVENTORY_UPDATED';
const LOW_STOCK_ALERT = 'LOW_STOCK_ALERT';
const ORDER_STATUS_CHANGED = 'ORDER_STATUS_CHANGED';

const resolvers = {
  Subscription: {
    inventoryUpdated: {
      subscribe: withFilter(
        () => pubsub.asyncIterator(INVENTORY_UPDATED),
        (payload, variables) => {
          return payload.inventoryUpdated.product.toString() === variables.productId;
        }
      )
    },
    
    lowStockAlert: {
      subscribe: () => pubsub.asyncIterator(LOW_STOCK_ALERT)
    },
    
    orderStatusChanged: {
      subscribe: withFilter(
        () => pubsub.asyncIterator(ORDER_STATUS_CHANGED),
        (payload, variables) => {
          return payload.orderStatusChanged.id === variables.orderId;
        }
      )
    }
  },
  
  Mutation: {
    updateInventory: async (_, { productId, variantId, quantity }, { user, loaders }) => {
      // Authorization check
      if (!user || user.role !== 'ADMIN') {
        throw new Error('Unauthorized');
      }
      
      const inventory = await Inventory.findOneAndUpdate(
        { product: productId, variant: variantId },
        { 
          quantity,
          available: quantity - inventory.reserved,
          lastUpdated: new Date()
        },
        { new: true }
      ).populate('product');
      
      // Publish real-time update
      pubsub.publish(INVENTORY_UPDATED, { inventoryUpdated: inventory });
      
      // Check for low stock
      if (inventory.available <= inventory.lowStockThreshold) {
        pubsub.publish(LOW_STOCK_ALERT, { lowStockAlert: inventory });
      }
      
      return inventory;
    }
  }
};
```

### Payment Integration with Stripe
```javascript
const stripe = require('stripe')(process.env.STRIPE_SECRET_KEY);

const paymentResolvers = {
  Mutation: {
    createPaymentIntent: async (_, { orderId }, { user }) => {
      const order = await Order.findById(orderId).populate('user');
      
      if (!order || order.user.id !== user.id) {
        throw new Error('Order not found');
      }
      
      const paymentIntent = await stripe.paymentIntents.create({
        amount: Math.round(order.total * 100), // Convert to cents
        currency: 'usd',
        metadata: {
          orderId: order.id,
          userId: user.id
        }
      });
      
      return {
        clientSecret: paymentIntent.client_secret,
        id: paymentIntent.id
      };
    },
    
    confirmPayment: async (_, { paymentIntentId }, { user }) => {
      const paymentIntent = await stripe.paymentIntents.retrieve(paymentIntentId);
      
      if (paymentIntent.status === 'succeeded') {
        const orderId = paymentIntent.metadata.orderId;
        
        // Update order and create payment record
        const order = await Order.findByIdAndUpdate(
          orderId,
          { status: 'CONFIRMED' },
          { new: true }
        );
        
        const payment = await Payment.create({
          order: orderId,
          amount: paymentIntent.amount / 100,
          currency: paymentIntent.currency,
          status: 'SUCCEEDED',
          method: 'CREDIT_CARD',
          stripePaymentId: paymentIntent.id,
          transactionId: paymentIntent.id
        });
        
        // Publish order status change
        pubsub.publish(ORDER_STATUS_CHANGED, { orderStatusChanged: order });
        
        return payment;
      }
      
      throw new Error('Payment not confirmed');
    }
  }
};
```

### File Upload Handling
```javascript
const { GraphQLUpload } = require('graphql-upload');
const AWS = require('aws-sdk');
const sharp = require('sharp');

const s3 = new AWS.S3({
  accessKeyId: process.env.AWS_ACCESS_KEY_ID,
  secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
  region: process.env.AWS_REGION
});

const uploadResolvers = {
  Upload: GraphQLUpload,
  
  Mutation: {
    uploadImage: async (_, { file }) => {
      const { createReadStream, filename, mimetype } = await file;
      const stream = createReadStream();
      
      // Optimize image with Sharp
      const optimizedBuffer = await sharp(stream)
        .resize(800, 600, { fit: 'inside', withoutEnlargement: true })
        .jpeg({ quality: 80 })
        .toBuffer();
      
      // Upload to S3
      const key = `products/${Date.now()}-${filename}`;
      const uploadResult = await s3.upload({
        Bucket: process.env.S3_BUCKET,
        Key: key,
        Body: optimizedBuffer,
        ContentType: mimetype,
        ACL: 'public-read'
      }).promise();
      
      // Save to database
      const image = await Image.create({
        url: uploadResult.Location,
        alt: filename,
        size: optimizedBuffer.length,
        mimeType: mimetype
      });
      
      return image;
    }
  }
};
```

## Advanced Client Implementation

### Apollo Client Setup with Cache
```javascript
import { InMemoryCache } from '@apollo/client';

const cache = new InMemoryCache({
  typePolicies: {
    Product: {
      fields: {
        reviews: {
          merge(existing = [], incoming) {
            return [...existing, ...incoming];
          }
        }
      }
    },
    Query: {
      fields: {
        products: {
          keyArgs: ['category', 'search', 'priceRange'],
          merge(existing, incoming) {
            if (!existing) return incoming;
            return {
              ...incoming,
              edges: [...existing.edges, ...incoming.edges]
            };
          }
        }
      }
    }
  }
});
```

### Real-time Product Updates
```jsx
import { useSubscription, useQuery } from '@apollo/client';

const INVENTORY_UPDATED_SUBSCRIPTION = gql`
  subscription InventoryUpdated($productId: ID!) {
    inventoryUpdated(productId: $productId) {
      quantity
      available
      isLowStock
    }
  }
`;

function ProductDetail({ productId }) {
  const { data: product } = useQuery(GET_PRODUCT, {
    variables: { id: productId }
  });
  
  useSubscription(INVENTORY_UPDATED_SUBSCRIPTION, {
    variables: { productId },
    onSubscriptionData: ({ subscriptionData, client }) => {
      const newInventory = subscriptionData.data.inventoryUpdated;
      
      // Update cache
      client.cache.modify({
        id: client.cache.identify({ __typename: 'Product', id: productId }),
        fields: {
          inventory: () => newInventory
        }
      });
    }
  });
  
  return (
    <div>
      <h1>{product?.name}</h1>
      <p>In Stock: {product?.inventory.available}</p>
      {product?.inventory.isLowStock && (
        <span className="low-stock-warning">Low Stock!</span>
      )}
    </div>
  );
}
```

## Learning Outcomes

### Advanced GraphQL Concepts
- Complex schema design with relationships
- DataLoader pattern for performance optimization
- Real-time subscriptions with filters
- File upload handling
- Payment processing integration
- Advanced caching strategies

### Production Concepts
- Database optimization and indexing
- Image processing and CDN integration
- Security best practices
- Error handling and logging
- Performance monitoring
- Scalable architecture patterns

### Integration Skills
- Third-party payment processing
- Cloud storage services
- Real-time communication
- Advanced client-side state management
- Optimistic updates and error recovery

## Next Steps
1. Implement GraphQL Federation for microservices
2. Add comprehensive testing suite
3. Implement advanced search with Elasticsearch
4. Add recommendation engine
5. Scale with Redis caching and CDN
6. Mobile app development with GraphQL
