

## 🔵 **11. GraphQL with NestJS**

GraphQL provides a flexible, efficient alternative to REST APIs, allowing clients to request only the data they need.

---

### ✅ 1. **GraphQL vs REST**

| REST                   | GraphQL                         |
| ---------------------- | ------------------------------- |
| Fixed endpoints        | Single endpoint (`/graphql`)    |
| Over-fetching possible | Precise data queries            |
| Uses HTTP Verbs        | Uses Queries, Mutations, Subs   |
| Multiple round-trips   | Nested queries in a single call |

---

### ✅ 2. **Install Required Packages**

```bash
npm install @nestjs/graphql @nestjs/apollo graphql apollo-server-express
npm install class-validator class-transformer
```

---

### ✅ 3. **Enable GraphQL Module**

In `AppModule`:

```ts
import { GraphQLModule } from '@nestjs/graphql';

@Module({
  imports: [
    GraphQLModule.forRoot({
      autoSchemaFile: true, // or path like 'src/schema.gql'
    }),
  ],
})
export class AppModule {}
```

---

### ✅ 4. **Create GraphQL Schema with Decorators**

#### DTOs / Types

```ts
@ObjectType()
export class User {
  @Field()
  id: string;

  @Field()
  username: string;
}
```

#### Input Types

```ts
@InputType()
export class CreateUserInput {
  @Field()
  username: string;

  @Field()
  password: string;
}
```

---

### ✅ 5. **Resolvers (Controllers for GraphQL)**

```ts
@Resolver(() => User)
export class UserResolver {
  constructor(private readonly userService: UserService) {}

  @Query(() => [User])
  async users() {
    return this.userService.findAll();
  }

  @Mutation(() => User)
  async createUser(@Args('data') data: CreateUserInput) {
    return this.userService.create(data);
  }
}
```

---

### ✅ 6. **GraphQL Playground**

Run your app and go to `http://localhost:3000/graphql` to test queries and mutations in a UI.

---

### ✅ 7. **Advanced Features**

* **@ResolveField** – for nested object resolution
* **@Context()** – for accessing request/session
* **Subscriptions** – for real-time GraphQL (via WebSockets)

---

### 💪 Exercise

✅ **Convert an existing REST module to GraphQL**

1. Replace the REST controller with a GraphQL resolver.
2. Define ObjectType and InputType classes.
3. Use queries and mutations to interact with your service.
4. Test everything in the GraphQL Playground.

