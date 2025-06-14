

## 🟣 **18. Custom Libraries and Monorepo (Nx)**

Using custom libraries and a monorepo setup improves code reuse, separation of concerns, and scalability in large NestJS projects.

---

### ✅ 1. Why Use Custom Libraries?

* **Reusability**: Share logic like validation, logging, auth, etc.
* **Maintainability**: Encapsulate and version internal features.
* **Isolation**: Prevent accidental dependency bleed.
* **Testing**: Easily unit test modules in isolation.

---

### ✅ 2. Monorepo with Nx

**Nx** is a powerful build system for full-stack monorepos.

#### 🔧 Install Nx

```bash
npx create-nx-workspace@latest nest-monorepo
```

Choose:

* `nest` as preset
* Name the app (e.g., `api`)
* Set package manager (npm/yarn/pnpm)

---

### ✅ 3. Create a Custom Library

```bash
nx g @nrwl/nest:library auth
```

This creates a shared library under `libs/auth`.

Organize code like:

```
libs/
└── auth/
    ├── src/
    │   ├── lib/
    │   │   ├── auth.module.ts
    │   │   ├── auth.service.ts
    │   │   └── jwt.strategy.ts
    └── index.ts
```

---

### ✅ 4. Use the Library in an App

In your main NestJS app (e.g., `apps/api`):

```ts
import { AuthModule } from '@nest-monorepo/auth';

@Module({
  imports: [AuthModule],
})
export class AppModule {}
```

✅ Automatically handled by TypeScript path aliases in `tsconfig.base.json`.

---

### ✅ 5. Shared Services and DTOs

You can create multiple libraries:

* `auth`
* `validation`
* `shared-dto`
* `common-utils`

For example:

```bash
nx g @nrwl/nest:library shared-dto
```

Then define DTOs:

```ts
// libs/shared-dto/src/lib/user.dto.ts
export class CreateUserDto {
  name: string;
  email: string;
}
```

---

### ✅ 6. Affected-based CI (Nx Feature)

Nx can intelligently rebuild/test only affected projects:

```bash
nx affected:build
nx affected:test
```

This saves time in CI/CD pipelines.

---

### 🧪 Exercise

✅ **Create a shared auth library and use it in two different NestJS services**:

1. Set up Nx workspace with two apps: `api-auth` and `api-orders`.
2. Create `libs/auth`.
3. Implement JWT auth in `libs/auth`.
4. Import `AuthModule` in both services.
