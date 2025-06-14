
## 🟣 **17. CI/CD and Dockerization**

Ensuring smooth, consistent deployments is essential for scaling and collaboration. NestJS fits well into container-based and CI/CD workflows.

---

### ✅ 1. Dockerization of NestJS

#### 📦 Step 1: `Dockerfile`

```dockerfile
# Stage 1: Build
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Stage 2: Run
FROM node:18-alpine
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY .env .env
CMD ["node", "dist/main.js"]
```

---

#### 📦 Step 2: `docker-compose.yml`

```yaml
version: '3'
services:
  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
    env_file:
      - .env
    depends_on:
      - postgres

  postgres:
    image: postgres
    restart: always
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: nestdb
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
```

---

### ✅ 2. CI/CD with GitHub Actions

#### 🛠 `.github/workflows/deploy.yml`

```yaml
name: Deploy to Production

on:
  push:
    branches:
      - main

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install dependencies
        run: npm install

      - name: Run tests
        run: npm run test

      - name: Build project
        run: npm run build

      - name: Deploy to server via SSH
        uses: appleboy/scp-action@master
        with:
          host: ${{ secrets.HOST }}
          username: ${{ secrets.USERNAME }}
          key: ${{ secrets.PRIVATE_KEY }}
          source: "dist/*"
          target: "/home/ubuntu/app"
```

You can replace this step with container deploy (Docker Hub, ECR, Heroku, etc.)

---

### ✅ 3. Deployment Targets

#### 🔧 Options:

* **Heroku**: Simple, but limited for complex apps.
* **Render**: Like Heroku but more flexible.
* **AWS/GCP/Azure**: Full control, use ECS, GKE, or VMs.
* **Docker Swarm / Kubernetes**: For container orchestration.
* **Vercel/Netlify**: Only for static frontends, not NestJS.

---

### 🧪 Exercise

✅ **Containerize your NestJS app and deploy it to a cloud provider.**

1. Create a Dockerfile and `.dockerignore`.
2. Test the container locally.
3. Push to GitHub.
4. Set up GitHub Actions to build and deploy on `push`.
5. (Optional) Use Render, Railway, or AWS EC2 for deployment.

