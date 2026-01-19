## 🥖 Bun Cheatsheet (Quick Reference)

---

## 1️⃣ Installation & Version

```bash
curl -fsSL https://bun.sh/install | bash
bun --version
bun upgrade
```

---

## 2️⃣ Project Setup

```bash
bun init
bun init -y
```

**Run files**

```bash
bun index.ts
bun src/server.ts
```

---

## 3️⃣ Package Management (npm replacement)

```bash
bun install              # install deps
bun add express           # add dependency
bun add -d typescript     # dev dependency
bun remove lodash
bun update
```

**Lockfile**

* `bun.lockb` (binary, fast, deterministic)

---

## 4️⃣ Scripts (package.json)

```json
{
  "scripts": {
    "dev": "bun src/index.ts",
    "test": "bun test"
  }
}
```

```bash
bun run dev
```

---

## 5️⃣ TypeScript (Zero Config)

```bash
bun app.ts
bun app.tsx
```

Optional `tsconfig.json`:

```json
{
  "compilerOptions": {
    "strict": true,
    "target": "ESNext"
  }
}
```

---

## 6️⃣ Environment Variables

```bash
bun run index.ts
```

```env
DATABASE_URL=postgres://...
```

```ts
process.env.DATABASE_URL
```

Bun auto-loads `.env`

---

## 7️⃣ File System (Bun Native)

```ts
// Read file
const file = Bun.file("data.json");
const text = await file.text();
const json = await file.json();

// Write file
await Bun.write("out.txt", "Hello Bun");
```

---

## 8️⃣ Fetch API (Built-in)

```ts
const res = await fetch("https://api.example.com");
const data = await res.json();
```

No axios needed.

---

## 9️⃣ HTTP Server (Fast)

```ts
Bun.serve({
  port: 3000,
  fetch(req) {
    return new Response("Hello Bun");
  }
});
```

---

## 🔌 REST API Example

```ts
Bun.serve({
  fetch(req) {
    if (req.method === "POST") {
      return Response.json({ ok: true });
    }
    return new Response("Not Found", { status: 404 });
  }
});
```

---

## 1️⃣0️⃣ WebSockets

```ts
Bun.serve({
  websocket: {
    message(ws, msg) {
      ws.send(`Echo: ${msg}`);
    }
  },
  fetch(req, server) {
    if (server.upgrade(req)) return;
    return new Response("Upgrade failed", { status: 400 });
  }
});
```

---

## 1️⃣1️⃣ Testing (Jest replacement)

```bash
bun test
bun test --watch
```

```ts
import { test, expect } from "bun:test";

test("sum", () => {
  expect(1 + 2).toBe(3);
});
```

---

## 1️⃣2️⃣ Mocks & Spies

```ts
const fn = vi.fn();
fn("hello");
expect(fn).toHaveBeenCalled();
```

---

## 1️⃣3️⃣ Watch Mode

```bash
bun --watch index.ts
bun test --watch
```

---

## 1️⃣4️⃣ SQLite (Built-in)

```ts
import { Database } from "bun:sqlite";

const db = new Database("db.sqlite");
db.query("CREATE TABLE users (id INTEGER)").run();
```

---

## 1️⃣5️⃣ PostgreSQL / MySQL

```bash
bun add pg
```

```ts
import { Client } from "pg";

const client = new Client({ connectionString: process.env.DATABASE_URL });
await client.connect();
```

---

## 1️⃣6️⃣ Elysia (Bun-native Framework)

```bash
bun add elysia
```

```ts
import { Elysia } from "elysia";

new Elysia()
  .get("/", () => "Hello")
  .listen(3000);
```

---

## 1️⃣7️⃣ CLI Tools

```ts
console.log(Bun.argv);
```

```bash
bun mycli.ts --name hasib
```

---

## 1️⃣8️⃣ Bundling

```bash
bun build src/index.ts --outdir dist
```

---

## 1️⃣9️⃣ Docker (Minimal)

```dockerfile
FROM oven/bun
WORKDIR /app
COPY . .
RUN bun install
CMD ["bun", "src/index.ts"]
```

---

## 2️⃣0️⃣ Common Replacements

| Instead of | Use      |
| ---------- | -------- |
| node       | bun      |
| npm        | bun      |
| ts-node    | bun      |
| jest       | bun test |
| axios      | fetch    |
| fs         | Bun.file |

---

## ⚠️ Gotchas (Read This)

* Native Node addons may break
* Some Node APIs are partial
* Not all serverless platforms support Bun
* Bun favors **web standards**, not Node legacy

---

## 🎯 Pro Tips

* Prefer **Elysia + Bun.serve**
* Use Bun for **APIs, CLIs, tooling**
* Measure performance — don’t assume
