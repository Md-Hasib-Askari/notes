
## 🔵 **12. WebSockets with Gateway in NestJS**

WebSockets allow real-time, two-way communication between client and server. NestJS uses `@nestjs/websockets` with `socket.io` or `ws`.

---

### ✅ 1. **Install Required Packages**

```bash
npm install @nestjs/websockets @nestjs/platform-socket.io socket.io
```

> Optional for WS protocol instead of socket.io:

```bash
npm install ws
```

---

### ✅ 2. **Basic Gateway Setup**

```ts
import {
  SubscribeMessage,
  WebSocketGateway,
  OnGatewayConnection,
  OnGatewayDisconnect,
  WebSocketServer,
} from '@nestjs/websockets';
import { Server, Socket } from 'socket.io';

@WebSocketGateway()
export class ChatGateway implements OnGatewayConnection, OnGatewayDisconnect {
  @WebSocketServer()
  server: Server;

  handleConnection(client: Socket) {
    console.log(`Client connected: ${client.id}`);
  }

  handleDisconnect(client: Socket) {
    console.log(`Client disconnected: ${client.id}`);
  }

  @SubscribeMessage('message')
  handleMessage(client: Socket, payload: any): void {
    this.server.emit('message', payload); // Broadcast to all
  }
}
```

---

### ✅ 3. **Client-side (Socket.IO Example)**

```html
<script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
<script>
  const socket = io('http://localhost:3000');

  socket.on('connect', () => {
    console.log('Connected:', socket.id);
    socket.emit('message', { user: 'Hasib', msg: 'Hello' });
  });

  socket.on('message', (data) => {
    console.log('Received:', data);
  });
</script>
```

---

### ✅ 4. **Namespacing and Rooms**

#### Namespaces:

```ts
@WebSocketGateway({ namespace: '/chat' })
```

#### Rooms:

```ts
client.join('room1');
this.server.to('room1').emit('message', payload);
```

---

### ✅ 5. **Inject Services into Gateways**

```ts
constructor(private readonly chatService: ChatService) {}

@SubscribeMessage('message')
handleMessage(client: Socket, payload: any) {
  this.chatService.saveMessage(payload);
  client.emit('message', payload);
}
```

---

### 💪 Exercise

✅ **Create a real-time chat or notification system**

1. Build a gateway that listens to `chat` or `notification` messages.
2. Broadcast the message to all clients or to specific rooms.
3. Add reconnection logic on the frontend.
4. (Bonus) Use a service to persist messages to a database.

