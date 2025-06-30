# Deployment Patterns for LangGraph

Production-ready deployment strategies for LangGraph applications, focusing on containerization, scaling, and integration patterns.

## Production Architecture

### Containerization Strategies

```dockerfile
# Dockerfile for LangGraph Application
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create non-root user
RUN useradd -m -u 1000 langraph && chown -R langraph:langraph /app
USER langraph

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml for Development
version: '3.8'

services:
  langraph-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - POSTGRES_URL=postgresql://user:pass@postgres:5432/langraph_db
      - LOG_LEVEL=INFO
    depends_on:
      - redis
      - postgres
    volumes:
      - ./logs:/app/logs
    networks:
      - langraph-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - langraph-network

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: langraph_db
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    networks:
      - langraph-network

volumes:
  redis_data:
  postgres_data:

networks:
  langraph-network:
    driver: bridge
```

### Scaling Considerations

```python
# src/scaling/load_balancer.py
from typing import List, Dict, Any
import asyncio
import aiohttp
from dataclasses import dataclass
import random
from datetime import datetime, timedelta

@dataclass
class GraphInstance:
    id: str
    url: str
    health_status: str
    last_health_check: datetime
    current_load: int
    max_capacity: int

class LangGraphLoadBalancer:
    def __init__(self):
        self.instances: List[GraphInstance] = []
        self.round_robin_index = 0
        self.health_check_interval = 30  # seconds
    
    def register_instance(self, instance_id: str, url: str, max_capacity: int = 100):
        """Register a new LangGraph instance"""
        instance = GraphInstance(
            id=instance_id,
            url=url,
            health_status="unknown",
            last_health_check=datetime.now(),
            current_load=0,
            max_capacity=max_capacity
        )
        self.instances.append(instance)
    
    async def health_check(self):
        """Periodic health check for all instances"""
        while True:
            tasks = []
            for instance in self.instances:
                tasks.append(self._check_instance_health(instance))
            
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(self.health_check_interval)
    
    async def _check_instance_health(self, instance: GraphInstance):
        """Check health of a single instance"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{instance.url}/health", timeout=5) as response:
                    if response.status == 200:
                        instance.health_status = "healthy"
                        data = await response.json()
                        instance.current_load = data.get("current_load", 0)
                    else:
                        instance.health_status = "unhealthy"
        except Exception:
            instance.health_status = "unhealthy"
        
        instance.last_health_check = datetime.now()
    
    def select_instance(self, strategy: str = "least_loaded") -> GraphInstance:
        """Select an instance based on load balancing strategy"""
        healthy_instances = [
            inst for inst in self.instances 
            if inst.health_status == "healthy" and inst.current_load < inst.max_capacity
        ]
        
        if not healthy_instances:
            raise Exception("No healthy instances available")
        
        if strategy == "round_robin":
            return self._round_robin_selection(healthy_instances)
        elif strategy == "least_loaded":
            return min(healthy_instances, key=lambda x: x.current_load)
        elif strategy == "random":
            return random.choice(healthy_instances)
        else:
            return healthy_instances[0]
    
    def _round_robin_selection(self, instances: List[GraphInstance]) -> GraphInstance:
        """Round-robin selection algorithm"""
        instance = instances[self.round_robin_index % len(instances)]
        self.round_robin_index += 1
        return instance

# Auto-scaling configuration
class AutoScaler:
    def __init__(self, load_balancer: LangGraphLoadBalancer):
        self.load_balancer = load_balancer
        self.scale_up_threshold = 80  # CPU percentage
        self.scale_down_threshold = 20
        self.min_instances = 2
        self.max_instances = 10
    
    async def monitor_and_scale(self):
        """Monitor load and scale instances automatically"""
        while True:
            avg_load = self._calculate_average_load()
            
            if avg_load > self.scale_up_threshold:
                await self._scale_up()
            elif avg_load < self.scale_down_threshold:
                await self._scale_down()
            
            await asyncio.sleep(60)  # Check every minute
    
    def _calculate_average_load(self) -> float:
        """Calculate average load across all healthy instances"""
        healthy_instances = [
            inst for inst in self.load_balancer.instances 
            if inst.health_status == "healthy"
        ]
        
        if not healthy_instances:
            return 0
        
        total_load = sum(inst.current_load for inst in healthy_instances)
        return total_load / len(healthy_instances)
```

### State Persistence Solutions

```python
# src/persistence/state_store.py
import json
import asyncio
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import aioredis
import asyncpg
from datetime import datetime

class StateStore(ABC):
    @abstractmethod
    async def save_state(self, execution_id: str, state: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    async def load_state(self, execution_id: str) -> Optional[Dict[str, Any]]:
        pass
    
    @abstractmethod
    async def delete_state(self, execution_id: str) -> bool:
        pass

class RedisStateStore(StateStore):
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis = None
    
    async def connect(self):
        """Initialize Redis connection"""
        self.redis = await aioredis.from_url(self.redis_url)
    
    async def save_state(self, execution_id: str, state: Dict[str, Any]) -> bool:
        """Save state to Redis with TTL"""
        try:
            state_data = {
                "state": state,
                "timestamp": datetime.now().isoformat(),
                "version": 1
            }
            
            # Set with 24-hour expiration
            await self.redis.setex(
                f"langraph:state:{execution_id}",
                86400,  # 24 hours
                json.dumps(state_data, default=str)
            )
            return True
        except Exception as e:
            print(f"Error saving state to Redis: {e}")
            return False
    
    async def load_state(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Load state from Redis"""
        try:
            data = await self.redis.get(f"langraph:state:{execution_id}")
            if data:
                state_data = json.loads(data)
                return state_data["state"]
            return None
        except Exception as e:
            print(f"Error loading state from Redis: {e}")
            return None

class PostgresStateStore(StateStore):
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None
    
    async def initialize(self):
        """Initialize connection pool and create tables"""
        self.pool = await asyncpg.create_pool(self.connection_string)
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS langraph_states (
                    execution_id VARCHAR(255) PRIMARY KEY,
                    state JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
    
    async def save_state(self, execution_id: str, state: Dict[str, Any]) -> bool:
        """Save state to PostgreSQL"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO langraph_states (execution_id, state, updated_at)
                    VALUES ($1, $2, NOW())
                    ON CONFLICT (execution_id)
                    DO UPDATE SET state = $2, updated_at = NOW()
                """, execution_id, json.dumps(state, default=str))
            return True
        except Exception as e:
            print(f"Error saving state to PostgreSQL: {e}")
            return False
    
    async def load_state(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Load state from PostgreSQL"""
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT state FROM langraph_states WHERE execution_id = $1",
                    execution_id
                )
                if row:
                    return json.loads(row['state'])
                return None
        except Exception as e:
            print(f"Error loading state from PostgreSQL: {e}")
            return None
```

## Integration Patterns

### API Endpoints for Graphs

```python
# src/api/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uuid
from datetime import datetime

app = FastAPI(title="LangGraph API", version="1.0.0")

class GraphRequest(BaseModel):
    graph_id: str
    input_data: Dict[str, Any]
    execution_options: Optional[Dict[str, Any]] = {}

class GraphResponse(BaseModel):
    execution_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class GraphExecutor:
    def __init__(self):
        self.executions = {}
        self.state_store = RedisStateStore("redis://localhost:6379")
    
    async def execute_graph(self, request: GraphRequest) -> str:
        """Execute a graph asynchronously"""
        execution_id = str(uuid.uuid4())
        
        self.executions[execution_id] = {
            "status": "running",
            "started_at": datetime.now(),
            "graph_id": request.graph_id
        }
        
        try:
            # Load and execute the graph
            graph = await self._load_graph(request.graph_id)
            result = await graph.ainvoke(request.input_data)
            
            self.executions[execution_id].update({
                "status": "completed",
                "result": result,
                "completed_at": datetime.now()
            })
            
        except Exception as e:
            self.executions[execution_id].update({
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now()
            })
        
        return execution_id

executor = GraphExecutor()

@app.post("/graphs/execute", response_model=GraphResponse)
async def execute_graph(request: GraphRequest, background_tasks: BackgroundTasks):
    """Execute a graph and return execution ID"""
    execution_id = await executor.execute_graph(request)
    
    return GraphResponse(
        execution_id=execution_id,
        status="running"
    )

@app.get("/executions/{execution_id}", response_model=GraphResponse)
async def get_execution_status(execution_id: str):
    """Get the status of a graph execution"""
    if execution_id not in executor.executions:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    execution = executor.executions[execution_id]
    
    return GraphResponse(
        execution_id=execution_id,
        status=execution["status"],
        result=execution.get("result"),
        error=execution.get("error")
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "current_load": len([
            e for e in executor.executions.values() 
            if e["status"] == "running"
        ])
    }
```

### Webhook Integrations

```python
# src/webhooks/manager.py
import aiohttp
import asyncio
from typing import Dict, List, Callable
from dataclasses import dataclass
from datetime import datetime

@dataclass
class WebhookConfig:
    url: str
    events: List[str]
    headers: Dict[str, str]
    retry_attempts: int = 3
    timeout: int = 30

class WebhookManager:
    def __init__(self):
        self.webhooks: Dict[str, WebhookConfig] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
    
    def register_webhook(self, webhook_id: str, config: WebhookConfig):
        """Register a webhook for specific events"""
        self.webhooks[webhook_id] = config
    
    async def emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event to all registered webhooks"""
        tasks = []
        
        for webhook_id, config in self.webhooks.items():
            if event_type in config.events:
                task = self._send_webhook(webhook_id, event_type, data)
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_webhook(self, webhook_id: str, event_type: str, data: Dict[str, Any]):
        """Send webhook with retry logic"""
        config = self.webhooks[webhook_id]
        payload = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "webhook_id": webhook_id
        }
        
        for attempt in range(config.retry_attempts):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        config.url,
                        json=payload,
                        headers=config.headers,
                        timeout=config.timeout
                    ) as response:
                        if response.status < 400:
                            print(f"Webhook {webhook_id} sent successfully")
                            return
                        else:
                            print(f"Webhook {webhook_id} failed with status {response.status}")
            
            except Exception as e:
                print(f"Webhook {webhook_id} attempt {attempt + 1} failed: {e}")
                if attempt < config.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

# Integration with graph execution
webhook_manager = WebhookManager()

async def graph_with_webhooks(state: Dict[str, Any]) -> Dict[str, Any]:
    """Graph node that emits webhook events"""
    
    # Emit start event
    await webhook_manager.emit_event("graph.started", {
        "execution_id": state.get("execution_id"),
        "input_data": state.get("input_data")
    })
    
    try:
        # Process the graph
        result = await process_graph_logic(state)
        
        # Emit success event
        await webhook_manager.emit_event("graph.completed", {
            "execution_id": state.get("execution_id"),
            "result": result
        })
        
        return result
        
    except Exception as e:
        # Emit error event
        await webhook_manager.emit_event("graph.failed", {
            "execution_id": state.get("execution_id"),
            "error": str(e)
        })
        raise
```

### Real-time Processing Systems

```python
# src/realtime/processor.py
import asyncio
import websockets
import json
from typing import Dict, Set
from fastapi import WebSocket

class RealTimeGraphProcessor:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.graph_streams: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, graph_id: str):
        """Connect a client to real-time graph updates"""
        await websocket.accept()
        self.active_connections.add(websocket)
        
        if graph_id not in self.graph_streams:
            self.graph_streams[graph_id] = set()
        self.graph_streams[graph_id].add(websocket)
    
    def disconnect(self, websocket: WebSocket, graph_id: str):
        """Disconnect a client"""
        self.active_connections.discard(websocket)
        if graph_id in self.graph_streams:
            self.graph_streams[graph_id].discard(websocket)
    
    async def broadcast_update(self, graph_id: str, update: Dict[str, Any]):
        """Broadcast update to all connected clients for a graph"""
        if graph_id in self.graph_streams:
            message = json.dumps(update)
            disconnected = set()
            
            for websocket in self.graph_streams[graph_id]:
                try:
                    await websocket.send_text(message)
                except:
                    disconnected.add(websocket)
            
            # Clean up disconnected clients
            for ws in disconnected:
                self.disconnect(ws, graph_id)

# WebSocket endpoint
from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect

realtime_processor = RealTimeGraphProcessor()

@app.websocket("/ws/graphs/{graph_id}")
async def websocket_endpoint(websocket: WebSocket, graph_id: str):
    await realtime_processor.connect(websocket, graph_id)
    
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle client commands (pause, resume, cancel, etc.)
            await handle_client_command(graph_id, message)
            
    except WebSocketDisconnect:
        realtime_processor.disconnect(websocket, graph_id)
```

## Deployment Best Practices

1. **Containerization**: Use multi-stage builds, security scanning, and proper resource limits
2. **Load Balancing**: Implement health checks and intelligent routing
3. **State Management**: Choose appropriate persistence strategy based on requirements
4. **API Design**: Follow REST principles with proper error handling
5. **Real-time Updates**: Use WebSockets for live progress monitoring
6. **Monitoring**: Implement comprehensive logging and metrics collection
