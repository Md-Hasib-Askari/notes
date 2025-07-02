# Advanced Async Patterns with Tokio

## Overview
Master advanced asynchronous programming patterns using Tokio runtime and ecosystem for high-performance, concurrent applications.

## Tokio Runtime Fundamentals

### Runtime Configuration
```rust
use tokio::runtime::{Builder, Runtime};
use std::time::Duration;

// Custom runtime configuration
fn create_custom_runtime() -> Runtime {
    Builder::new_multi_thread()
        .worker_threads(8)
        .thread_name("my-app")
        .thread_stack_size(3 * 1024 * 1024)
        .enable_all()
        .build()
        .unwrap()
}

// Runtime with time limits
#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    // Your async code here
    timeout_example().await;
}

async fn timeout_example() {
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        slow_operation()
    ).await;
    
    match result {
        Ok(value) => println!("Completed: {:?}", value),
        Err(_) => println!("Operation timed out"),
    }
}

async fn slow_operation() -> String {
    tokio::time::sleep(Duration::from_secs(3)).await;
    "Done!".to_string()
}
```

### Spawning and Task Management
```rust
use tokio::task::{JoinHandle, JoinSet};
use std::sync::Arc;

async fn task_management_examples() {
    // Basic task spawning
    let handle: JoinHandle<i32> = tokio::spawn(async {
        tokio::time::sleep(Duration::from_millis(100)).await;
        42
    });
    
    let result = handle.await.unwrap();
    println!("Task result: {}", result);
    
    // Spawning multiple tasks with JoinSet
    let mut set = JoinSet::new();
    
    for i in 0..5 {
        set.spawn(async move {
            tokio::time::sleep(Duration::from_millis(i * 100)).await;
            format!("Task {} completed", i)
        });
    }
    
    // Wait for all tasks
    while let Some(res) = set.join_next().await {
        match res {
            Ok(message) => println!("{}", message),
            Err(e) => eprintln!("Task error: {}", e),
        }
    }
}

// Task with cancellation
use tokio_util::sync::CancellationToken;

async fn cancellable_task() {
    let token = CancellationToken::new();
    let child_token = token.child_token();
    
    let task = tokio::spawn(async move {
        tokio::select! {
            _ = long_running_work() => {
                println!("Work completed");
            }
            _ = child_token.cancelled() => {
                println!("Task was cancelled");
            }
        }
    });
    
    // Cancel after 2 seconds
    tokio::time::sleep(Duration::from_secs(2)).await;
    token.cancel();
    
    task.await.unwrap();
}

async fn long_running_work() {
    tokio::time::sleep(Duration::from_secs(5)).await;
}
```

## Advanced Async Patterns

### Stream Processing
```rust
use tokio_stream::{self as stream, StreamExt};
use futures::stream::FuturesUnordered;

async fn stream_processing() {
    // Creating streams
    let mut numbers = stream::iter(1..=10);
    
    // Transform stream
    let doubled: Vec<_> = numbers
        .map(|x| x * 2)
        .filter(|&x| x > 5)
        .collect()
        .await;
    
    println!("Doubled and filtered: {:?}", doubled);
    
    // Async stream processing
    let urls = vec![
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/2",
        "https://httpbin.org/delay/3",
    ];
    
    let futures: FuturesUnordered<_> = urls
        .into_iter()
        .map(|url| async move {
            let client = reqwest::Client::new();
            client.get(url).send().await
        })
        .collect();
    
    let results: Vec<_> = futures
        .filter_map(|result| async {
            match result {
                Ok(response) => Some(response.status()),
                Err(_) => None,
            }
        })
        .collect()
        .await;
    
    println!("Response statuses: {:?}", results);
}

// Custom async stream
use std::pin::Pin;
use std::task::{Context, Poll};
use futures::Stream;

struct NumberStream {
    current: usize,
    max: usize,
}

impl NumberStream {
    fn new(max: usize) -> Self {
        NumberStream { current: 0, max }
    }
}

impl Stream for NumberStream {
    type Item = usize;
    
    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.current < self.max {
            let current = self.current;
            self.current += 1;
            Poll::Ready(Some(current))
        } else {
            Poll::Ready(None)
        }
    }
}

async fn custom_stream_example() {
    let mut stream = NumberStream::new(5);
    
    while let Some(number) = stream.next().await {
        println!("Number: {}", number);
    }
}
```

### Actor Pattern with Tokio
```rust
use tokio::sync::{mpsc, oneshot};
use std::collections::HashMap;

// Actor message types
#[derive(Debug)]
enum ActorMessage {
    Get {
        key: String,
        respond_to: oneshot::Sender<Option<String>>,
    },
    Set {
        key: String,
        value: String,
        respond_to: oneshot::Sender<()>,
    },
    Delete {
        key: String,
        respond_to: oneshot::Sender<bool>,
    },
}

// Actor struct
struct KeyValueActor {
    receiver: mpsc::Receiver<ActorMessage>,
    store: HashMap<String, String>,
}

impl KeyValueActor {
    fn new(receiver: mpsc::Receiver<ActorMessage>) -> Self {
        Self {
            receiver,
            store: HashMap::new(),
        }
    }
    
    async fn run(&mut self) {
        while let Some(msg) = self.receiver.recv().await {
            self.handle_message(msg).await;
        }
    }
    
    async fn handle_message(&mut self, msg: ActorMessage) {
        match msg {
            ActorMessage::Get { key, respond_to } => {
                let value = self.store.get(&key).cloned();
                let _ = respond_to.send(value);
            }
            ActorMessage::Set { key, value, respond_to } => {
                self.store.insert(key, value);
                let _ = respond_to.send(());
            }
            ActorMessage::Delete { key, respond_to } => {
                let existed = self.store.remove(&key).is_some();
                let _ = respond_to.send(existed);
            }
        }
    }
}

// Actor handle for external communication
#[derive(Clone)]
struct KeyValueHandle {
    sender: mpsc::Sender<ActorMessage>,
}

impl KeyValueHandle {
    fn new() -> Self {
        let (sender, receiver) = mpsc::channel(8);
        let mut actor = KeyValueActor::new(receiver);
        
        tokio::spawn(async move {
            actor.run().await;
        });
        
        Self { sender }
    }
    
    async fn get(&self, key: String) -> Option<String> {
        let (send, recv) = oneshot::channel();
        let msg = ActorMessage::Get {
            key,
            respond_to: send,
        };
        
        let _ = self.sender.send(msg).await;
        recv.await.unwrap_or(None)
    }
    
    async fn set(&self, key: String, value: String) {
        let (send, recv) = oneshot::channel();
        let msg = ActorMessage::Set {
            key,
            value,
            respond_to: send,
        };
        
        let _ = self.sender.send(msg).await;
        let _ = recv.await;
    }
    
    async fn delete(&self, key: String) -> bool {
        let (send, recv) = oneshot::channel();
        let msg = ActorMessage::Delete {
            key,
            respond_to: send,
        };
        
        let _ = self.sender.send(msg).await;
        recv.await.unwrap_or(false)
    }
}

async fn actor_example() {
    let kv = KeyValueHandle::new();
    
    kv.set("name".to_string(), "Alice".to_string()).await;
    
    if let Some(value) = kv.get("name".to_string()).await {
        println!("Name: {}", value);
    }
    
    let deleted = kv.delete("name".to_string()).await;
    println!("Deleted: {}", deleted);
}
```

### Connection Pooling and Resource Management
```rust
use tokio::sync::Semaphore;
use std::sync::Arc;

// Connection pool implementation
pub struct ConnectionPool<T> {
    connections: Arc<tokio::sync::Mutex<Vec<T>>>,
    semaphore: Arc<Semaphore>,
    factory: Box<dyn Fn() -> T + Send + Sync>,
}

impl<T> ConnectionPool<T>
where
    T: Send + 'static,
{
    pub fn new<F>(size: usize, factory: F) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
    {
        let connections = Arc::new(tokio::sync::Mutex::new(Vec::with_capacity(size)));
        let semaphore = Arc::new(Semaphore::new(size));
        
        Self {
            connections,
            semaphore,
            factory: Box::new(factory),
        }
    }
    
    pub async fn get(&self) -> PooledConnection<T> {
        let permit = self.semaphore.acquire().await.unwrap();
        
        let connection = {
            let mut connections = self.connections.lock().await;
            connections.pop().unwrap_or_else(|| (self.factory)())
        };
        
        PooledConnection {
            connection: Some(connection),
            pool: self.connections.clone(),
            _permit: permit,
        }
    }
}

pub struct PooledConnection<T> {
    connection: Option<T>,
    pool: Arc<tokio::sync::Mutex<Vec<T>>>,
    _permit: tokio::sync::SemaphorePermit<'_>,
}

impl<T> std::ops::Deref for PooledConnection<T> {
    type Target = T;
    
    fn deref(&self) -> &Self::Target {
        self.connection.as_ref().unwrap()
    }
}

impl<T> std::ops::DerefMut for PooledConnection<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.connection.as_mut().unwrap()
    }
}

impl<T> Drop for PooledConnection<T> {
    fn drop(&mut self) {
        if let Some(connection) = self.connection.take() {
            let pool = self.pool.clone();
            tokio::spawn(async move {
                let mut connections = pool.lock().await;
                connections.push(connection);
            });
        }
    }
}
```

### Rate Limiting and Backpressure
```rust
use tokio::time::{interval, Duration, Instant};
use std::sync::atomic::{AtomicU64, Ordering};

// Token bucket rate limiter
pub struct RateLimiter {
    tokens: Arc<AtomicU64>,
    max_tokens: u64,
    refill_rate: u64, // tokens per second
    last_refill: Arc<tokio::sync::Mutex<Instant>>,
}

impl RateLimiter {
    pub fn new(max_tokens: u64, refill_rate: u64) -> Self {
        Self {
            tokens: Arc::new(AtomicU64::new(max_tokens)),
            max_tokens,
            refill_rate,
            last_refill: Arc::new(tokio::sync::Mutex::new(Instant::now())),
        }
    }
    
    pub async fn acquire(&self, tokens: u64) -> Result<(), ()> {
        self.refill().await;
        
        let current = self.tokens.load(Ordering::Relaxed);
        if current >= tokens {
            self.tokens.fetch_sub(tokens, Ordering::Relaxed);
            Ok(())
        } else {
            Err(())
        }
    }
    
    pub async fn wait_for(&self, tokens: u64) {
        loop {
            if self.acquire(tokens).await.is_ok() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
    
    async fn refill(&self) {
        let mut last_refill = self.last_refill.lock().await;
        let now = Instant::now();
        let elapsed = now.duration_since(*last_refill);
        
        if elapsed >= Duration::from_millis(100) {
            let new_tokens = (elapsed.as_secs_f64() * self.refill_rate as f64) as u64;
            let current = self.tokens.load(Ordering::Relaxed);
            let new_total = std::cmp::min(current + new_tokens, self.max_tokens);
            
            self.tokens.store(new_total, Ordering::Relaxed);
            *last_refill = now;
        }
    }
}

// Backpressure-aware processor
async fn process_with_backpressure<T, F, Fut>(
    mut input: mpsc::Receiver<T>,
    processor: F,
    max_concurrent: usize,
) where
    F: Fn(T) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = ()> + Send,
    T: Send + 'static,
{
    let semaphore = Arc::new(Semaphore::new(max_concurrent));
    let processor = Arc::new(processor);
    
    while let Some(item) = input.recv().await {
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let processor = processor.clone();
        
        tokio::spawn(async move {
            processor(item).await;
            drop(permit); // Release permit when done
        });
    }
}
```

### Error Recovery and Circuit Breaker
```rust
use std::sync::atomic::{AtomicU64, AtomicBool};

#[derive(Debug, Clone)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

pub struct CircuitBreaker {
    state: Arc<tokio::sync::RwLock<CircuitState>>,
    failure_count: Arc<AtomicU64>,
    success_count: Arc<AtomicU64>,
    failure_threshold: u64,
    recovery_timeout: Duration,
    last_failure: Arc<tokio::sync::Mutex<Option<Instant>>>,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u64, recovery_timeout: Duration) -> Self {
        Self {
            state: Arc::new(tokio::sync::RwLock::new(CircuitState::Closed)),
            failure_count: Arc::new(AtomicU64::new(0)),
            success_count: Arc::new(AtomicU64::new(0)),
            failure_threshold,
            recovery_timeout,
            last_failure: Arc::new(tokio::sync::Mutex::new(None)),
        }
    }
    
    pub async fn call<F, T, E>(&self, f: F) -> Result<T, E>
    where
        F: std::future::Future<Output = Result<T, E>>,
    {
        // Check if circuit should transition to half-open
        self.check_recovery().await;
        
        let state = self.state.read().await.clone();
        
        match state {
            CircuitState::Open => {
                return Err(/* CircuitOpenError */);
            }
            _ => {}
        }
        
        match f.await {
            Ok(result) => {
                self.on_success().await;
                Ok(result)
            }
            Err(error) => {
                self.on_failure().await;
                Err(error)
            }
        }
    }
    
    async fn on_success(&self) {
        self.success_count.fetch_add(1, Ordering::Relaxed);
        
        let state = self.state.read().await.clone();
        if matches!(state, CircuitState::HalfOpen) {
            let mut state = self.state.write().await;
            *state = CircuitState::Closed;
            self.failure_count.store(0, Ordering::Relaxed);
        }
    }
    
    async fn on_failure(&self) {
        let failures = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
        
        if failures >= self.failure_threshold {
            let mut state = self.state.write().await;
            *state = CircuitState::Open;
            
            let mut last_failure = self.last_failure.lock().await;
            *last_failure = Some(Instant::now());
        }
    }
    
    async fn check_recovery(&self) {
        let state = self.state.read().await.clone();
        
        if matches!(state, CircuitState::Open) {
            let last_failure = self.last_failure.lock().await;
            
            if let Some(last) = *last_failure {
                if last.elapsed() >= self.recovery_timeout {
                    drop(last_failure);
                    let mut state = self.state.write().await;
                    *state = CircuitState::HalfOpen;
                }
            }
        }
    }
}
```

## Performance Optimization

### Async Profiling and Monitoring
```rust
use std::time::Instant;

// Task timing wrapper
async fn timed_task<F, T>(name: &str, future: F) -> T
where
    F: std::future::Future<Output = T>,
{
    let start = Instant::now();
    let result = future.await;
    let duration = start.elapsed();
    
    println!("Task '{}' completed in {:?}", name, duration);
    result
}

// Memory usage tracking
#[cfg(feature = "tracing")]
async fn traced_operation() {
    use tracing::{info, instrument};
    
    #[instrument]
    async fn expensive_operation(id: u64) -> String {
        info!("Starting expensive operation for {}", id);
        tokio::time::sleep(Duration::from_millis(100)).await;
        format!("Result for {}", id)
    }
    
    let result = expensive_operation(42).await;
    info!("Operation completed: {}", result);
}
```

## Key Takeaways

1. **Runtime Configuration**: Customize Tokio runtime for your needs
2. **Task Management**: Use JoinSet and cancellation tokens effectively
3. **Stream Processing**: Leverage async streams for data pipelines
4. **Actor Pattern**: Implement concurrent actors for state management
5. **Resource Pooling**: Manage expensive resources efficiently
6. **Rate Limiting**: Control resource consumption and prevent overload
7. **Error Recovery**: Implement circuit breakers and retry logic
8. **Performance**: Profile and monitor async applications
