# Phase 3: Advanced Beginner (Weeks 13-20)

## Overview
This phase focuses on concurrency, systems programming concepts, and building practical applications. You'll learn to work with multiple threads, async programming, smart pointers, and create real-world projects.

## Learning Path

### Core Concepts
1. **Concurrency & Async Programming** - Understanding threads, channels, and async/await
2. **Smart Pointers** - Memory management with Box, Rc, Arc, and RefCell  
3. **Unsafe Rust & FFI** - Low-level programming and interfacing with other languages
4. **Memory Management** - Deep dive into Rust's memory model

### Practical Projects
1. **Multi-threaded File Processor** - Concurrent file processing with thread pools
2. **HTTP Client/Server** - Network programming and web services
3. **Database Interactions** - Working with SQL and NoSQL databases

## File Structure

```
3-advanced-beginner/
├── README.md                           # This file
├── 01-threads-channels-async.md        # Concurrency fundamentals
├── 02-smart-pointers.md               # Memory management tools
├── 03-unsafe-rust-ffi.md              # Low-level programming
├── 04-memory-management.md            # Memory model deep dive
├── 05-multi-threaded-file-processor.md # Project: File processing
├── 06-http-client-server.md           # Project: Web services
└── 07-database-interactions.md        # Project: Database operations
```

## Key Concepts Covered

### Concurrency & Async Programming
- **Threads**: Creating and managing OS threads
- **Channels**: Message passing between threads (mpsc, crossbeam)
- **Async/Await**: Non-blocking I/O with futures and tokio
- **Shared State**: Mutex, RwLock, and atomic operations
- **Thread Safety**: Send and Sync traits

### Smart Pointers
- **Box<T>**: Heap allocation and recursive types
- **Rc<T>**: Reference counting for shared ownership
- **Arc<T>**: Atomic reference counting for threading
- **RefCell<T>**: Interior mutability with runtime borrow checking
- **Weak<T>**: Breaking reference cycles

### Systems Programming
- **Unsafe Rust**: Raw pointers, unsafe functions, and blocks
- **FFI**: Interfacing with C libraries and other languages
- **Memory Layout**: Understanding how data is stored
- **Zero-Cost Abstractions**: Performance without overhead

### Practical Skills
- **Network Programming**: TCP/UDP sockets, HTTP protocols
- **Database Operations**: SQL and NoSQL database interactions
- **Error Handling**: Robust error management in complex systems
- **Performance**: Profiling and optimization techniques

## Projects Overview

### 1. Multi-threaded File Processor
Build a file processing system that can handle multiple files concurrently.

**Skills Learned:**
- Thread pool implementation
- Worker pattern with channels
- Progress tracking and reporting
- Concurrent error handling

**Key Features:**
- Process multiple files simultaneously
- Thread-safe progress reporting
- Configurable worker pool size
- Various processing operations (line counting, word frequency, etc.)

### 2. HTTP Client/Server
Create HTTP client and server implementations from scratch and with popular crates.

**Skills Learned:**
- TCP socket programming
- HTTP protocol implementation
- Request/response handling
- Middleware and routing

**Key Features:**
- Basic HTTP server with routing
- HTTP client for API consumption
- Async programming with tokio
- JSON serialization/deserialization

### 3. Database Interactions
Work with both SQL and NoSQL databases using popular Rust crates.

**Skills Learned:**
- Database connection management
- SQL query building and execution
- NoSQL document operations
- Transaction handling

**Key Features:**
- SQLite operations with rusqlite
- PostgreSQL with tokio-postgres
- MongoDB with mongodb driver
- Connection pooling with r2d2

## Prerequisites
- Completed Phase 1 (Foundation) and Phase 2 (Intermediate)
- Comfortable with ownership, borrowing, and lifetimes
- Experience with structs, enums, traits, and generics
- Basic understanding of error handling

## Learning Tips

### Practice Approach
1. **Start with Theory**: Read the concept files first
2. **Follow Examples**: Type out the code examples
3. **Build Projects**: Implement the practical projects
4. **Experiment**: Modify examples to understand behavior
5. **Debug Issues**: Practice troubleshooting concurrent code

### Common Challenges
- **Data Races**: Understanding thread safety and synchronization
- **Lifetime Issues**: Managing lifetimes in concurrent code
- **Async Complexity**: Grasping async/await and future combinators
- **Error Propagation**: Handling errors across thread boundaries

### Debugging Concurrent Code
- Use `println!` debugging strategically
- Learn to read thread panic messages
- Understand deadlock scenarios
- Use tools like `cargo test` for concurrent testing

## Tools and Crates Introduced

### Core Concurrency
- `std::thread` - Thread creation and management
- `std::sync::mpsc` - Multi-producer, single-consumer channels
- `tokio` - Async runtime and utilities
- `futures` - Future trait and combinators

### Network Programming
- `std::net` - TCP/UDP socket APIs
- `reqwest` - HTTP client
- `warp` / `axum` - Web frameworks
- `serde` - Serialization/deserialization

### Database Access
- `rusqlite` - SQLite database driver
- `tokio-postgres` - PostgreSQL async driver
- `mongodb` - MongoDB driver
- `r2d2` - Connection pooling

### Development Tools
- `crossbeam` - Advanced concurrency utilities
- `rayon` - Data parallelism
- `dashmap` - Concurrent hash map
- `parking_lot` - More efficient synchronization primitives

## Next Steps

After completing Phase 3, you should be able to:
- Build concurrent applications safely
- Work with async/await for I/O-intensive tasks
- Interface with databases effectively
- Create network services and clients
- Understand when and how to use unsafe code

### Preparing for Phase 4 (Advanced)
- Practice with larger projects combining multiple concepts
- Explore more advanced async patterns
- Learn about macro systems
- Dive deeper into performance optimization
- Start contributing to open source projects

## Common Patterns

### Worker Pool Pattern
```rust
// Thread pool for CPU-intensive tasks
let (sender, receiver) = mpsc::channel();
let receiver = Arc::new(Mutex::new(receiver));

for _ in 0..num_workers {
    let receiver = receiver.clone();
    thread::spawn(move || {
        while let Ok(task) = receiver.lock().unwrap().recv() {
            // Process task
        }
    });
}
```

### Async Error Handling
```rust
// Combining async operations with proper error handling
async fn fetch_and_process(url: &str) -> Result<ProcessedData, MyError> {
    let response = reqwest::get(url).await?;
    let data = response.json::<RawData>().await?;
    let processed = process_data(data).await?;
    Ok(processed)
}
```

### Database Transaction Pattern
```rust
// Safe database operations with transactions
let mut tx = conn.transaction()?;
tx.execute("INSERT INTO ...", params)?;
tx.execute("UPDATE ...", params)?;
tx.commit()?; // Only commit if all operations succeed
```

## Assessment

To validate your Phase 3 learning:

1. **Build a Multi-threaded Web Scraper**
   - Fetch multiple URLs concurrently
   - Parse and store data in a database
   - Implement rate limiting and error handling

2. **Create a Chat Server**
   - Handle multiple client connections
   - Broadcast messages between clients
   - Store message history in a database

3. **Implement a File Synchronization Tool**
   - Monitor file system changes
   - Sync files between directories
   - Handle conflicts and errors gracefully

These projects will test your understanding of concurrency, networking, and database operations in real-world scenarios.
