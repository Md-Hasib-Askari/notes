# Threads, Channels, and Async Programming

## Basic Threading

### Creating and Joining Threads
```rust
use std::thread;
use std::time::Duration;

fn main() {
    // Spawn a simple thread
    let handle = thread::spawn(|| {
        for i in 1..10 {
            println!("Thread: {}", i);
            thread::sleep(Duration::from_millis(1));
        }
    });
    
    // Main thread work
    for i in 1..5 {
        println!("Main: {}", i);
        thread::sleep(Duration::from_millis(1));
    }
    
    // Wait for thread to complete
    handle.join().unwrap();
}
```

### Moving Data into Threads
```rust
use std::thread;

fn main() {
    let v = vec![1, 2, 3];
    
    let handle = thread::spawn(move || {
        println!("Vector: {:?}", v);  // v is moved into closure
    });
    
    handle.join().unwrap();
    
    // Can't use v here anymore - it was moved
}
```

### Thread Builder for Configuration
```rust
use std::thread;

fn main() {
    let builder = thread::Builder::new()
        .name("worker".into())
        .stack_size(32 * 1024); // 32KB stack
    
    let handle = builder.spawn(|| {
        println!("Thread name: {:?}", thread::current().name());
    }).unwrap();
    
    handle.join().unwrap();
}
```

## Message Passing with Channels

### Basic Channel Communication
```rust
use std::sync::mpsc;
use std::thread;

fn main() {
    let (tx, rx) = mpsc::channel();
    
    thread::spawn(move || {
        let vals = vec![
            String::from("hi"),
            String::from("from"),
            String::from("the"),
            String::from("thread"),
        ];
        
        for val in vals {
            tx.send(val).unwrap();
            thread::sleep(Duration::from_secs(1));
        }
    });
    
    for received in rx {
        println!("Got: {}", received);
    }
}
```

### Multiple Producers
```rust
use std::sync::mpsc;
use std::thread;

fn main() {
    let (tx, rx) = mpsc::channel();
    let tx1 = tx.clone();
    
    thread::spawn(move || {
        let vals = vec![String::from("1"), String::from("2")];
        for val in vals {
            tx1.send(val).unwrap();
        }
    });
    
    thread::spawn(move || {
        let vals = vec![String::from("a"), String::from("b")];
        for val in vals {
            tx.send(val).unwrap();
        }
    });
    
    for received in rx {
        println!("Got: {}", received);
    }
}
```

### Bounded Channels (sync_channel)
```rust
use std::sync::mpsc;
use std::thread;

fn main() {
    let (tx, rx) = mpsc::sync_channel(2); // Buffer size of 2
    
    let producer = thread::spawn(move || {
        for i in 0..5 {
            println!("Sending: {}", i);
            tx.send(i).unwrap(); // Blocks when buffer is full
        }
    });
    
    thread::sleep(Duration::from_secs(2));
    
    for received in rx {
        println!("Received: {}", received);
        thread::sleep(Duration::from_millis(500));
    }
    
    producer.join().unwrap();
}
```

## Async Programming Basics

### Basic async/await
```rust
// Cargo.toml: tokio = { version = "1", features = ["full"] }
use tokio::time::{sleep, Duration};

async fn say_hello() {
    println!("Hello");
    sleep(Duration::from_secs(1)).await;
    println!("World");
}

async fn do_work(id: u32) -> u32 {
    println!("Starting work {}", id);
    sleep(Duration::from_secs(2)).await;
    println!("Finished work {}", id);
    id * 2
}

#[tokio::main]
async fn main() {
    say_hello().await;
    
    let result = do_work(5).await;
    println!("Result: {}", result);
}
```

### Concurrent async Tasks
```rust
use tokio::time::{sleep, Duration};

async fn fetch_data(id: u32) -> String {
    sleep(Duration::from_secs(1)).await;
    format!("Data {}", id)
}

#[tokio::main]
async fn main() {
    // Sequential execution (slow)
    let data1 = fetch_data(1).await;
    let data2 = fetch_data(2).await;
    println!("{}, {}", data1, data2);
    
    // Concurrent execution (fast)
    let (data3, data4) = tokio::join!(
        fetch_data(3),
        fetch_data(4)
    );
    println!("{}, {}", data3, data4);
    
    // With error handling
    let results = tokio::try_join!(
        fetch_data(5),
        fetch_data(6)
    );
    
    match results {
        Ok((d1, d2)) => println!("{}, {}", d1, d2),
        Err(e) => println!("Error: {}", e),
    }
}
```

### Spawning Tasks
```rust
use tokio::time::{sleep, Duration};

async fn background_task(name: &str) {
    for i in 0..5 {
        println!("{}: {}", name, i);
        sleep(Duration::from_millis(100)).await;
    }
}

#[tokio::main]
async fn main() {
    // Spawn tasks to run concurrently
    let task1 = tokio::spawn(background_task("Task1"));
    let task2 = tokio::spawn(background_task("Task2"));
    
    // Do other work
    sleep(Duration::from_millis(250)).await;
    println!("Main thread work");
    
    // Wait for tasks to complete
    let _ = tokio::join!(task1, task2);
}
```

### Async Channels
```rust
use tokio::sync::mpsc;
use tokio::time::{sleep, Duration};

async fn producer(tx: mpsc::Sender<i32>) {
    for i in 0..5 {
        println!("Sending: {}", i);
        if tx.send(i).await.is_err() {
            println!("Receiver dropped");
            break;
        }
        sleep(Duration::from_millis(500)).await;
    }
}

async fn consumer(mut rx: mpsc::Receiver<i32>) {
    while let Some(value) = rx.recv().await {
        println!("Received: {}", value);
        sleep(Duration::from_millis(200)).await;
    }
}

#[tokio::main]
async fn main() {
    let (tx, rx) = mpsc::channel(32);
    
    tokio::join!(
        producer(tx),
        consumer(rx)
    );
}
```

## Selecting Between Futures

### Using tokio::select!
```rust
use tokio::time::{sleep, Duration, timeout};
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    let (tx, mut rx) = mpsc::channel(10);
    
    // Spawn a task that sends data
    tokio::spawn(async move {
        sleep(Duration::from_secs(2)).await;
        tx.send("Hello").await.unwrap();
    });
    
    // Select between multiple futures
    tokio::select! {
        msg = rx.recv() => {
            match msg {
                Some(msg) => println!("Received: {}", msg),
                None => println!("Channel closed"),
            }
        }
        _ = sleep(Duration::from_secs(1)) => {
            println!("Timeout occurred");
        }
    }
    
    // Using timeout wrapper
    let result = timeout(Duration::from_secs(1), rx.recv()).await;
    match result {
        Ok(Some(msg)) => println!("Got message: {}", msg),
        Ok(None) => println!("Channel closed"),
        Err(_) => println!("Timeout"),
    }
}
```

### Error Handling in Async
```rust
use std::error::Error;
use std::fmt;

#[derive(Debug)]
struct CustomError(String);

impl fmt::Display for CustomError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Custom error: {}", self.0)
    }
}

impl Error for CustomError {}

async fn fallible_operation(should_fail: bool) -> Result<String, CustomError> {
    if should_fail {
        Err(CustomError("Operation failed".to_string()))
    } else {
        Ok("Success".to_string())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let result = fallible_operation(false).await?;
    println!("Result: {}", result);
    
    // Handle multiple potential errors
    let results = tokio::try_join!(
        fallible_operation(false),
        fallible_operation(false)
    )?;
    
    println!("Both succeeded: {:?}", results);
    Ok(())
}
```

This covers the essential patterns for concurrent and asynchronous programming in Rust.
