# Rust Basics for Tauri Development

## Overview
This guide covers the essential Rust concepts you need to understand before diving into Tauri backend development. While you don't need to be a Rust expert, understanding these fundamentals will help you write effective Tauri commands and handle backend logic.

## Variables and Data Types

### Primitive Types

#### Integer Types
```rust
// Signed integers
let small_number: i8 = -128;      // 8-bit signed (-128 to 127)
let medium_number: i16 = -32_768; // 16-bit signed
let normal_number: i32 = -2_000_000; // 32-bit signed (default)
let big_number: i64 = -9_000_000_000; // 64-bit signed
let huge_number: i128 = -170_000_000_000_000_000_000; // 128-bit signed

// Unsigned integers
let positive_small: u8 = 255;     // 8-bit unsigned (0 to 255)
let positive_medium: u16 = 65_535; // 16-bit unsigned
let positive_normal: u32 = 4_000_000; // 32-bit unsigned
let positive_big: u64 = 18_000_000_000; // 64-bit unsigned
let positive_huge: u128 = 340_000_000_000_000_000_000; // 128-bit unsigned

// Architecture-dependent
let pointer_sized: isize = -1000; // Pointer-sized signed integer
let pointer_unsigned: usize = 1000; // Pointer-sized unsigned integer
```

#### Floating Point Types
```rust
let decimal: f32 = 3.14159;      // 32-bit floating point
let precise_decimal: f64 = 2.718281828; // 64-bit floating point (default)

// Scientific notation
let large_number: f64 = 1.23e10; // 12,300,000,000
let small_number: f64 = 1.23e-4; // 0.000123
```

#### Boolean Type
```rust
let is_rust_awesome: bool = true;
let is_learning_easy: bool = false;

// Boolean operations
let result = is_rust_awesome && !is_learning_easy; // true
```

#### Character Type
```rust
let letter: char = 'a';
let emoji: char = '😀';
let unicode: char = '\u{1F600}'; // Unicode escape sequence
```

### Compound Types

#### Tuples
```rust
// Creating tuples
let coordinates: (f64, f64) = (40.7128, -74.0060); // (latitude, longitude)
let mixed_tuple: (i32, bool, char) = (42, true, 'z');

// Accessing tuple elements
let (lat, lon) = coordinates; // Destructuring
let x = coordinates.0;        // Index access
let y = coordinates.1;

// Unit tuple (empty tuple)
let unit: () = ();
```

#### Arrays
```rust
// Fixed-size arrays
let numbers: [i32; 5] = [1, 2, 3, 4, 5];
let zeros: [i32; 3] = [0; 3]; // [0, 0, 0]

// Accessing array elements
let first = numbers[0];
let last = numbers[numbers.len() - 1];

// Array slicing
let slice = &numbers[1..4]; // [2, 3, 4]
```

### String vs &str

#### String Type (Owned)
```rust
// Creating String
let mut owned_string = String::new();
let from_literal = String::from("Hello, World!");
let to_string = "Hello".to_string();

// Modifying String
owned_string.push_str("Hello");
owned_string.push(' ');
owned_string.push_str("Tauri!");

// String methods
let length = owned_string.len();
let is_empty = owned_string.is_empty();
let contains_tauri = owned_string.contains("Tauri");
```

#### String Slice (&str) - Borrowed
```rust
// String literals are &str
let string_slice: &str = "This is immutable";

// Getting slice from String
let owned = String::from("Hello, Tauri!");
let slice: &str = &owned[7..12]; // "Tauri"

// Function that accepts string slice
fn print_message(msg: &str) {
    println!("{}", msg);
}

// Can pass both String and &str
print_message("literal");        // &str
print_message(&owned_string);    // &String -> &str
```

## Functions and Control Flow

### Function Definition and Parameters

#### Basic Function Syntax
```rust
// Function with no parameters and no return value
fn greet() {
    println!("Hello from Tauri!");
}

// Function with parameters
fn add_numbers(a: i32, b: i32) -> i32 {
    a + b // No semicolon = return value
}

// Function with explicit return
fn multiply(x: i32, y: i32) -> i32 {
    return x * y; // Explicit return with semicolon
}

// Function for Tauri command
#[tauri::command]
fn calculate_area(width: f64, height: f64) -> f64 {
    width * height
}
```

#### Advanced Function Features
```rust
// Function with multiple return types using Result
fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        Err("Cannot divide by zero".to_string())
    } else {
        Ok(a / b)
    }
}

// Function with optional parameters using Option
fn create_user(name: String, email: Option<String>) -> String {
    match email {
        Some(e) => format!("User: {}, Email: {}", name, e),
        None => format!("User: {}, No email provided", name),
    }
}
```

### If/Else Statements and Match Expressions

#### If/Else Statements
```rust
fn check_temperature(temp: i32) -> String {
    if temp > 30 {
        "Hot".to_string()
    } else if temp > 20 {
        "Warm".to_string()
    } else if temp > 10 {
        "Cool".to_string()
    } else {
        "Cold".to_string()
    }
}

// If as expression
fn get_status(is_online: bool) -> &'static str {
    if is_online { "Online" } else { "Offline" }
}
```

#### Match Expressions
```rust
// Basic match
fn describe_number(n: i32) -> String {
    match n {
        0 => "zero".to_string(),
        1 => "one".to_string(),
        2 | 3 => "two or three".to_string(), // Multiple patterns
        4..=10 => "between four and ten".to_string(), // Range
        _ => "something else".to_string(), // Default case
    }
}

// Match with enums
enum Status {
    Active,
    Inactive,
    Pending,
}

fn handle_status(status: Status) -> String {
    match status {
        Status::Active => "System is running".to_string(),
        Status::Inactive => "System is stopped".to_string(),
        Status::Pending => "System is starting".to_string(),
    }
}

// Match with destructuring
fn process_coordinates(point: (i32, i32)) -> String {
    match point {
        (0, 0) => "Origin".to_string(),
        (x, 0) => format!("On X-axis at {}", x),
        (0, y) => format!("On Y-axis at {}", y),
        (x, y) => format!("Point at ({}, {})", x, y),
    }
}
```

### Loops

#### For Loops
```rust
// Iterating over ranges
fn count_up() {
    for i in 1..=5 {
        println!("Count: {}", i); // 1, 2, 3, 4, 5
    }
}

// Iterating over collections
fn process_items() {
    let items = vec!["apple", "banana", "orange"];
    
    // Iterate over references
    for item in &items {
        println!("Item: {}", item);
    }
    
    // Iterate with index
    for (index, item) in items.iter().enumerate() {
        println!("Item {}: {}", index, item);
    }
}
```

#### While Loops
```rust
fn countdown() {
    let mut count = 5;
    
    while count > 0 {
        println!("{}!", count);
        count -= 1;
    }
    
    println!("Liftoff!");
}

// While let for pattern matching
fn process_optional_values() {
    let mut values = vec![Some(1), Some(2), None, Some(3)];
    
    while let Some(value) = values.pop() {
        match value {
            Some(n) => println!("Got: {}", n),
            None => println!("Got None"),
        }
    }
}
```

#### Loop (Infinite Loop)
```rust
fn event_loop() {
    let mut counter = 0;
    
    loop {
        counter += 1;
        
        if counter == 10 {
            break; // Exit the loop
        }
        
        if counter % 2 == 0 {
            continue; // Skip to next iteration
        }
        
        println!("Odd number: {}", counter);
    }
}

// Loop with return value
fn find_first_even() -> i32 {
    let numbers = vec![1, 3, 5, 8, 9, 12];
    
    loop {
        for &num in &numbers {
            if num % 2 == 0 {
                break num; // Return value from loop
            }
        }
        break -1; // Not found
    }
}
```

## Ownership and Borrowing Concepts

### Understanding Ownership Rules

#### The Three Rules of Ownership
1. Each value in Rust has a variable that's called its owner
2. There can only be one owner at a time
3. When the owner goes out of scope, the value will be dropped

```rust
fn ownership_examples() {
    // Rule 1: Each value has an owner
    let s1 = String::from("hello"); // s1 owns the string
    
    // Rule 2: Only one owner at a time
    let s2 = s1; // Ownership moved to s2, s1 is no longer valid
    // println!("{}", s1); // This would cause a compile error!
    println!("{}", s2); // This is fine
    
    // Rule 3: Value is dropped when owner goes out of scope
    {
        let s3 = String::from("temporary");
        // s3 owns the string
    } // s3 goes out of scope, string is automatically freed
}

// Functions and ownership
fn take_ownership(s: String) {
    println!("{}", s);
} // s goes out of scope and is dropped

fn give_ownership() -> String {
    String::from("transferred")
}

fn ownership_transfer_example() {
    let s1 = String::from("hello");
    take_ownership(s1); // s1's ownership is moved into function
    // println!("{}", s1); // Error! s1 is no longer valid
    
    let s2 = give_ownership(); // Function returns ownership to s2
    println!("{}", s2); // This works
}
```

### References and Borrowing

#### Immutable References
```rust
// Borrowing with immutable references
fn calculate_length(s: &String) -> usize {
    s.len() // We can read but not modify
}

fn borrowing_example() {
    let s1 = String::from("hello");
    let len = calculate_length(&s1); // Borrow s1
    println!("Length of '{}' is {}", s1, len); // s1 is still valid
}

// Multiple immutable references are allowed
fn multiple_immutable_refs() {
    let s = String::from("hello");
    let r1 = &s;
    let r2 = &s;
    println!("{} and {}", r1, r2); // This is fine
}
```

#### Mutable References
```rust
// Borrowing with mutable references
fn modify_string(s: &mut String) {
    s.push_str(", world!");
}

fn mutable_borrowing_example() {
    let mut s = String::from("hello");
    modify_string(&mut s);
    println!("{}", s); // "hello, world!"
}

// Borrowing rules demonstration
fn borrowing_rules() {
    let mut s = String::from("hello");
    
    // Only one mutable reference at a time
    let r1 = &mut s;
    // let r2 = &mut s; // Error! Cannot have two mutable references
    
    println!("{}", r1);
    
    // After r1 is no longer used, we can create another mutable reference
    let r2 = &mut s;
    println!("{}", r2);
}
```

### Lifetimes (Basic Understanding)

```rust
// Lifetime annotations ensure references are valid
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

fn lifetime_example() {
    let string1 = String::from("long string is long");
    let string2 = String::from("xyz");
    
    let result = longest(&string1, &string2);
    println!("The longest string is {}", result);
}

// Structs with lifetimes
struct ImportantExcerpt<'a> {
    part: &'a str,
}

fn struct_lifetime_example() {
    let novel = String::from("Call me Ishmael. Some years ago...");
    let first_sentence = novel.split('.').next().expect("Could not find a '.'");
    
    let excerpt = ImportantExcerpt {
        part: first_sentence,
    };
    
    println!("Excerpt: {}", excerpt.part);
}
```

## Basic Error Handling with Result<T, E>

### Using Result for Error Handling

#### Basic Result Usage
```rust
// Function that can fail
fn divide_numbers(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        Err("Division by zero is not allowed".to_string())
    } else {
        Ok(a / b)
    }
}

// Handling Result with match
fn handle_division_match() {
    let result = divide_numbers(10.0, 3.0);
    
    match result {
        Ok(value) => println!("Result: {}", value),
        Err(error) => println!("Error: {}", error),
    }
}

// Handling Result with if let
fn handle_division_if_let() {
    let result = divide_numbers(10.0, 0.0);
    
    if let Err(error) = result {
        println!("Operation failed: {}", error);
    }
}
```

### Pattern Matching with Result

```rust
// More complex error handling
#[derive(Debug)]
enum CalculationError {
    DivisionByZero,
    NegativeInput,
    Overflow,
}

fn advanced_calculation(a: f64, b: f64) -> Result<f64, CalculationError> {
    if b == 0.0 {
        return Err(CalculationError::DivisionByZero);
    }
    
    if a < 0.0 || b < 0.0 {
        return Err(CalculationError::NegativeInput);
    }
    
    let result = a / b;
    if result.is_infinite() {
        return Err(CalculationError::Overflow);
    }
    
    Ok(result)
}

fn handle_advanced_calculation() {
    let result = advanced_calculation(-5.0, 2.0);
    
    match result {
        Ok(value) => println!("Calculation successful: {}", value),
        Err(CalculationError::DivisionByZero) => {
            println!("Error: Cannot divide by zero");
        }
        Err(CalculationError::NegativeInput) => {
            println!("Error: Negative inputs are not allowed");
        }
        Err(CalculationError::Overflow) => {
            println!("Error: Result is too large");
        }
    }
}
```

### Propagating Errors with ? Operator

```rust
use std::fs::File;
use std::io::{self, Read};

// Manual error propagation
fn read_file_manual(filename: &str) -> Result<String, io::Error> {
    let mut file = match File::open(filename) {
        Ok(file) => file,
        Err(error) => return Err(error),
    };
    
    let mut contents = String::new();
    match file.read_to_string(&mut contents) {
        Ok(_) => Ok(contents),
        Err(error) => Err(error),
    }
}

// Using ? operator for cleaner code
fn read_file_with_question_mark(filename: &str) -> Result<String, io::Error> {
    let mut file = File::open(filename)?;  // Propagates error if file opening fails
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;   // Propagates error if reading fails
    Ok(contents)
}

// Tauri command example with error handling
#[tauri::command]
fn load_config_file(path: String) -> Result<String, String> {
    read_file_with_question_mark(&path)
        .map_err(|e| format!("Failed to read config file: {}", e))
}
```

## Cargo Package Manager

### Creating New Projects

#### Basic Project Creation
```bash
# Create a new binary project
cargo new my_tauri_backend
cd my_tauri_backend

# Create a new library project
cargo new my_library --lib

# Create with version control
cargo new my_project --vcs git

# Create without version control
cargo new my_project --vcs none
```

#### Project Structure
```
my_tauri_backend/
├── Cargo.toml          # Project metadata and dependencies
├── Cargo.lock          # Dependency lock file (auto-generated)
├── src/
│   └── main.rs         # Main entry point
└── target/             # Build artifacts (auto-generated)
    ├── debug/          # Debug builds
    └── release/        # Release builds
```

### Managing Dependencies in Cargo.toml

#### Basic Cargo.toml Structure
```toml
[package]
name = "my_tauri_backend"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <your.email@example.com>"]
description = "A Tauri backend application"
license = "MIT"
repository = "https://github.com/username/my_tauri_backend"

[dependencies]
# Basic dependency
serde = "1.0"

# Dependency with features
serde = { version = "1.0", features = ["derive"] }

# Tauri dependencies
tauri = { version = "1.0", features = ["api-all"] }
tokio = { version = "1.0", features = ["full"] }

# Optional dependencies
optional_dep = { version = "1.0", optional = true }

[dev-dependencies]
# Dependencies only for testing
tokio-test = "0.4"

[build-dependencies]
# Dependencies for build scripts
cc = "1.0"

[features]
# Feature flags
default = ["feature1"]
feature1 = []
feature2 = ["optional_dep"]

[[bin]]
# Additional binary targets
name = "helper"
path = "src/bin/helper.rs"
```

#### Common Tauri Dependencies
```toml
[dependencies]
# Core Tauri
tauri = { version = "1.0", features = ["api-all"] }

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Async runtime
tokio = { version = "1.0", features = ["full"] }

# HTTP client
reqwest = { version = "0.11", features = ["json"] }

# Database
sqlx = { version = "0.7", features = ["runtime-tokio-rustls", "sqlite"] }

# File system utilities
dirs = "5.0"

# Date and time
chrono = { version = "0.4", features = ["serde"] }

# Error handling
thiserror = "1.0"
anyhow = "1.0"

# Logging
log = "0.4"
env_logger = "0.10"

# Configuration
config = "0.13"

# UUID generation
uuid = { version = "1.0", features = ["v4"] }
```

### Building and Running Projects

#### Basic Cargo Commands
```bash
# Check code for errors without building
cargo check

# Build in debug mode (default)
cargo build

# Build in release mode (optimized)
cargo build --release

# Run the project
cargo run

# Run with arguments
cargo run -- --arg1 value1 --arg2 value2

# Run a specific binary
cargo run --bin helper

# Clean build artifacts
cargo clean
```

#### Testing with Cargo
```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_function_name

# Run tests in a specific module
cargo test module_name

# Run integration tests only
cargo test --test integration_tests

# Run doc tests
cargo test --doc
```

#### Tauri-Specific Commands
```bash
# Install Tauri CLI
cargo install tauri-cli

# Create new Tauri project
cargo tauri init

# Run Tauri in development mode
cargo tauri dev

# Build Tauri application
cargo tauri build

# Build for specific target
cargo tauri build --target x86_64-pc-windows-msvc

# Generate Tauri icons
cargo tauri icon path/to/icon.png
```

#### Cargo Configuration
```bash
# Update dependencies
cargo update

# Add a new dependency
cargo add serde --features derive

# Remove a dependency
cargo remove old_dependency

# Install binary from crates.io
cargo install ripgrep

# Check for outdated dependencies
cargo outdated

# Audit dependencies for security vulnerabilities
cargo audit
```

### Example: Complete Tauri Backend Setup

```rust
// src/main.rs
use tauri::Command;

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[tauri::command]
async fn calculate(a: f64, b: f64, operation: String) -> Result<f64, String> {
    match operation.as_str() {
        "add" => Ok(a + b),
        "subtract" => Ok(a - b),
        "multiply" => Ok(a * b),
        "divide" => {
            if b == 0.0 {
                Err("Cannot divide by zero".to_string())
            } else {
                Ok(a / b)
            }
        }
        _ => Err("Unknown operation".to_string()),
    }
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![greet, calculate])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

```toml
# Cargo.toml
[package]
name = "tauri-rust-basics"
version = "0.1.0"
edition = "2021"

[dependencies]
tauri = { version = "1.0", features = ["api-all"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1.0", features = ["full"] }

[build-dependencies]
tauri-build = { version = "1.0", features = [] }
```

## Practice Exercises

### Exercise 1: Basic Data Types
Create a function that takes various data types and returns a formatted string with information about each type.

### Exercise 2: Ownership Practice
Write functions that demonstrate ownership transfer, borrowing, and the rules of ownership.

### Exercise 3: Error Handling
Create a calculator that handles various error conditions using Result types.

### Exercise 4: Cargo Project
Set up a new Cargo project with dependencies and create a simple Tauri command.

## Next Steps

Once you're comfortable with these Rust basics:
1. Practice writing simple Tauri commands
2. Experiment with different data types in Tauri functions
3. Learn about more advanced Rust concepts like traits and generics
4. Explore the Tauri-specific Rust patterns and best practices

## Resources

### Essential Reading
- [The Rust Programming Language](https://doc.rust-lang.org/book/) - The official Rust book
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/) - Learn Rust with examples
- [Tauri Rust Examples](https://github.com/tauri-apps/tauri/tree/dev/examples) - Official Tauri examples

### Practice Platforms
- [Rustlings](https://github.com/rust-lang/rustlings) - Small exercises to get you used to reading and writing Rust code
- [Exercism Rust Track](https://exercism.org/tracks/rust) - Programming exercises with mentoring
- [Codewars Rust](https://www.codewars.com/kata/search/rust) - Coding challenges in Rust
