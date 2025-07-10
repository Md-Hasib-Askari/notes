# Understanding the Standard Library

## Core Collections Deep Dive

### Vec<T> Advanced Usage
```rust
use std::vec::Vec;

fn main() {
    let mut v = Vec::with_capacity(10); // Pre-allocate capacity
    
    // Efficient building patterns
    v.extend([1, 2, 3, 4, 5]);
    v.extend_from_slice(&[6, 7, 8]);
    
    // Deduplication
    v.sort();
    v.dedup();
    
    // Partitioning
    let (even, odd): (Vec<_>, Vec<_>) = v.into_iter()
        .partition(|&x| x % 2 == 0);
    
    println!("Even: {:?}, Odd: {:?}", even, odd);
    
    // Vec as stack
    let mut stack = vec![1, 2, 3];
    stack.push(4);
    while let Some(item) = stack.pop() {
        println!("Popped: {}", item);
    }
}
```

### String and &str Operations
```rust
fn main() {
    let mut s = String::new();
    
    // Building strings efficiently
    s.push_str("Hello");
    s.push(' ');
    s.push_str("World");
    
    // String formatting
    let formatted = format!("Number: {}, Float: {:.2}", 42, 3.14159);
    
    // String manipulation
    let text = "  Hello, World!  ";
    let cleaned = text.trim().to_lowercase();
    
    // String slicing and iteration
    for (i, ch) in cleaned.char_indices() {
        println!("Character '{}' at byte position {}", ch, i);
    }
    
    // Pattern matching
    let data = "name=John,age=30,city=NYC";
    let pairs: Vec<(&str, &str)> = data
        .split(',')
        .filter_map(|pair| {
            let mut parts = pair.split('=');
            Some((parts.next()?, parts.next()?))
        })
        .collect();
    
    println!("Pairs: {:?}", pairs);
}
```

## Iterator Patterns and Combinators

### Advanced Iterator Usage
```rust
fn main() {
    let numbers = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    
    // Chaining operations
    let result: Vec<String> = numbers
        .iter()
        .filter(|&&x| x % 2 == 0)      // Even numbers
        .map(|x| x * x)                 // Square them
        .filter(|&&x| x > 10)           // Greater than 10
        .map(|x| format!("#{}", x))     // Format as string
        .collect();
    
    println!("Results: {:?}", result);
    
    // Finding and searching
    let found = numbers.iter().find(|&&x| x > 5);
    let position = numbers.iter().position(|&x| x == 7);
    let any_even = numbers.iter().any(|&x| x % 2 == 0);
    let all_positive = numbers.iter().all(|&x| x > 0);
    
    // Folding and reducing
    let sum = numbers.iter().fold(0, |acc, &x| acc + x);
    let product = numbers.iter().reduce(|acc, &x| acc * x);
    
    // Grouping and partitioning
    use std::collections::HashMap;
    let grouped: HashMap<bool, Vec<i32>> = numbers
        .into_iter()
        .fold(HashMap::new(), |mut acc, x| {
            acc.entry(x % 2 == 0).or_default().push(x);
            acc
        });
}
```

### Custom Iterator Implementation
```rust
struct Counter {
    current: usize,
    max: usize,
}

impl Counter {
    fn new(max: usize) -> Counter {
        Counter { current: 0, max }
    }
}

impl Iterator for Counter {
    type Item = usize;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.max {
            let current = self.current;
            self.current += 1;
            Some(current)
        } else {
            None
        }
    }
}

fn main() {
    let counter = Counter::new(5);
    
    for num in counter {
        println!("Count: {}", num);
    }
    
    // Using iterator methods
    let sum: usize = Counter::new(10).sum();
    let collected: Vec<_> = Counter::new(5).collect();
}
```

## Option and Result Combinators

### Option Chaining Patterns
```rust
fn main() {
    let data = vec!["1", "2", "not_a_number", "4"];
    
    // Traditional approach
    for item in &data {
        match item.parse::<i32>() {
            Ok(num) => println!("Parsed: {}", num),
            Err(_) => println!("Failed to parse: {}", item),
        }
    }
    
    // Functional approach
    let parsed: Vec<i32> = data
        .iter()
        .filter_map(|s| s.parse().ok())
        .collect();
    
    // Option combinators
    let maybe_number = Some(42);
    let result = maybe_number
        .filter(|&x| x > 0)
        .map(|x| x * 2)
        .and_then(|x| if x < 100 { Some(x) } else { None })
        .unwrap_or(0);
    
    println!("Result: {}", result);
}
```

### Result Error Handling Patterns
```rust
use std::num::ParseIntError;

fn parse_numbers(input: &str) -> Result<Vec<i32>, ParseIntError> {
    input
        .split_whitespace()
        .map(|s| s.parse::<i32>())
        .collect()  // Collects into Result<Vec<i32>, ParseIntError>
}

fn main() {
    let input = "1 2 3 4 5";
    match parse_numbers(input) {
        Ok(numbers) => println!("Parsed: {:?}", numbers),
        Err(e) => println!("Parse error: {}", e),
    }
    
    // Result combinators
    let result = "42"
        .parse::<i32>()
        .map(|x| x * 2)
        .and_then(|x| if x > 0 { Ok(x) } else { Err("Negative result".into()) })
        .unwrap_or_else(|_| 0);
}
```

## Smart Pointers Overview

### Box<T> for Heap Allocation
```rust
fn main() {
    // Simple heap allocation
    let boxed_int = Box::new(5);
    println!("Boxed value: {}", boxed_int);
    
    // Recursive data structures
    #[derive(Debug)]
    enum List {
        Cons(i32, Box<List>),
        Nil,
    }
    
    use List::{Cons, Nil};
    
    let list = Cons(1, Box::new(Cons(2, Box::new(Cons(3, Box::new(Nil))))));
    println!("List: {:?}", list);
}
```

### Rc<T> for Shared Ownership
```rust
use std::rc::Rc;

#[derive(Debug)]
struct Node {
    value: i32,
    children: Vec<Rc<Node>>,
}

fn main() {
    let leaf = Rc::new(Node {
        value: 3,
        children: vec![],
    });
    
    let branch = Rc::new(Node {
        value: 5,
        children: vec![Rc::clone(&leaf)],
    });
    
    let root = Rc::new(Node {
        value: 10,
        children: vec![Rc::clone(&branch), Rc::clone(&leaf)],
    });
    
    println!("Reference count for leaf: {}", Rc::strong_count(&leaf));
}
```

## Path and Filesystem Utilities

### Path Manipulation
```rust
use std::path::{Path, PathBuf};
use std::env;

fn main() {
    let current_dir = env::current_dir().unwrap();
    println!("Current directory: {}", current_dir.display());
    
    // Building paths
    let mut path = PathBuf::from("/home/user");
    path.push("documents");
    path.push("file.txt");
    
    // Path components
    if let Some(parent) = path.parent() {
        println!("Parent: {}", parent.display());
    }
    
    if let Some(filename) = path.file_name() {
        println!("Filename: {:?}", filename);
    }
    
    if let Some(extension) = path.extension() {
        println!("Extension: {:?}", extension);
    }
    
    // Path operations
    let absolute = path.canonicalize().unwrap_or(path);
    let relative = Path::new("../config.toml");
    
    // Cross-platform path handling
    let cross_platform = Path::new("documents").join("subfolder").join("file.txt");
}
```

## Time and Duration

### Working with Time
```rust
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::thread;

fn main() {
    // Measuring elapsed time
    let start = Instant::now();
    
    // Simulate some work
    thread::sleep(Duration::from_millis(100));
    
    let duration = start.elapsed();
    println!("Time elapsed: {:?}", duration);
    
    // System time
    let now = SystemTime::now();
    let since_epoch = now.duration_since(UNIX_EPOCH).unwrap();
    println!("Seconds since epoch: {}", since_epoch.as_secs());
    
    // Duration arithmetic
    let one_hour = Duration::from_secs(3600);
    let thirty_minutes = Duration::from_secs(1800);
    let ninety_minutes = one_hour + thirty_minutes;
    
    println!("90 minutes = {} seconds", ninety_minutes.as_secs());
}
```

## Environment and Process

### Environment Variables and Arguments
```rust
use std::env;

fn main() {
    // Command line arguments
    let args: Vec<String> = env::args().collect();
    println!("Program: {}", args[0]);
    
    for (i, arg) in args.iter().skip(1).enumerate() {
        println!("Argument {}: {}", i + 1, arg);
    }
    
    // Environment variables
    match env::var("HOME") {
        Ok(home) => println!("Home directory: {}", home),
        Err(_) => println!("HOME not set"),
    }
    
    // Set environment variable
    env::set_var("MY_VAR", "my_value");
    
    // Current directory
    let current_dir = env::current_dir().unwrap();
    println!("Current directory: {}", current_dir.display());
    
    // Change directory
    env::set_current_dir("/tmp").unwrap();
}
```

This overview covers the most commonly used parts of Rust's extensive standard library.
