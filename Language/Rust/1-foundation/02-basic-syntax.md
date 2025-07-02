# Rust Basic Syntax

## Variables and Mutability

### Variable Declaration
```rust
fn main() {
    let x = 5;                    // Immutable by default
    let mut y = 10;               // Mutable variable
    
    // x = 6;                     // Error! x is immutable
    y = 15;                       // OK, y is mutable
    
    println!("x: {}, y: {}", x, y);
}
```

### Variable Shadowing
```rust
fn main() {
    let x = 5;
    let x = x + 1;                // Shadows previous x
    let x = x * 2;                // Shadows again
    
    println!("x: {}", x);         // Prints: x: 12
}
```

### Constants
```rust
const MAX_POINTS: u32 = 100_000;  // Must be type annotated
static LANGUAGE: &str = "Rust";   // Global static variable
```

## Data Types

### Scalar Types
```rust
fn main() {
    // Integers
    let small: i8 = -128;         // 8-bit signed
    let big: u64 = 1_000_000;     // 64-bit unsigned
    let default = 42;             // i32 by default
    
    // Floating point
    let pi: f64 = 3.14159;        // 64-bit (default)
    let e: f32 = 2.718;           // 32-bit
    
    // Boolean
    let is_rust_fun: bool = true;
    
    // Character
    let heart: char = '💖';       // Unicode scalar value
}
```

### Integer Types
| Type | Size | Range |
|------|------|-------|
| i8   | 8-bit | -128 to 127 |
| u8   | 8-bit | 0 to 255 |
| i32  | 32-bit | -2³¹ to 2³¹-1 |
| u32  | 32-bit | 0 to 2³²-1 |
| isize | arch | Pointer size |

### Compound Types
```rust
fn main() {
    // Tuples
    let tup: (i32, f64, char) = (42, 6.28, 'R');
    let (x, y, z) = tup;          // Destructuring
    let first = tup.0;            // Access by index
    
    // Arrays
    let arr: [i32; 5] = [1, 2, 3, 4, 5];
    let zeros = [0; 100];         // 100 zeros
    let first_element = arr[0];
}
```

## Functions

### Function Definition
```rust
fn main() {
    greet("Rust");
    let result = add(5, 3);
    println!("Result: {}", result);
}

fn greet(name: &str) {
    println!("Hello, {}!", name);
}

fn add(a: i32, b: i32) -> i32 {
    a + b                         // No semicolon = return value
}
```

### Function Features
```rust
// Multiple return values using tuples
fn swap(a: i32, b: i32) -> (i32, i32) {
    (b, a)
}

// Early return
fn divide(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        return 0.0;               // Early return with semicolon
    }
    a / b                         // Implicit return
}
```

## Control Flow

### if Expressions
```rust
fn main() {
    let number = 6;
    
    if number % 4 == 0 {
        println!("Divisible by 4");
    } else if number % 3 == 0 {
        println!("Divisible by 3");
    } else {
        println!("Not divisible by 3 or 4");
    }
    
    // if as expression
    let result = if number > 5 { "big" } else { "small" };
}
```

### Loops
```rust
fn main() {
    // Infinite loop
    let mut counter = 0;
    let result = loop {
        counter += 1;
        if counter == 10 {
            break counter * 2;     // Break with value
        }
    };
    
    // While loop
    let mut number = 3;
    while number != 0 {
        println!("{}!", number);
        number -= 1;
    }
    
    // For loop
    let arr = [10, 20, 30, 40, 50];
    for element in arr {
        println!("Value: {}", element);
    }
    
    // Range loops
    for number in 1..4 {          // 1, 2, 3
        println!("{}", number);
    }
    
    for number in (1..=5).rev() { // 5, 4, 3, 2, 1
        println!("{}", number);
    }
}
```

## Comments

### Comment Types
```rust
// Single line comment

/*
   Multi-line comment
   Can span multiple lines
*/

/// Documentation comment for items
/// Supports **markdown** formatting
fn documented_function() {
    //! Inner documentation comment
    println!("This function is documented");
}
```

## String Basics

### String Types
```rust
fn main() {
    // String literal (string slice)
    let greeting: &str = "Hello";
    
    // Owned String
    let mut message = String::from("Hello");
    message.push_str(", world!");
    
    // String methods
    let len = message.len();
    let is_empty = message.is_empty();
    let contains_world = message.contains("world");
}
```

## Print Macros

### println! Variants
```rust
fn main() {
    let name = "Rust";
    let version = 1.70;
    
    println!("Hello, world!");                    // Simple
    println!("Hello, {}!", name);                 // Positional
    println!("Version: {:.2}", version);          // Formatting
    println!("{name} version {version}");         // Named params
    
    print!("No newline");                         // No newline
    eprintln!("Error message");                   // To stderr
}
```

This covers the essential syntax you need to start writing Rust programs effectively.
