# Error Handling with Result and Option

## Option Type

### Understanding Option
```rust
fn main() {
    let some_number = Some(5);
    let some_string = Some("a string");
    let absent_number: Option<i32> = None;
    
    // Option is enum with Some(T) and None variants
    match some_number {
        Some(value) => println!("Got a value: {}", value),
        None => println!("No value"),
    }
}
```

### Working with Option
```rust
fn divide(numerator: f64, denominator: f64) -> Option<f64> {
    if denominator == 0.0 {
        None
    } else {
        Some(numerator / denominator)
    }
}

fn main() {
    let result = divide(2.0, 3.0);
    
    match result {
        Some(x) => println!("Result: {}", x),
        None => println!("Cannot divide by 0"),
    }
    
    // Using if let
    if let Some(x) = divide(4.0, 2.0) {
        println!("Result: {}", x);
    }
}
```

### Option Methods
```rust
fn main() {
    let x: Option<i32> = Some(2);
    let y: Option<i32> = None;
    
    // unwrap - panics if None
    println!("x: {}", x.unwrap());
    
    // unwrap_or - provides default value
    println!("y: {}", y.unwrap_or(0));
    
    // unwrap_or_else - lazy default
    println!("y: {}", y.unwrap_or_else(|| 42));
    
    // expect - custom panic message
    println!("x: {}", x.expect("x should have a value"));
    
    // is_some / is_none
    if x.is_some() {
        println!("x has a value");
    }
    
    // map - transform the value if Some
    let doubled = x.map(|val| val * 2);
    println!("doubled: {:?}", doubled);
    
    // and_then - chain operations
    let result = x.and_then(|val| {
        if val > 0 {
            Some(val * 2)
        } else {
            None
        }
    });
}
```

## Result Type

### Understanding Result
```rust
use std::fs::File;
use std::io::ErrorKind;

fn main() {
    // Result<T, E> has Ok(T) and Err(E) variants
    let f = File::open("hello.txt");
    
    match f {
        Ok(file) => println!("File opened successfully"),
        Err(error) => println!("Failed to open file: {:?}", error),
    }
}
```

### Error Propagation with ?
```rust
use std::fs::File;
use std::io::{self, Read};

fn read_username_from_file() -> Result<String, io::Error> {
    let mut f = File::open("hello.txt")?;  // ? propagates error
    let mut s = String::new();
    f.read_to_string(&mut s)?;             // ? propagates error
    Ok(s)
}

// Even more concise
fn read_username_from_file_v2() -> Result<String, io::Error> {
    let mut s = String::new();
    File::open("hello.txt")?.read_to_string(&mut s)?;
    Ok(s)
}

// Using fs convenience function
fn read_username_from_file_v3() -> Result<String, io::Error> {
    std::fs::read_to_string("hello.txt")
}
```

### Custom Error Types
```rust
#[derive(Debug)]
enum MathError {
    DivisionByZero,
    NegativeLogarithm,
    NegativeSquareRoot,
}

fn divide(x: f64, y: f64) -> Result<f64, MathError> {
    if y == 0.0 {
        Err(MathError::DivisionByZero)
    } else {
        Ok(x / y)
    }
}

fn sqrt(x: f64) -> Result<f64, MathError> {
    if x < 0.0 {
        Err(MathError::NegativeSquareRoot)
    } else {
        Ok(x.sqrt())
    }
}

fn main() {
    match divide(4.0, 2.0) {
        Ok(result) => println!("Result: {}", result),
        Err(MathError::DivisionByZero) => println!("Cannot divide by zero"),
        Err(err) => println!("Math error: {:?}", err),
    }
}
```

### Result Methods
```rust
fn main() {
    let good_result: Result<i32, &str> = Ok(10);
    let bad_result: Result<i32, &str> = Err("Something went wrong");
    
    // unwrap - panics on Err
    println!("Good: {}", good_result.unwrap());
    
    // unwrap_or - provides default on Err
    println!("Bad: {}", bad_result.unwrap_or(0));
    
    // unwrap_or_else - lazy default
    println!("Bad: {}", bad_result.unwrap_or_else(|err| {
        println!("Error: {}", err);
        -1
    }));
    
    // expect - custom panic message
    println!("Good: {}", good_result.expect("Should be a number"));
    
    // is_ok / is_err
    if good_result.is_ok() {
        println!("Result is OK");
    }
    
    // map - transform Ok value
    let doubled = good_result.map(|val| val * 2);
    
    // map_err - transform Err value
    let better_error = bad_result.map_err(|err| format!("Error: {}", err));
}
```

## Combining Option and Result

### Converting Between Types
```rust
fn main() {
    let opt: Option<i32> = Some(42);
    let res: Result<i32, &str> = Ok(42);
    
    // Option to Result
    let opt_to_res: Result<i32, &str> = opt.ok_or("No value");
    
    // Result to Option
    let res_to_opt: Option<i32> = res.ok();
    
    // Transpose Option<Result> to Result<Option>
    let opt_res: Option<Result<i32, &str>> = Some(Ok(42));
    let res_opt: Result<Option<i32>, &str> = opt_res.transpose();
}
```

### Error Handling Patterns
```rust
use std::num::ParseIntError;

fn multiply(first_number_str: &str, second_number_str: &str) -> Result<i32, ParseIntError> {
    let first_number = first_number_str.parse::<i32>()?;
    let second_number = second_number_str.parse::<i32>()?;
    
    Ok(first_number * second_number)
}

fn main() {
    let result = multiply("10", "2");
    match result {
        Ok(n) => println!("Result: {}", n),
        Err(e) => println!("Error: {}", e),
    }
    
    // Multiple error handling
    let results = vec!["42", "not_a_number", "100"];
    for s in results {
        match s.parse::<i32>() {
            Ok(n) => println!("Parsed: {}", n),
            Err(_) => println!("Could not parse: {}", s),
        }
    }
}
```

### Early Returns and Guards
```rust
fn validate_age(age: &str) -> Result<u8, String> {
    let age: u8 = age.parse()
        .map_err(|_| "Age must be a number".to_string())?;
    
    if age > 150 {
        return Err("Age too high".to_string());
    }
    
    if age < 1 {
        return Err("Age too low".to_string());
    }
    
    Ok(age)
}

fn main() {
    let ages = vec!["25", "abc", "200", "0"];
    
    for age_str in ages {
        match validate_age(age_str) {
            Ok(age) => println!("Valid age: {}", age),
            Err(msg) => println!("Invalid age '{}': {}", age_str, msg),
        }
    }
}
```

### Main Function with Result
```rust
use std::error::Error;
use std::fs::File;

fn main() -> Result<(), Box<dyn Error>> {
    let f = File::open("hello.txt")?;
    
    // Do something with file
    
    Ok(())
}
```

These patterns form the foundation of robust error handling in Rust, making your programs more reliable and predictable.
