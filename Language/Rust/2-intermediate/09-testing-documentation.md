# Writing Tests and Documentation

## Unit Testing Fundamentals

### Basic Test Structure
```rust
// src/lib.rs
pub fn add(left: usize, right: usize) -> usize {
    left + right
}

pub fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        Err("Division by zero".to_string())
    } else {
        Ok(a / b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 3), 5);
        assert_eq!(add(0, 0), 0);
        assert_eq!(add(100, 1), 101);
    }

    #[test]
    fn test_divide_success() {
        let result = divide(10.0, 2.0).unwrap();
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_divide_by_zero() {
        let result = divide(10.0, 0.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Division by zero");
    }

    #[test]
    #[should_panic]
    fn test_panic_behavior() {
        panic!("This test expects a panic");
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn test_specific_panic() {
        let _result = 10 / 0; // This will panic
    }
}
```

### Test Assertions
```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_assertions() {
        // Equality assertions
        assert_eq!(2 + 2, 4);
        assert_ne!(2 + 2, 5);
        
        // Boolean assertions
        assert!(true);
        assert!(!false);
        
        // Floating point comparisons
        let result = 0.1 + 0.2;
        assert!((result - 0.3).abs() < f64::EPSILON);
        
        // Custom messages
        assert_eq!(2 + 2, 4, "Math is broken!");
        assert!(2 > 1, "Expected {} to be greater than {}", 2, 1);
    }
    
    #[test]
    fn test_result_assertions() -> Result<(), Box<dyn std::error::Error>> {
        let result = "42".parse::<i32>()?;
        assert_eq!(result, 42);
        Ok(())
    }
}
```

## Integration Testing

### Integration Test Structure
```rust
// tests/integration_test.rs
use my_crate;

#[test]
fn test_public_api() {
    let result = my_crate::add(5, 10);
    assert_eq!(result, 15);
}

#[test]
fn test_complex_workflow() {
    // Test multiple components working together
    let data = vec![1, 2, 3, 4, 5];
    let processed = my_crate::process_data(&data);
    assert_eq!(processed.len(), 5);
}
```

### Test Organization
```rust
// tests/common/mod.rs
pub fn setup() -> TestData {
    TestData {
        values: vec![1, 2, 3, 4, 5],
    }
}

pub struct TestData {
    pub values: Vec<i32>,
}

// tests/my_test.rs
mod common;

#[test]
fn test_with_setup() {
    let test_data = common::setup();
    assert_eq!(test_data.values.len(), 5);
}
```

## Advanced Testing Patterns

### Testing with Mock Data
```rust
pub trait DataSource {
    fn get_data(&self) -> Result<Vec<String>, String>;
}

pub struct FileDataSource {
    path: String,
}

impl DataSource for FileDataSource {
    fn get_data(&self) -> Result<Vec<String>, String> {
        std::fs::read_to_string(&self.path)
            .map(|content| content.lines().map(String::from).collect())
            .map_err(|e| e.to_string())
    }
}

pub fn process_data<T: DataSource>(source: &T) -> Result<usize, String> {
    let data = source.get_data()?;
    Ok(data.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    struct MockDataSource {
        data: Vec<String>,
        should_error: bool,
    }
    
    impl DataSource for MockDataSource {
        fn get_data(&self) -> Result<Vec<String>, String> {
            if self.should_error {
                Err("Mock error".to_string())
            } else {
                Ok(self.data.clone())
            }
        }
    }
    
    #[test]
    fn test_process_data_success() {
        let mock = MockDataSource {
            data: vec!["line1".to_string(), "line2".to_string()],
            should_error: false,
        };
        
        let result = process_data(&mock).unwrap();
        assert_eq!(result, 2);
    }
    
    #[test]
    fn test_process_data_error() {
        let mock = MockDataSource {
            data: vec![],
            should_error: true,
        };
        
        let result = process_data(&mock);
        assert!(result.is_err());
    }
}
```

### Property-Based Testing with Proptest
```toml
# Cargo.toml
[dev-dependencies]
proptest = "1.0"
```

```rust
#[cfg(test)]
mod proptests {
    use proptest::prelude::*;
    use super::*;
    
    proptest! {
        #[test]
        fn test_add_commutative(a in 0..1000u32, b in 0..1000u32) {
            assert_eq!(add(a as usize, b as usize), add(b as usize, a as usize));
        }
        
        #[test]
        fn test_division_multiply_inverse(a in 1.0..1000.0f64, b in 1.0..1000.0f64) {
            let divided = divide(a, b).unwrap();
            let multiplied = divided * b;
            assert!((multiplied - a).abs() < 0.0001);
        }
    }
}
```

## Documentation

### Doc Comments and Examples
```rust
/// Adds two numbers together.
/// 
/// # Examples
/// 
/// ```
/// use my_crate::add;
/// 
/// let result = add(2, 3);
/// assert_eq!(result, 5);
/// ```
/// 
/// # Panics
/// 
/// This function will never panic.
/// 
/// # Errors
/// 
/// This function doesn't return errors.
pub fn add(left: usize, right: usize) -> usize {
    left + right
}

/// A calculator that can perform basic operations.
/// 
/// # Examples
/// 
/// ```
/// use my_crate::Calculator;
/// 
/// let mut calc = Calculator::new();
/// calc.add(5);
/// calc.multiply(2);
/// assert_eq!(calc.result(), 10);
/// ```
pub struct Calculator {
    value: f64,
}

impl Calculator {
    /// Creates a new calculator with initial value of 0.
    /// 
    /// # Examples
    /// 
    /// ```
    /// use my_crate::Calculator;
    /// 
    /// let calc = Calculator::new();
    /// assert_eq!(calc.result(), 0.0);
    /// ```
    pub fn new() -> Self {
        Self { value: 0.0 }
    }
    
    /// Adds a value to the current result.
    /// 
    /// # Arguments
    /// 
    /// * `value` - The value to add
    /// 
    /// # Examples
    /// 
    /// ```
    /// use my_crate::Calculator;
    /// 
    /// let mut calc = Calculator::new();
    /// calc.add(5.0);
    /// assert_eq!(calc.result(), 5.0);
    /// ```
    pub fn add(&mut self, value: f64) {
        self.value += value;
    }
    
    /// Returns the current result.
    pub fn result(&self) -> f64 {
        self.value
    }
}
```

### Module-Level Documentation
```rust
//! # My Crate
//! 
//! This crate provides utilities for mathematical operations and calculations.
//! 
//! ## Quick Start
//! 
//! ```
//! use my_crate::{add, Calculator};
//! 
//! // Simple addition
//! let sum = add(2, 3);
//! 
//! // Using calculator
//! let mut calc = Calculator::new();
//! calc.add(10.0);
//! println!("Result: {}", calc.result());
//! ```
//! 
//! ## Features
//! 
//! - Basic arithmetic operations
//! - Calculator with state
//! - Error handling for edge cases

pub mod calculator;
pub mod math;

pub use calculator::Calculator;
pub use math::{add, divide};
```

### Documentation Tests
```rust
/// Parses a string into a number.
/// 
/// # Examples
/// 
/// Basic usage:
/// 
/// ```
/// use my_crate::parse_number;
/// 
/// let result = parse_number("42").unwrap();
/// assert_eq!(result, 42);
/// ```
/// 
/// Error handling:
/// 
/// ```
/// use my_crate::parse_number;
/// 
/// let result = parse_number("not_a_number");
/// assert!(result.is_err());
/// ```
/// 
/// ```should_panic
/// use my_crate::parse_number;
/// 
/// // This will panic
/// parse_number("not_a_number").unwrap();
/// ```
/// 
/// ```ignore
/// // This example is ignored during testing
/// use my_crate::parse_number;
/// 
/// let result = parse_number("42");
/// // Some complex setup that doesn't work in doc tests
/// ```
pub fn parse_number(s: &str) -> Result<i32, std::num::ParseIntError> {
    s.parse()
}
```

## Running Tests

### Test Commands
```bash
# Run all tests
cargo test

# Run specific test
cargo test test_add

# Run tests with pattern matching
cargo test divide

# Run ignored tests
cargo test -- --ignored

# Run tests with output
cargo test -- --nocapture

# Run doc tests only
cargo test --doc

# Run integration tests only
cargo test --test integration_test

# Run with multiple threads
cargo test -- --test-threads=1
```

### Test Configuration
```rust
// Cargo.toml
[package]
name = "my_crate"
version = "0.1.0"

[dev-dependencies]
proptest = "1.0"
criterion = "0.4"

[[bench]]
name = "my_benchmark"
harness = false
```

This comprehensive approach to testing and documentation ensures code quality and helps other developers understand and use your code effectively.
