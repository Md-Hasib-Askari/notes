# Modules and Crate Organization

## Module System Basics

### Defining Modules
```rust
// src/main.rs or src/lib.rs
mod front_of_house {
    mod hosting {
        fn add_to_waitlist() {}
        fn seat_at_table() {}
    }

    mod serving {
        fn take_order() {}
        fn serve_order() {}
        fn take_payment() {}
    }
}
```

### Module Tree Structure
```
crate
 └── front_of_house
     ├── hosting
     │   ├── add_to_waitlist
     │   └── seat_at_table
     └── serving
         ├── take_order
         ├── serve_order
         └── take_payment
```

### Privacy Rules
```rust
mod front_of_house {
    pub mod hosting {              // Public module
        pub fn add_to_waitlist() {} // Public function
        fn seat_at_table() {}       // Private function
    }
    
    mod serving {                   // Private module
        pub fn take_order() {}      // Public function (but module is private)
    }
}

pub fn eat_at_restaurant() {
    // Absolute path
    crate::front_of_house::hosting::add_to_waitlist();
    
    // Relative path
    front_of_house::hosting::add_to_waitlist();
}
```

### Using `use` to Bring Paths into Scope
```rust
mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {}
    }
}

use crate::front_of_house::hosting;

pub fn eat_at_restaurant() {
    hosting::add_to_waitlist();  // Shorter path
}

// Bringing function directly into scope (less idiomatic)
use crate::front_of_house::hosting::add_to_waitlist;

pub fn eat_at_restaurant_v2() {
    add_to_waitlist();
}
```

### Re-exporting with `pub use`
```rust
mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {}
    }
}

pub use crate::front_of_house::hosting;  // Re-export for external use

pub fn eat_at_restaurant() {
    hosting::add_to_waitlist();
}
```

## File-based Module Organization

### Separating Modules into Files
```rust
// src/lib.rs
mod front_of_house;  // Tells Rust to load from src/front_of_house.rs

pub use crate::front_of_house::hosting;

pub fn eat_at_restaurant() {
    hosting::add_to_waitlist();
}
```

```rust
// src/front_of_house.rs
pub mod hosting {
    pub fn add_to_waitlist() {}
}
```

### Directory-based Modules
```
src/
├── lib.rs
├── front_of_house/
│   ├── mod.rs        # Module definition
│   ├── hosting.rs    # Submodule
│   └── serving.rs    # Submodule
```

```rust
// src/front_of_house/mod.rs
pub mod hosting;
pub mod serving;
```

```rust
// src/front_of_house/hosting.rs
pub fn add_to_waitlist() {}
pub fn seat_at_table() {}
```

## Advanced Use Patterns

### Multiple Items and Aliases
```rust
use std::collections::{HashMap, BTreeMap, HashSet};
use std::io::{self, Write};               // Bring std::io and Write
use std::io::Result as IoResult;          // Alias to avoid naming conflicts

fn main() -> IoResult<()> {
    let mut map = HashMap::new();
    map.insert("key", "value");
    
    io::stdout().write_all(b"Hello")?;
    Ok(())
}
```

### Glob Operator
```rust
use std::collections::*;  // Brings all public items

// Use sparingly, can make code less clear
```

### External Crates
```rust
// Cargo.toml
// [dependencies]
// rand = "0.8"

use rand::Rng;  // External crate

fn main() {
    let secret_number = rand::thread_rng().gen_range(1..=100);
    println!("Secret number: {}", secret_number);
}
```

## Workspace Organization

### Multi-crate Workspace
```toml
# Cargo.toml (workspace root)
[workspace]
members = [
    "adder",
    "add_one",
]
```

```toml
# adder/Cargo.toml
[package]
name = "adder"
version = "0.1.0"
edition = "2021"

[dependencies]
add_one = { path = "../add_one" }
```

```toml
# add_one/Cargo.toml
[package]
name = "add_one"
version = "0.1.0"
edition = "2021"
```

### Workspace Structure
```
workspace/
├── Cargo.toml
├── adder/
│   ├── Cargo.toml
│   └── src/
│       └── main.rs
├── add_one/
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs
└── target/
```

## Common Module Patterns

### Nested Modules Example
```rust
// src/lib.rs
pub mod network {
    pub mod client {
        pub fn connect() {}
    }
    
    pub mod server {
        pub fn listen() {}
    }
}

pub mod utils {
    pub mod math {
        pub fn add(a: i32, b: i32) -> i32 {
            a + b
        }
    }
    
    pub mod string {
        pub fn capitalize(s: &str) -> String {
            let mut chars = s.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            }
        }
    }
}
```

### Tests Module
```rust
// src/lib.rs
pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;  // Bring parent module items into scope

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
```

### Conditional Compilation
```rust
#[cfg(target_os = "windows")]
mod windows_specific {
    pub fn do_windows_thing() {}
}

#[cfg(target_os = "linux")]
mod linux_specific {
    pub fn do_linux_thing() {}
}

#[cfg(feature = "advanced")]
mod advanced_features {
    pub fn advanced_function() {}
}
```

## Best Practices

### Module Organization Guidelines
1. **Start with single file**: Begin with `main.rs` or `lib.rs`
2. **Split when it grows**: Move to separate files when modules get large
3. **Group related functionality**: Keep related functions and types together
4. **Use clear names**: Module names should reflect their purpose
5. **Consider the public API**: Only expose what users need

### Naming Conventions
```rust
mod snake_case_module_names;
use std::collections::HashMap;  // External types stay as-is
use crate::my_module::MyStruct; // Your types follow Rust conventions

const SCREAMING_SNAKE_CASE: i32 = 42;
static GLOBAL_COUNTER: i32 = 0;

struct PascalCaseStruct;
enum PascalCaseEnum;
trait PascalCaseTrait;

fn snake_case_function() {}
```

### Common Library Structure
```
src/
├── lib.rs          # Main library entry point
├── types.rs        # Common types and structs
├── error.rs        # Error types
├── utils.rs        # Utility functions
├── config.rs       # Configuration
├── parser/         # Complex functionality in subdirectory
│   ├── mod.rs
│   ├── lexer.rs
│   └── ast.rs
└── tests/          # Integration tests
    └── common/
        └── mod.rs
```

This modular approach helps organize code as projects grow and makes collaboration easier.
