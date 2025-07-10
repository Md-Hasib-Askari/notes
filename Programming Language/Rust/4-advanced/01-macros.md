# Macros - Declarative and Procedural

## Overview
Macros allow you to write code that writes other code (metaprogramming). Rust has two types: declarative macros (`macro_rules!`) and procedural macros.

## Declarative Macros (`macro_rules!`)

### Basic Syntax
```rust
macro_rules! macro_name {
    (pattern) => {
        expansion
    };
}
```

### Simple Examples
```rust
// Basic replacement macro
macro_rules! say_hello {
    () => {
        println!("Hello, World!");
    };
}

// Macro with parameters
macro_rules! create_function {
    ($func_name:ident) => {
        fn $func_name() {
            println!("You called {:?}()", stringify!($func_name));
        }
    };
}

// Usage
create_function!(foo);
create_function!(bar);

fn main() {
    say_hello!();
    foo();
    bar();
}
```

### Pattern Matching in Macros
```rust
macro_rules! test {
    // Match single expression
    ($left:expr; and $right:expr) => {
        println!("{:?} and {:?} is {:?}",
                 stringify!($left),
                 stringify!($right),
                 $left && $right)
    };
    
    // Match multiple expressions
    ($left:expr; or $right:expr) => {
        println!("{:?} or {:?} is {:?}",
                 stringify!($left),
                 stringify!($right),
                 $left || $right)
    };
}

// Usage
test!(1i32 + 1 == 2i32; and 2i32 * 2 == 4i32);
test!(true; or false);
```

### Repetition Patterns
```rust
macro_rules! find_min {
    ($x:expr) => ($x);
    ($x:expr, $($y:expr),+) => (
        std::cmp::min($x, find_min!($($y),+))
    )
}

// Create a vector with initial values
macro_rules! vec_of_strings {
    ($($x:expr),*) => {
        {
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push($x.to_string());
            )*
            temp_vec
        }
    };
}

fn main() {
    println!("{}", find_min!(1u32, 2, 3, 4));
    
    let v = vec_of_strings!["hello", "world", "rust"];
    println!("{:?}", v);
}
```

### Advanced Declarative Macros
```rust
// HashMap creation macro
macro_rules! hashmap {
    ($($key:expr => $val:expr),*) => {
        {
            let mut map = std::collections::HashMap::new();
            $(
                map.insert($key, $val);
            )*
            map
        }
    };
}

// Builder pattern macro
macro_rules! builder {
    (struct $name:ident {
        $($field:ident: $field_type:ty),*
    }) => {
        #[derive(Debug, Default)]
        struct $name {
            $($field: Option<$field_type>),*
        }
        
        impl $name {
            fn new() -> Self {
                Self::default()
            }
            
            $(
                fn $field(mut self, $field: $field_type) -> Self {
                    self.$field = Some($field);
                    self
                }
            )*
            
            fn build(self) -> Result<Built$name, String> {
                Ok(Built$name {
                    $($field: self.$field.ok_or(concat!(
                        "Field '", stringify!($field), "' is required"
                    ))?),*
                })
            }
        }
        
        #[derive(Debug)]
        struct Built$name {
            $($field: $field_type),*
        }
    };
}

// Usage
builder! {
    struct User {
        name: String,
        age: u32,
        email: String
    }
}

fn main() {
    let user = User::new()
        .name("Alice".to_string())
        .age(30)
        .email("alice@example.com".to_string())
        .build()
        .unwrap();
    
    println!("{:?}", user);
}
```

## Procedural Macros

### Setup for Procedural Macros
```toml
# Cargo.toml for proc macro crate
[lib]
proc-macro = true

[dependencies]
syn = { version = "2.0", features = ["full"] }
quote = "1.0"
proc-macro2 = "1.0"
```

### Function-like Procedural Macros
```rust
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, LitStr};

#[proc_macro]
pub fn make_answer(_item: TokenStream) -> TokenStream {
    "fn answer() -> u32 { 42 }".parse().unwrap()
}

#[proc_macro]
pub fn sql(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as LitStr);
    let sql = input.value();
    
    let output = quote! {
        {
            // Validate SQL at compile time
            println!("Executing SQL: {}", #sql);
            #sql
        }
    };
    
    output.into()
}
```

### Derive Macros
```rust
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Data, Fields};

#[proc_macro_derive(Builder)]
pub fn derive_builder(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let builder_name = format!("{}Builder", name);
    let builder_ident = syn::Ident::new(&builder_name, name.span());
    
    let fields = match &input.data {
        Data::Struct(data) => {
            match &data.fields {
                Fields::Named(fields) => &fields.named,
                _ => panic!("Builder only supports named fields"),
            }
        }
        _ => panic!("Builder only supports structs"),
    };
    
    let field_names: Vec<_> = fields.iter()
        .map(|f| f.ident.as_ref().unwrap())
        .collect();
    
    let field_types: Vec<_> = fields.iter()
        .map(|f| &f.ty)
        .collect();
    
    let expanded = quote! {
        impl #name {
            pub fn builder() -> #builder_ident {
                #builder_ident::default()
            }
        }
        
        #[derive(Default)]
        pub struct #builder_ident {
            #(#field_names: Option<#field_types>),*
        }
        
        impl #builder_ident {
            #(
                pub fn #field_names(mut self, #field_names: #field_types) -> Self {
                    self.#field_names = Some(#field_names);
                    self
                }
            )*
            
            pub fn build(self) -> Result<#name, Box<dyn std::error::Error>> {
                Ok(#name {
                    #(#field_names: self.#field_names.ok_to_string()?),*
                })
            }
        }
    };
    
    expanded.into()
}

// Usage
#[derive(Builder)]
struct User {
    name: String,
    age: u32,
}
```

### Attribute Macros
```rust
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn, AttributeArgs};

#[proc_macro_attribute]
pub fn time_it(_args: TokenStream, input: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(input as ItemFn);
    let fn_name = &input_fn.sig.ident;
    let fn_block = &input_fn.block;
    let fn_vis = &input_fn.vis;
    let fn_sig = &input_fn.sig;
    
    let expanded = quote! {
        #fn_vis #fn_sig {
            let start = std::time::Instant::now();
            let result = (|| #fn_block)();
            let duration = start.elapsed();
            println!("Function {} took: {:?}", stringify!(#fn_name), duration);
            result
        }
    };
    
    expanded.into()
}

// Usage
#[time_it]
fn slow_function() {
    std::thread::sleep(std::time::Duration::from_millis(100));
    println!("Done!");
}
```

## Advanced Macro Patterns

### Conditional Compilation
```rust
macro_rules! debug_print {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        println!($($arg)*);
    };
}

// Log levels macro
macro_rules! log {
    (ERROR, $($arg:tt)*) => {
        eprintln!("[ERROR] {}", format!($($arg)*));
    };
    (WARN, $($arg:tt)*) => {
        #[cfg(feature = "warnings")]
        println!("[WARN] {}", format!($($arg)*));
    };
    (INFO, $($arg:tt)*) => {
        #[cfg(debug_assertions)]
        println!("[INFO] {}", format!($($arg)*));
    };
}
```

### State Machine Macro
```rust
macro_rules! state_machine {
    (
        $name:ident {
            states: [$($state:ident),*],
            transitions: {
                $($from:ident => $to:ident on $event:ident),*
            }
        }
    ) => {
        #[derive(Debug, Clone, PartialEq)]
        enum State {
            $($state),*
        }
        
        #[derive(Debug)]
        enum Event {
            $($event),*
        }
        
        struct $name {
            state: State,
        }
        
        impl $name {
            fn new() -> Self {
                Self {
                    state: State::Initial, // Assume first state is initial
                }
            }
            
            fn transition(&mut self, event: Event) -> Result<(), String> {
                let new_state = match (&self.state, event) {
                    $(
                        (State::$from, Event::$event) => State::$to,
                    )*
                    _ => return Err(format!(
                        "Invalid transition from {:?} with event {:?}", 
                        self.state, event
                    )),
                };
                
                self.state = new_state;
                Ok(())
            }
            
            fn current_state(&self) -> &State {
                &self.state
            }
        }
    };
}

// Usage
state_machine! {
    Door {
        states: [Closed, Open, Locked],
        transitions: {
            Closed => Open on OpenEvent,
            Open => Closed on CloseEvent,
            Closed => Locked on LockEvent,
            Locked => Closed on UnlockEvent
        }
    }
}
```

### Error Handling Macro
```rust
macro_rules! try_all {
    ($($expr:expr),*) => {
        {
            let mut errors = Vec::new();
            $(
                if let Err(e) = $expr {
                    errors.push(format!("{}: {}", stringify!($expr), e));
                }
            )*
            
            if errors.is_empty() {
                Ok(())
            } else {
                Err(errors.join("; "))
            }
        }
    };
}

// Usage example
fn validate_user(name: &str, email: &str, age: u32) -> Result<(), String> {
    try_all!(
        validate_name(name),
        validate_email(email),
        validate_age(age)
    )
}
```

## Best Practices

### Hygiene and Scoping
```rust
// Use local bindings to avoid name conflicts
macro_rules! safe_macro {
    ($x:expr) => {
        {
            let temp = $x; // Local binding
            temp * 2
        }
    };
}
```

### Documentation and Testing
```rust
/// Creates a vector with repeated elements
/// 
/// # Examples
/// 
/// ```
/// let v = repeat_vec![5; 3];
/// assert_eq!(v, vec![5, 5, 5]);
/// ```
macro_rules! repeat_vec {
    ($elem:expr; $n:expr) => {
        vec![$elem; $n]
    };
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_repeat_vec() {
        let v = repeat_vec![42; 3];
        assert_eq!(v, vec![42, 42, 42]);
    }
}
```

## Common Use Cases

1. **Domain-Specific Languages (DSL)**: Create syntax for specific problem domains
2. **Code Generation**: Generate repetitive code patterns
3. **Compile-time Validation**: Check constraints at compile time
4. **API Wrappers**: Simplify complex API usage
5. **Serialization/Deserialization**: Auto-generate trait implementations

## Key Takeaways

- Use declarative macros for simple code generation
- Use procedural macros for complex transformations
- Always consider macro hygiene and error messages
- Test macros thoroughly with edge cases
- Document macro usage with examples
- Prefer functions over macros when possible
