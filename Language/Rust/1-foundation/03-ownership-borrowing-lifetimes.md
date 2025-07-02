# Ownership, Borrowing, and Lifetimes

## Ownership Rules

### The Three Rules
1. Each value in Rust has a variable that's called its **owner**
2. There can only be **one owner** at a time
3. When the owner goes **out of scope**, the value will be **dropped**

### Basic Ownership
```rust
fn main() {
    let s1 = String::from("hello");   // s1 owns the string
    let s2 = s1;                      // Move: s1 no longer valid
    
    // println!("{}", s1);            // Error! s1 moved to s2
    println!("{}", s2);               // OK: s2 owns the string
}
```

### Copy vs Move
```rust
fn main() {
    // Stack data: Copy
    let x = 5;
    let y = x;                        // Copy: both x and y are valid
    println!("x: {}, y: {}", x, y);   // OK
    
    // Heap data: Move
    let s1 = String::from("hello");
    let s2 = s1;                      // Move: s1 is no longer valid
    // println!("{}", s1);            // Error!
}
```

### Functions and Ownership
```rust
fn main() {
    let s = String::from("hello");
    takes_ownership(s);               // s moved into function
    // println!("{}", s);             // Error! s no longer valid
    
    let x = 5;
    makes_copy(x);                    // Copy: x still valid
    println!("x: {}", x);             // OK
}

fn takes_ownership(some_string: String) {
    println!("{}", some_string);
} // some_string dropped here

fn makes_copy(some_integer: i32) {
    println!("{}", some_integer);
} // some_integer goes out of scope, nothing special happens
```

## Borrowing

### Immutable References
```rust
fn main() {
    let s1 = String::from("hello");
    let len = calculate_length(&s1);  // Borrow s1
    println!("Length of '{}' is {}.", s1, len); // s1 still valid
}

fn calculate_length(s: &String) -> usize {
    s.len()
} // s goes out of scope, but doesn't drop the value (just a reference)
```

### Mutable References
```rust
fn main() {
    let mut s = String::from("hello");
    change(&mut s);                   // Mutable borrow
    println!("{}", s);                // Prints: "hello, world"
}

fn change(some_string: &mut String) {
    some_string.push_str(", world");
}
```

### Borrowing Rules
```rust
fn main() {
    let mut s = String::from("hello");
    
    // Multiple immutable references OK
    let r1 = &s;
    let r2 = &s;
    println!("{} and {}", r1, r2);
    
    // One mutable reference OK (after immutable refs done)
    let r3 = &mut s;
    println!("{}", r3);
    
    // But not both at the same time!
    // let r4 = &s;                   // Error if r3 still used
}
```

### Reference Rules
1. At any given time, you can have **either** one mutable reference **or** any number of immutable references
2. References must always be **valid** (no dangling references)

## Lifetimes

### Lifetime Basics
```rust
// Explicit lifetime annotation
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

fn main() {
    let string1 = String::from("long string is long");
    let string2 = "xyz";
    
    let result = longest(string1.as_str(), string2);
    println!("The longest string is {}", result);
}
```

### Lifetime Elision Rules
```rust
// These functions don't need explicit lifetime annotations:

fn first_word(s: &str) -> &str {      // Rule 1: input lifetime
    &s[..1]
}

fn announce_and_return(announcement: &str, value: &str) -> &str {
    println!("Attention: {}", announcement);
    value                             // Returns second parameter
}
```

### Struct Lifetimes
```rust
struct ImportantExcerpt<'a> {
    part: &'a str,                    // Reference must live as long as struct
}

impl<'a> ImportantExcerpt<'a> {
    fn level(&self) -> i32 {
        3
    }
    
    fn announce_and_return_part(&self, announcement: &str) -> &str {
        println!("Attention: {}", announcement);
        self.part
    }
}

fn main() {
    let novel = String::from("Call me Ishmael. Some years ago...");
    let first_sentence = novel.split('.').next().expect("Could not find a '.'");
    let i = ImportantExcerpt {
        part: first_sentence,
    };
}
```

### Static Lifetime
```rust
// 'static lifetime: lives for entire program duration
let s: &'static str = "I have a static lifetime.";

// String literals have 'static lifetime
static HELLO_WORLD: &str = "Hello, world!";
```

## Common Ownership Patterns

### Returning Owned Data
```rust
fn give_ownership() -> String {
    let some_string = String::from("hello");
    some_string                       // Return ownership
}

fn take_and_give_back(a_string: String) -> String {
    a_string                          // Return ownership back
}
```

### Clone for Multiple Ownership
```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1.clone();              // Explicit copy of heap data
    
    println!("s1: {}, s2: {}", s1, s2); // Both valid
}
```

### String Slices
```rust
fn main() {
    let s = String::from("hello world");
    
    let hello = &s[0..5];             // String slice
    let world = &s[6..11];
    let whole = &s[..];               // Entire string
    
    println!("{} {}", hello, world);
}

fn first_word(s: &str) -> &str {      // Works with &String and &str
    let bytes = s.as_bytes();
    
    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            return &s[0..i];
        }
    }
    
    &s[..]
}
```

## Memory Safety Benefits

### Prevents Common Errors
- **Use after free**: Ownership prevents using moved values
- **Double free**: Only one owner can drop a value
- **Memory leaks**: Automatic cleanup when owner goes out of scope
- **Dangling pointers**: Lifetime system ensures references are valid
- **Data races**: Borrowing rules prevent concurrent mutation

Understanding these concepts is crucial for writing safe, efficient Rust code!
