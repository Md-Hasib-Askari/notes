# Traits and Generics

## Traits

### Defining and Implementing Traits
```rust
trait Summary {
    fn summarize(&self) -> String;
    
    // Default implementation
    fn summarize_author(&self) -> String {
        String::from("(Read more...)")
    }
}

struct NewsArticle {
    headline: String,
    location: String,
    author: String,
    content: String,
}

impl Summary for NewsArticle {
    fn summarize(&self) -> String {
        format!("{}, by {} ({})", self.headline, self.author, self.location)
    }
}

struct Tweet {
    username: String,
    content: String,
    reply: bool,
    retweet: bool,
}

impl Summary for Tweet {
    fn summarize(&self) -> String {
        format!("{}: {}", self.username, self.content)
    }
}

fn main() {
    let tweet = Tweet {
        username: String::from("horse_ebooks"),
        content: String::from("of course, as you probably already know, people"),
        reply: false,
        retweet: false,
    };
    
    println!("1 new tweet: {}", tweet.summarize());
}
```

### Trait Bounds
```rust
// Function that accepts any type implementing Summary
fn notify(item: &impl Summary) {
    println!("Breaking news! {}", item.summarize());
}

// Equivalent trait bound syntax
fn notify_v2<T: Summary>(item: &T) {
    println!("Breaking news! {}", item.summarize());
}

// Multiple trait bounds
fn notify_v3<T: Summary + std::fmt::Display>(item: &T) {
    println!("Breaking news! {}", item.summarize());
}

// Where clause for complex bounds
fn some_function<T, U>(t: &T, u: &U) -> i32
where
    T: std::fmt::Display + Clone,
    U: Clone + std::fmt::Debug,
{
    42
}
```

### Returning Types that Implement Traits
```rust
fn returns_summarizable() -> impl Summary {
    Tweet {
        username: String::from("horse_ebooks"),
        content: String::from("of course, as you probably already know, people"),
        reply: false,
        retweet: false,
    }
}
```

### Common Standard Traits
```rust
#[derive(Debug, Clone, PartialEq)]
struct Point {
    x: i32,
    y: i32,
}

impl std::fmt::Display for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

fn main() {
    let p1 = Point { x: 1, y: 2 };
    let p2 = p1.clone();  // Clone trait
    
    println!("{:?}", p1);  // Debug trait
    println!("{}", p1);    // Display trait
    
    if p1 == p2 {          // PartialEq trait
        println!("Points are equal");
    }
}
```

## Generics

### Generic Functions
```rust
fn largest<T: PartialOrd + Copy>(list: &[T]) -> T {
    let mut largest = list[0];
    
    for &item in list {
        if item > largest {
            largest = item;
        }
    }
    
    largest
}

fn main() {
    let number_list = vec![34, 50, 25, 100, 65];
    let result = largest(&number_list);
    println!("The largest number is {}", result);
    
    let char_list = vec!['y', 'm', 'a', 'q'];
    let result = largest(&char_list);
    println!("The largest char is {}", result);
}
```

### Generic Structs
```rust
struct Point<T> {
    x: T,
    y: T,
}

impl<T> Point<T> {
    fn new(x: T, y: T) -> Self {
        Point { x, y }
    }
    
    fn x(&self) -> &T {
        &self.x
    }
}

// Implementation for specific type
impl Point<f32> {
    fn distance_from_origin(&self) -> f32 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }
}

// Multiple generic types
struct Rectangle<T, U> {
    width: T,
    height: U,
}

fn main() {
    let integer_point = Point::new(5, 10);
    let float_point = Point::new(1.0, 4.0);
    
    println!("x = {}", integer_point.x());
    println!("Distance: {}", float_point.distance_from_origin());
}
```

### Generic Enums
```rust
enum Option<T> {
    Some(T),
    None,
}

enum Result<T, E> {
    Ok(T),
    Err(E),
}

// Custom generic enum
enum Either<L, R> {
    Left(L),
    Right(R),
}

impl<L, R> Either<L, R> {
    fn is_left(&self) -> bool {
        matches!(self, Either::Left(_))
    }
    
    fn is_right(&self) -> bool {
        matches!(self, Either::Right(_))
    }
}
```

### Associated Types
```rust
trait Iterator {
    type Item;  // Associated type
    
    fn next(&mut self) -> Option<Self::Item>;
}

struct Counter {
    current: u32,
    max: u32,
}

impl Counter {
    fn new(max: u32) -> Counter {
        Counter { current: 0, max }
    }
}

impl Iterator for Counter {
    type Item = u32;
    
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
    let mut counter = Counter::new(5);
    
    while let Some(value) = counter.next() {
        println!("{}", value);
    }
}
```

### Generic Constraints and Where Clauses
```rust
use std::fmt::Display;

fn compare_and_display<T>(t: &T, u: &T) -> bool
where
    T: Display + PartialOrd,
{
    println!("Comparing {} and {}", t, u);
    t >= u
}

// Blanket implementations
impl<T: Display> ToString for T {
    fn to_string(&self) -> String {
        format!("{}", self)
    }
}
```

### Advanced Generic Patterns
```rust
// Generic with lifetime
struct ImportantExcerpt<'a, T> {
    part: &'a T,
}

// Multiple constraints
fn notify_and_return<T>(item: T) -> T
where
    T: Summary + Clone,
{
    println!("Breaking news! {}", item.summarize());
    item.clone()
}

// Conditional implementation
struct Pair<T> {
    x: T,
    y: T,
}

impl<T> Pair<T> {
    fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

impl<T: Display + PartialOrd> Pair<T> {
    fn cmp_display(&self) {
        if self.x >= self.y {
            println!("The largest member is x = {}", self.x);
        } else {
            println!("The largest member is y = {}", self.y);
        }
    }
}
```

### Zero-Cost Abstractions
```rust
// Generic code compiles to specific implementations
// This is monomorphization - no runtime cost

fn main() {
    let integer = Some(5);
    let float = Some(5.0);
    
    // Compiler generates separate code for Option<i32> and Option<f64>
    // No performance penalty
}
```

Traits and generics provide Rust's powerful abstraction mechanisms while maintaining zero-cost performance characteristics.
