# Advanced Trait Usage

## Overview
Master advanced trait concepts including associated types, higher-ranked trait bounds (HRTBs), trait objects, and complex trait relationships.

## Associated Types

### Basic Associated Types
```rust
// Iterator trait with associated type
trait Iterator {
    type Item;
    
    fn next(&mut self) -> Option<Self::Item>;
}

// Custom iterator implementation
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

// Generic trait with associated types
trait Collect<T> {
    type Output;
    
    fn collect(self) -> Self::Output;
}

impl<T> Collect<T> for Vec<T> {
    type Output = Vec<T>;
    
    fn collect(self) -> Self::Output {
        self
    }
}

// Associated types vs generic parameters
trait Graph {
    type Node;
    type Edge;
    
    fn nodes(&self) -> Vec<Self::Node>;
    fn edges(&self) -> Vec<Self::Edge>;
}

// This would be less ergonomic:
// trait Graph<Node, Edge> {
//     fn nodes(&self) -> Vec<Node>;
//     fn edges(&self) -> Vec<Edge>;
// }
```

### Complex Associated Type Patterns
```rust
// Associated types with bounds
trait Parser {
    type Output: Clone + std::fmt::Debug;
    type Error: std::error::Error;
    
    fn parse(&self, input: &str) -> Result<Self::Output, Self::Error>;
}

// Associated types with lifetimes
trait Borrower<'a> {
    type Borrowed: 'a;
    
    fn borrow(&'a self) -> Self::Borrowed;
}

// Generic associated types (GATs) - requires nightly
trait StreamingIterator {
    type Item<'a> where Self: 'a;
    
    fn next<'a>(&'a mut self) -> Option<Self::Item<'a>>;
}

// Example GAT implementation
struct WindowIterator<T> {
    data: Vec<T>,
    window_size: usize,
    position: usize,
}

impl<T> StreamingIterator for WindowIterator<T> {
    type Item<'a> = &'a [T] where Self: 'a;
    
    fn next<'a>(&'a mut self) -> Option<Self::Item<'a>> {
        if self.position + self.window_size <= self.data.len() {
            let window = &self.data[self.position..self.position + self.window_size];
            self.position += 1;
            Some(window)
        } else {
            None
        }
    }
}
```

## Higher-Ranked Trait Bounds (HRTBs)

### Understanding HRTBs
```rust
// Function that works with any lifetime
fn call_with_closure<F>(f: F) -> i32
where
    F: for<'a> Fn(&'a str) -> i32,
{
    f("hello")
}

// HRTB with multiple parameters
fn map_and_print<F, T>(items: Vec<T>, f: F)
where
    F: for<'a> Fn(&'a T) -> String,
    T: std::fmt::Debug,
{
    for item in &items {
        println!("{}", f(item));
    }
}

// Complex HRTB example
trait Closure<Args> {
    type Output;
    fn call(&self, args: Args) -> Self::Output;
}

fn higher_order_function<F>(f: F) -> i32
where
    F: for<'a, 'b> Closure<(&'a str, &'b str), Output = i32>,
{
    f.call(("hello", "world"))
}

// Real-world example: Parser combinators
trait ParseResult<'a, T> {
    fn into_result(self) -> Result<(&'a str, T), &'a str>;
}

trait Parser<T> {
    fn parse<'a>(&self, input: &'a str) -> Box<dyn ParseResult<'a, T> + 'a>;
}

fn combine_parsers<P1, P2, T1, T2>(p1: P1, p2: P2) -> impl Parser<(T1, T2)>
where
    P1: for<'a> Parser<T1>,
    P2: for<'a> Parser<T2>,
{
    // Implementation would combine the parsers
    todo!()
}
```

### Lifetime Subtyping with HRTBs
```rust
// Function signature with HRTB
fn process_strings<F>(f: F)
where
    F: for<'a> Fn(&'a str) -> &'a str,
{
    let s1 = "hello";
    let result = f(s1);
    println!("{}", result);
}

// This works because of HRTB
fn identity(s: &str) -> &str {
    s
}

fn usage_example() {
    process_strings(identity); // Works!
    process_strings(|s| s);    // Also works!
}
```

## Trait Objects and Dynamic Dispatch

### Advanced Trait Objects
```rust
// Object-safe trait
trait Drawable {
    fn draw(&self);
    fn area(&self) -> f64;
}

// Object-safe trait with associated types
trait Renderer {
    type Surface;
    
    fn render(&self, surface: &Self::Surface);
}

// Making traits object-safe
trait CloneableDrawable: Drawable {
    fn clone_box(&self) -> Box<dyn CloneableDrawable>;
}

impl<T> CloneableDrawable for T
where
    T: Drawable + Clone + 'static,
{
    fn clone_box(&self) -> Box<dyn CloneableDrawable> {
        Box::new(self.clone())
    }
}

// Complex trait object with multiple bounds
fn process_items(items: Vec<Box<dyn Drawable + Send + Sync>>) {
    for item in items {
        item.draw();
    }
}

// Trait object with lifetime parameters
trait Processor<'a> {
    fn process(&self, data: &'a [u8]) -> &'a [u8];
}

fn use_processor(processor: &dyn for<'a> Processor<'a>, data: &[u8]) -> &[u8] {
    processor.process(data)
}
```

### Custom VTable Implementation
```rust
// Manual trait object implementation
struct DrawableVTable {
    draw: fn(*const ()),
    area: fn(*const ()) -> f64,
    drop: fn(*mut ()),
}

struct DrawableObject {
    data: *const (),
    vtable: &'static DrawableVTable,
}

impl DrawableObject {
    fn new<T: Drawable + 'static>(obj: T) -> Self {
        let boxed = Box::into_raw(Box::new(obj));
        
        static VTABLE: DrawableVTable = DrawableVTable {
            draw: |ptr| unsafe {
                let obj = &*(ptr as *const T);
                obj.draw();
            },
            area: |ptr| unsafe {
                let obj = &*(ptr as *const T);
                obj.area()
            },
            drop: |ptr| unsafe {
                let _ = Box::from_raw(ptr as *mut T);
            },
        };
        
        DrawableObject {
            data: boxed as *const (),
            vtable: &VTABLE,
        }
    }
    
    fn draw(&self) {
        (self.vtable.draw)(self.data);
    }
    
    fn area(&self) -> f64 {
        (self.vtable.area)(self.data)
    }
}

impl Drop for DrawableObject {
    fn drop(&mut self) {
        (self.vtable.drop)(self.data as *mut ());
    }
}
```

## Advanced Trait Patterns

### Phantom Types
```rust
use std::marker::PhantomData;

// State machine with phantom types
struct State<S> {
    _phantom: PhantomData<S>,
}

struct Locked;
struct Unlocked;

struct Door<S> {
    state: State<S>,
}

impl Door<Locked> {
    fn new() -> Door<Locked> {
        Door {
            state: State { _phantom: PhantomData },
        }
    }
    
    fn unlock(self, key: &str) -> Result<Door<Unlocked>, Door<Locked>> {
        if key == "correct_key" {
            Ok(Door {
                state: State { _phantom: PhantomData },
            })
        } else {
            Err(self)
        }
    }
}

impl Door<Unlocked> {
    fn open(&self) {
        println!("Door opened!");
    }
    
    fn lock(self) -> Door<Locked> {
        Door {
            state: State { _phantom: PhantomData },
        }
    }
}

// Type-level programming with traits
trait Nat {}

struct Zero;
struct Succ<N: Nat>(PhantomData<N>);

impl Nat for Zero {}
impl<N: Nat> Nat for Succ<N> {}

trait Add<Other: Nat>: Nat {
    type Output: Nat;
}

impl<N: Nat> Add<Zero> for N {
    type Output = N;
}

impl<N: Nat, M: Nat> Add<Succ<M>> for N
where
    N: Add<M>,
    Succ<N::Output>: Nat,
{
    type Output = Succ<N::Output>;
}
```

### Sealed Traits
```rust
// Sealed trait pattern
mod sealed {
    pub trait Sealed {}
}

pub trait SpecialTrait: sealed::Sealed {
    fn special_method(&self);
}

// Only types in this module can implement SpecialTrait
impl sealed::Sealed for i32 {}
impl SpecialTrait for i32 {
    fn special_method(&self) {
        println!("Special method for i32: {}", self);
    }
}

impl sealed::Sealed for String {}
impl SpecialTrait for String {
    fn special_method(&self) {
        println!("Special method for String: {}", self);
    }
}

// Users cannot implement SpecialTrait for their own types
```

### Trait Aliases and Complex Bounds
```rust
// Trait alias (requires feature flag)
#![feature(trait_alias)]

trait SendSync = Send + Sync;
trait JsonSerializable = serde::Serialize + serde::Deserialize<'static>;

// Complex trait bounds
fn complex_function<T>() 
where
    T: Clone + Send + Sync + 'static,
    T: std::fmt::Debug + std::fmt::Display,
    T: serde::Serialize + serde::Deserialize<'static>,
{
    // Function implementation
}

// Conditional trait implementations
trait ConditionalTrait {
    fn method(&self);
}

impl<T> ConditionalTrait for Vec<T>
where
    T: std::fmt::Debug + Clone,
{
    fn method(&self) {
        println!("Vec with {} debuggable, cloneable items", self.len());
    }
}

// Blanket implementations
trait MyTrait {
    fn my_method(&self);
}

impl<T> MyTrait for T
where
    T: std::fmt::Display + Clone,
{
    fn my_method(&self) {
        println!("Blanket implementation for: {}", self);
    }
}
```

### Extension Traits
```rust
// Extension trait for existing types
trait VectorExtensions<T> {
    fn first_and_last(&self) -> Option<(&T, &T)>;
    fn middle_element(&self) -> Option<&T>;
}

impl<T> VectorExtensions<T> for Vec<T> {
    fn first_and_last(&self) -> Option<(&T, &T)> {
        if self.len() >= 2 {
            Some((&self[0], &self[self.len() - 1]))
        } else {
            None
        }
    }
    
    fn middle_element(&self) -> Option<&T> {
        if !self.is_empty() {
            Some(&self[self.len() / 2])
        } else {
            None
        }
    }
}

// Extension trait for Results
trait ResultExtensions<T, E> {
    fn tap_err<F>(self, f: F) -> Self
    where
        F: FnOnce(&E);
    
    fn context<C>(self, context: C) -> Result<T, (E, C)>;
}

impl<T, E> ResultExtensions<T, E> for Result<T, E> {
    fn tap_err<F>(self, f: F) -> Self
    where
        F: FnOnce(&E),
    {
        if let Err(ref e) = self {
            f(e);
        }
        self
    }
    
    fn context<C>(self, context: C) -> Result<T, (E, C)> {
        self.map_err(|e| (e, context))
    }
}
```

## Trait Coherence and Orphan Rules

### Understanding Coherence
```rust
// Valid: implementing your trait for your type
struct MyStruct;
trait MyTrait {
    fn my_method(&self);
}

impl MyTrait for MyStruct {
    fn my_method(&self) {
        println!("My implementation");
    }
}

// Valid: implementing foreign trait for your type
impl std::fmt::Display for MyStruct {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MyStruct")
    }
}

// Valid: implementing your trait for foreign type with local wrapper
struct Wrapper<T>(T);

impl MyTrait for Wrapper<Vec<i32>> {
    fn my_method(&self) {
        println!("Wrapped Vec implementation");
    }
}

// Invalid: would violate orphan rule
// impl std::fmt::Display for Vec<i32> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         // This would conflict with potential upstream implementations
//     }
// }
```

### Newtype Pattern for Trait Implementation
```rust
// Newtype to implement foreign traits on foreign types
struct JsonVec<T>(Vec<T>);

impl<T> std::ops::Deref for JsonVec<T> {
    type Target = Vec<T>;
    
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> std::ops::DerefMut for JsonVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> serde::Serialize for JsonVec<T>
where
    T: serde::Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}
```

## Type-Level Programming

### Type-Level Lists
```rust
// Type-level list implementation
trait TypeList {}

struct Nil;
struct Cons<Head, Tail: TypeList>(PhantomData<(Head, Tail)>);

impl TypeList for Nil {}
impl<Head, Tail: TypeList> TypeList for Cons<Head, Tail> {}

// Type-level operations
trait Length {
    const LEN: usize;
}

impl Length for Nil {
    const LEN: usize = 0;
}

impl<Head, Tail: TypeList + Length> Length for Cons<Head, Tail> {
    const LEN: usize = 1 + Tail::LEN;
}

// Type-level contains check
trait Contains<T> {
    const CONTAINS: bool;
}

impl<T> Contains<T> for Nil {
    const CONTAINS: bool = false;
}

impl<T, Head, Tail> Contains<T> for Cons<Head, Tail>
where
    Tail: TypeList + Contains<T>,
{
    const CONTAINS: bool = Tail::CONTAINS;
}

impl<T, Tail> Contains<T> for Cons<T, Tail>
where
    Tail: TypeList + Contains<T>,
{
    const CONTAINS: bool = true;
}

// Usage example
type MyList = Cons<i32, Cons<String, Cons<f64, Nil>>>;

fn type_level_demo() {
    println!("List length: {}", MyList::LEN);
    println!("Contains i32: {}", <MyList as Contains<i32>>::CONTAINS);
    println!("Contains bool: {}", <MyList as Contains<bool>>::CONTAINS);
}
```

## Best Practices

### Trait Design Guidelines
1. **Prefer Associated Types**: Use when there's one natural implementation per type
2. **Use Generic Parameters**: When multiple implementations make sense
3. **Keep Traits Focused**: Single responsibility principle
4. **Consider Object Safety**: Design for dynamic dispatch when needed
5. **Use Extension Traits**: Add methods to existing types safely
6. **Document Trait Contracts**: Specify expected behavior clearly

### Performance Considerations
```rust
// Static dispatch (faster)
fn static_dispatch<T: MyTrait>(item: T) {
    item.my_method();
}

// Dynamic dispatch (more flexible)
fn dynamic_dispatch(item: &dyn MyTrait) {
    item.my_method();
}

// Enum dispatch (balance of both)
enum DrawableEnum {
    Circle(Circle),
    Rectangle(Rectangle),
}

impl Drawable for DrawableEnum {
    fn draw(&self) {
        match self {
            DrawableEnum::Circle(c) => c.draw(),
            DrawableEnum::Rectangle(r) => r.draw(),
        }
    }
    
    fn area(&self) -> f64 {
        match self {
            DrawableEnum::Circle(c) => c.area(),
            DrawableEnum::Rectangle(r) => r.area(),
        }
    }
}
```

## Key Takeaways

1. **Associated Types**: Define relationships between types in traits
2. **HRTBs**: Work with any lifetime, enabling powerful abstractions
3. **Trait Objects**: Enable runtime polymorphism with some constraints
4. **Phantom Types**: Encode state and invariants in the type system
5. **Sealed Traits**: Control trait implementation to prevent breaking changes
6. **Extension Traits**: Add functionality to existing types safely
7. **Type-Level Programming**: Compute at the type level for zero-cost abstractions
