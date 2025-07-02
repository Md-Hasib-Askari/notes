# Memory Management Deep Dive

## Stack vs Heap Memory

### Understanding Memory Layout
```rust
fn main() {
    // Stack allocation - fast, limited size
    let stack_var = 42;               // Stored on stack
    let stack_array = [1, 2, 3, 4];   // Small arrays on stack
    
    // Heap allocation - slower, unlimited size
    let heap_var = Box::new(42);      // Stored on heap
    let heap_vec = vec![1, 2, 3, 4];  // Dynamic size on heap
    
    println!("Stack variable: {}", stack_var);
    println!("Heap variable: {}", heap_var);
}
```

### Memory Representation
```rust
use std::mem;

fn analyze_memory<T>(value: &T, name: &str) {
    println!("{}: size = {} bytes, align = {} bytes", 
             name, 
             mem::size_of::<T>(), 
             mem::align_of::<T>());
}

fn main() {
    analyze_memory(&42u8, "u8");
    analyze_memory(&42u32, "u32");
    analyze_memory(&42u64, "u64");
    analyze_memory(&String::from("hello"), "String");
    analyze_memory(&vec![1, 2, 3], "Vec<i32>");
    
    // Check pointer size
    let boxed = Box::new(42);
    analyze_memory(&boxed, "Box<i32>");
    
    // String internals
    let s = String::from("Hello, world!");
    println!("String '{}': ptr={:p}, len={}, capacity={}", 
             s, s.as_ptr(), s.len(), s.capacity());
}
```

## Zero-Cost Abstractions

### Iterator Performance
```rust
fn imperative_sum(data: &[i32]) -> i32 {
    let mut sum = 0;
    for i in 0..data.len() {
        sum += data[i];
    }
    sum
}

fn functional_sum(data: &[i32]) -> i32 {
    data.iter().sum()  // Zero-cost abstraction
}

fn main() {
    let data = vec![1, 2, 3, 4, 5];
    
    // Both compile to the same optimized assembly
    println!("Imperative: {}", imperative_sum(&data));
    println!("Functional: {}", functional_sum(&data));
}
```

### Generic Monomorphization
```rust
fn add<T>(a: T, b: T) -> T 
where 
    T: std::ops::Add<Output = T>
{
    a + b  // No runtime cost - specialized at compile time
}

fn main() {
    // Compiler generates separate functions for each type
    let int_result = add(5, 10);        // add_i32 function
    let float_result = add(5.5, 10.5);  // add_f64 function
    
    println!("Int: {}, Float: {}", int_result, float_result);
}
```

## Memory Layouts and Optimization

### Struct Layout and Padding
```rust
use std::mem;

#[repr(C)]
struct BadLayout {
    a: u8,   // 1 byte
    b: u64,  // 8 bytes (7 bytes padding after a)
    c: u8,   // 1 byte (7 bytes padding after c)
}

#[repr(C)]
struct GoodLayout {
    b: u64,  // 8 bytes
    a: u8,   // 1 byte
    c: u8,   // 1 byte (6 bytes padding after c)
}

fn main() {
    println!("BadLayout size: {}", mem::size_of::<BadLayout>());   // 24 bytes
    println!("GoodLayout size: {}", mem::size_of::<GoodLayout>()); // 16 bytes
    
    // Rust automatically reorders fields for optimal layout unless #[repr(C)]
    struct AutoLayout {
        a: u8,
        b: u64,
        c: u8,
    }
    
    println!("AutoLayout size: {}", mem::size_of::<AutoLayout>()); // 16 bytes
}
```

### Enum Layout Optimization
```rust
use std::mem;

// Null pointer optimization
enum Option<T> {
    Some(T),
    None,
}

// When T is a non-null pointer, Option<T> is same size as T
fn main() {
    println!("Box<i32> size: {}", mem::size_of::<Box<i32>>());
    println!("Option<Box<i32>> size: {}", mem::size_of::<Option<Box<i32>>>());
    
    // Tagged union
    enum Message {
        Quit,
        Move { x: i32, y: i32 },
        Write(String),
    }
    
    println!("Message size: {}", mem::size_of::<Message>());
}
```

## Custom Allocators

### Global Allocator
```rust
use std::alloc::{GlobalAlloc, Layout, System};

struct CountingAllocator;

static mut ALLOCATION_COUNT: usize = 0;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATION_COUNT += 1;
        System.alloc(layout)
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn main() {
    let _vec1 = vec![1, 2, 3];
    let _vec2 = vec![4, 5, 6, 7, 8];
    let _string = String::from("Hello, world!");
    
    unsafe {
        println!("Number of allocations: {}", ALLOCATION_COUNT);
    }
}
```

## Memory Leaks Prevention

### RAII and Drop
```rust
struct Resource {
    name: String,
}

impl Resource {
    fn new(name: &str) -> Self {
        println!("Acquiring resource: {}", name);
        Resource {
            name: name.to_string(),
        }
    }
}

impl Drop for Resource {
    fn drop(&mut self) {
        println!("Releasing resource: {}", self.name);
    }
}

fn main() {
    {
        let _resource1 = Resource::new("File handle");
        let _resource2 = Resource::new("Network socket");
    } // Resources automatically cleaned up here
    
    println!("Resources have been cleaned up");
}
```

### Handling Cycles with Weak References
```rust
use std::rc::{Rc, Weak};
use std::cell::RefCell;

#[derive(Debug)]
struct Node {
    value: i32,
    children: RefCell<Vec<Rc<Node>>>,
    parent: RefCell<Weak<Node>>, // Weak to prevent cycles
}

impl Node {
    fn new(value: i32) -> Rc<Self> {
        Rc::new(Node {
            value,
            children: RefCell::new(vec![]),
            parent: RefCell::new(Weak::new()),
        })
    }
}

fn main() {
    let root = Node::new(1);
    let child1 = Node::new(2);
    let child2 = Node::new(3);
    
    // Set up parent-child relationships
    child1.parent.borrow_mut().replace(Rc::downgrade(&root));
    child2.parent.borrow_mut().replace(Rc::downgrade(&root));
    
    root.children.borrow_mut().push(child1);
    root.children.borrow_mut().push(child2);
    
    println!("Tree created without memory leaks");
    // All nodes will be properly deallocated
}
```

## Performance Monitoring

### Memory Usage Tracking
```rust
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct TrackingAllocator;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATED: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        DEALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
    }
}

fn memory_usage() -> (usize, usize) {
    (
        ALLOCATED.load(Ordering::Relaxed),
        DEALLOCATED.load(Ordering::Relaxed),
    )
}

fn main() {
    let (start_alloc, start_dealloc) = memory_usage();
    
    {
        let mut data = Vec::with_capacity(1000);
        for i in 0..1000 {
            data.push(i);
        }
        
        let (mid_alloc, mid_dealloc) = memory_usage();
        println!("After allocation: {} bytes allocated, {} deallocated", 
                 mid_alloc - start_alloc, mid_dealloc - start_dealloc);
    }
    
    let (end_alloc, end_dealloc) = memory_usage();
    println!("After deallocation: {} bytes allocated, {} deallocated", 
             end_alloc - start_alloc, end_dealloc - start_dealloc);
}
```

### Stack vs Heap Performance
```rust
use std::time::Instant;

const SIZE: usize = 1_000_000;

fn stack_allocation() {
    let start = Instant::now();
    
    // This won't actually work - stack overflow
    // let array = [0; SIZE];
    
    println!("Stack allocation would be instant but limited");
}

fn heap_allocation() {
    let start = Instant::now();
    
    let vec = vec![0; SIZE];
    let duration = start.elapsed();
    
    println!("Heap allocation took: {:?}", duration);
    println!("Vector length: {}", vec.len());
}

fn main() {
    stack_allocation();
    heap_allocation();
}
```

Understanding memory management helps you write more efficient Rust programs and debug performance issues.
