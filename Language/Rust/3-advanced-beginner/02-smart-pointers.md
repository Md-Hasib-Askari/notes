# Smart Pointers (Box, Rc, Arc, RefCell)

## Box<T> - Heap Allocation

### Basic Box Usage
```rust
fn main() {
    // Simple heap allocation
    let b = Box::new(5);
    println!("b = {}", b);
    
    // Large data on heap
    let large_array = Box::new([0; 1_000_000]);
    println!("Array allocated on heap");
    
    // Box automatically deallocates when out of scope
}
```

### Recursive Data Structures
```rust
#[derive(Debug)]
enum List {
    Cons(i32, Box<List>),
    Nil,
}

use List::{Cons, Nil};

impl List {
    fn new() -> List {
        Nil
    }
    
    fn prepend(self, elem: i32) -> List {
        Cons(elem, Box::new(self))
    }
    
    fn len(&self) -> usize {
        match self {
            Cons(_, tail) => 1 + tail.len(),
            Nil => 0,
        }
    }
}

fn main() {
    let mut list = List::new();
    list = list.prepend(1).prepend(2).prepend(3);
    
    println!("List: {:?}", list);
    println!("Length: {}", list.len());
}
```

### Box with Trait Objects
```rust
trait Draw {
    fn draw(&self);
}

struct Circle {
    radius: f64,
}

struct Rectangle {
    width: f64,
    height: f64,
}

impl Draw for Circle {
    fn draw(&self) {
        println!("Drawing circle with radius {}", self.radius);
    }
}

impl Draw for Rectangle {
    fn draw(&self) {
        println!("Drawing rectangle {}x{}", self.width, self.height);
    }
}

fn main() {
    let shapes: Vec<Box<dyn Draw>> = vec![
        Box::new(Circle { radius: 5.0 }),
        Box::new(Rectangle { width: 10.0, height: 20.0 }),
    ];
    
    for shape in shapes {
        shape.draw();
    }
}
```

## Rc<T> - Reference Counted

### Shared Ownership
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
    
    println!("Reference count after creating leaf: {}", Rc::strong_count(&leaf));
    
    let branch = Rc::new(Node {
        value: 5,
        children: vec![Rc::clone(&leaf)],
    });
    
    println!("Reference count after creating branch: {}", Rc::strong_count(&leaf));
    
    {
        let _leaf2 = Rc::clone(&leaf);
        println!("Reference count after creating leaf2: {}", Rc::strong_count(&leaf));
    }
    
    println!("Reference count after leaf2 goes out of scope: {}", Rc::strong_count(&leaf));
}
```

### Rc with RefCell for Interior Mutability
```rust
use std::rc::Rc;
use std::cell::RefCell;

#[derive(Debug)]
struct Node {
    value: RefCell<i32>,
    children: RefCell<Vec<Rc<Node>>>,
}

impl Node {
    fn new(value: i32) -> Rc<Self> {
        Rc::new(Node {
            value: RefCell::new(value),
            children: RefCell::new(vec![]),
        })
    }
    
    fn add_child(&self, child: Rc<Node>) {
        self.children.borrow_mut().push(child);
    }
    
    fn set_value(&self, value: i32) {
        *self.value.borrow_mut() = value;
    }
}

fn main() {
    let root = Node::new(1);
    let child1 = Node::new(2);
    let child2 = Node::new(3);
    
    root.add_child(child1);
    root.add_child(child2);
    
    root.set_value(10);
    
    println!("Root value: {}", root.value.borrow());
    println!("Children count: {}", root.children.borrow().len());
}
```

## Arc<T> - Atomic Reference Counted

### Thread-Safe Shared Ownership
```rust
use std::sync::Arc;
use std::thread;

fn main() {
    let data = Arc::new(vec![1, 2, 3, 4, 5]);
    let mut handles = vec![];
    
    for i in 0..3 {
        let data = Arc::clone(&data);
        let handle = thread::spawn(move || {
            println!("Thread {}: {:?}", i, data);
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
}
```

### Arc with Mutex for Shared Mutable State
```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];
    
    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    println!("Result: {}", *counter.lock().unwrap());
}
```

### Arc with RwLock for Read-Heavy Workloads
```rust
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;

fn main() {
    let data = Arc::new(RwLock::new(vec![1, 2, 3, 4, 5]));
    let mut handles = vec![];
    
    // Spawn readers
    for i in 0..5 {
        let data = Arc::clone(&data);
        let handle = thread::spawn(move || {
            let reader = data.read().unwrap();
            println!("Reader {}: {:?}", i, *reader);
        });
        handles.push(handle);
    }
    
    // Spawn a writer
    let data_writer = Arc::clone(&data);
    let writer_handle = thread::spawn(move || {
        thread::sleep(Duration::from_millis(100));
        let mut writer = data_writer.write().unwrap();
        writer.push(6);
        println!("Writer: added 6");
    });
    handles.push(writer_handle);
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    println!("Final data: {:?}", *data.read().unwrap());
}
```

## RefCell<T> - Interior Mutability

### Basic RefCell Usage
```rust
use std::cell::RefCell;

fn main() {
    let data = RefCell::new(5);
    
    // Immutable borrow
    {
        let borrowed = data.borrow();
        println!("Value: {}", *borrowed);
        // Can have multiple immutable borrows
        let borrowed2 = data.borrow();
        println!("Value again: {}", *borrowed2);
    }
    
    // Mutable borrow
    {
        let mut borrowed_mut = data.borrow_mut();
        *borrowed_mut = 10;
    }
    
    println!("New value: {}", *data.borrow());
}
```

### Mock Object Pattern
```rust
use std::cell::RefCell;

trait Messenger {
    fn send(&self, msg: &str);
}

struct MockMessenger {
    sent_messages: RefCell<Vec<String>>,
}

impl MockMessenger {
    fn new() -> MockMessenger {
        MockMessenger {
            sent_messages: RefCell::new(vec![]),
        }
    }
}

impl Messenger for MockMessenger {
    fn send(&self, message: &str) {
        self.sent_messages.borrow_mut().push(String::from(message));
    }
}

fn main() {
    let mock_messenger = MockMessenger::new();
    
    mock_messenger.send("Hello");
    mock_messenger.send("World");
    
    let messages = mock_messenger.sent_messages.borrow();
    println!("Sent messages: {:?}", *messages);
}
```

## Combining Smart Pointers

### Complex Data Structure
```rust
use std::rc::{Rc, Weak};
use std::cell::RefCell;

#[derive(Debug)]
struct Node {
    value: i32,
    children: RefCell<Vec<Rc<Node>>>,
    parent: RefCell<Weak<Node>>,
}

impl Node {
    fn new(value: i32) -> Rc<Self> {
        Rc::new(Node {
            value,
            children: RefCell::new(vec![]),
            parent: RefCell::new(Weak::new()),
        })
    }
    
    fn add_child(parent: &Rc<Self>, child: Rc<Self>) {
        child.parent.borrow_mut().replace(Rc::downgrade(parent));
        parent.children.borrow_mut().push(child);
    }
}

fn main() {
    let root = Node::new(1);
    let child1 = Node::new(2);
    let child2 = Node::new(3);
    
    Node::add_child(&root, child1);
    Node::add_child(&root, child2);
    
    println!("Root children: {}", root.children.borrow().len());
    
    // Access parent from child
    if let Some(first_child) = root.children.borrow().first() {
        if let Some(parent) = first_child.parent.borrow().upgrade() {
            println!("Child's parent value: {}", parent.value);
        }
    }
}
```

Smart pointers provide powerful patterns for memory management and shared ownership while maintaining Rust's safety guarantees.
