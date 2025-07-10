# Unsafe Rust Basics

## Understanding Unsafe

### When and Why to Use Unsafe
```rust
fn main() {
    // Safe Rust prevents:
    // - Dereferencing null/dangling pointers
    // - Buffer overflows
    // - Use after free
    // - Data races
    
    // Unsafe allows:
    // - Dereferencing raw pointers
    // - Calling unsafe functions
    // - Accessing/modifying static variables
    // - Implementing unsafe traits
    // - Accessing union fields
}
```

### Raw Pointers
```rust
fn main() {
    let mut num = 5;
    
    // Creating raw pointers (safe)
    let r1 = &num as *const i32;           // Immutable raw pointer
    let r2 = &mut num as *mut i32;         // Mutable raw pointer
    
    // Can create pointers to arbitrary memory (safe but not useful)
    let address = 0x012345usize;
    let r3 = address as *const i32;
    
    // Dereferencing requires unsafe
    unsafe {
        println!("r1 is: {}", *r1);
        
        *r2 = 10;
        println!("r2 is: {}", *r2);
        
        // Don't dereference arbitrary pointers!
        // println!("r3 is: {}", *r3); // Would likely crash
    }
}
```

### Unsafe Functions
```rust
unsafe fn dangerous() {
    println!("This is an unsafe function");
}

fn safe_wrapper() {
    // Safe wrapper around unsafe code
    unsafe {
        dangerous();
    }
}

// Useful unsafe function example
unsafe fn get_unchecked(slice: &[i32], index: usize) -> i32 {
    // Skip bounds checking for performance
    *slice.as_ptr().add(index)
}

fn main() {
    unsafe {
        dangerous();
    }
    
    safe_wrapper(); // Can call this safely
    
    let numbers = vec![1, 2, 3, 4, 5];
    unsafe {
        let value = get_unchecked(&numbers, 2);
        println!("Value at index 2: {}", value);
    }
}
```

### Creating Safe Abstractions
```rust
use std::slice;

fn split_at_mut(values: &mut [i32], mid: usize) -> (&mut [i32], &mut [i32]) {
    let len = values.len();
    let ptr = values.as_mut_ptr();
    
    assert!(mid <= len);
    
    unsafe {
        (
            slice::from_raw_parts_mut(ptr, mid),
            slice::from_raw_parts_mut(ptr.add(mid), len - mid),
        )
    }
}

fn main() {
    let mut vector = vec![1, 2, 3, 4, 5, 6];
    let (left, right) = split_at_mut(&mut vector, 3);
    
    println!("Left: {:?}", left);
    println!("Right: {:?}", right);
}
```

## FFI (Foreign Function Interface)

### Calling C Functions
```rust
// Declare external C functions
extern "C" {
    fn abs(input: i32) -> i32;
    fn sqrt(input: f64) -> f64;
}

fn main() {
    unsafe {
        println!("Absolute value of -3 according to C: {}", abs(-3));
        println!("Square root of 9.0 according to C: {}", sqrt(9.0));
    }
}
```

### Creating Functions Callable from C
```rust
use std::os::raw::c_char;
use std::ffi::CString;

#[no_mangle]
pub extern "C" fn call_from_c() {
    println!("Just called a Rust function from C!");
}

#[no_mangle]
pub extern "C" fn add_numbers(a: i32, b: i32) -> i32 {
    a + b
}

// Working with C strings
#[no_mangle]
pub extern "C" fn hello_from_rust() -> *const c_char {
    let c_string = CString::new("Hello from Rust!").unwrap();
    c_string.into_raw() // Transfer ownership to C
}

// Free memory allocated by Rust
#[no_mangle]
pub extern "C" fn free_rust_string(ptr: *mut c_char) {
    unsafe {
        if !ptr.is_null() {
            let _ = CString::from_raw(ptr); // Reclaim ownership and drop
        }
    }
}

fn main() {
    // Test our functions
    let result = add_numbers(5, 3);
    println!("5 + 3 = {}", result);
}
```

### Working with C Structures
```rust
use std::os::raw::{c_int, c_char};

#[repr(C)]
struct Point {
    x: c_int,
    y: c_int,
}

extern "C" {
    fn process_point(point: *const Point) -> c_int;
}

#[no_mangle]
pub extern "C" fn create_point(x: c_int, y: c_int) -> Point {
    Point { x, y }
}

fn main() {
    let point = Point { x: 10, y: 20 };
    
    unsafe {
        let result = process_point(&point);
        println!("C function returned: {}", result);
    }
}
```

## Working with Raw Memory

### Manual Memory Management
```rust
use std::alloc::{alloc, dealloc, Layout};
use std::ptr;

fn main() {
    unsafe {
        // Allocate memory for 4 i32 values
        let layout = Layout::array::<i32>(4).unwrap();
        let ptr = alloc(layout) as *mut i32;
        
        if ptr.is_null() {
            panic!("Failed to allocate memory");
        }
        
        // Initialize the memory
        for i in 0..4 {
            ptr::write(ptr.add(i), i as i32 * 10);
        }
        
        // Read the values
        for i in 0..4 {
            let value = ptr::read(ptr.add(i));
            println!("Value at index {}: {}", i, value);
        }
        
        // Clean up
        dealloc(ptr as *mut u8, layout);
    }
}
```

### Implementing a Simple Vec
```rust
use std::alloc::{alloc, dealloc, realloc, Layout};
use std::ptr;

struct SimpleVec<T> {
    ptr: *mut T,
    capacity: usize,
    len: usize,
}

impl<T> SimpleVec<T> {
    fn new() -> Self {
        SimpleVec {
            ptr: ptr::null_mut(),
            capacity: 0,
            len: 0,
        }
    }
    
    fn push(&mut self, item: T) {
        if self.len == self.capacity {
            self.grow();
        }
        
        unsafe {
            ptr::write(self.ptr.add(self.len), item);
        }
        self.len += 1;
    }
    
    fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            unsafe {
                Some(&*self.ptr.add(index))
            }
        } else {
            None
        }
    }
    
    fn grow(&mut self) {
        let new_capacity = if self.capacity == 0 { 1 } else { self.capacity * 2 };
        let new_layout = Layout::array::<T>(new_capacity).unwrap();
        
        unsafe {
            if self.capacity == 0 {
                self.ptr = alloc(new_layout) as *mut T;
            } else {
                let old_layout = Layout::array::<T>(self.capacity).unwrap();
                self.ptr = realloc(self.ptr as *mut u8, old_layout, new_layout.size()) as *mut T;
            }
        }
        
        self.capacity = new_capacity;
    }
}

impl<T> Drop for SimpleVec<T> {
    fn drop(&mut self) {
        if self.capacity > 0 {
            unsafe {
                // Drop all elements
                for i in 0..self.len {
                    ptr::drop_in_place(self.ptr.add(i));
                }
                
                // Deallocate memory
                let layout = Layout::array::<T>(self.capacity).unwrap();
                dealloc(self.ptr as *mut u8, layout);
            }
        }
    }
}

fn main() {
    let mut vec = SimpleVec::new();
    vec.push(1);
    vec.push(2);
    vec.push(3);
    
    for i in 0..3 {
        if let Some(value) = vec.get(i) {
            println!("Value at {}: {}", i, value);
        }
    }
}
```

## Union Types

### Basic Union Usage
```rust
union MyUnion {
    i: i32,
    f: f32,
}

fn main() {
    let mut u = MyUnion { i: 42 };
    
    unsafe {
        println!("Integer: {}", u.i);
        
        u.f = 3.14;
        println!("Float: {}", u.f);
        
        // Reading as integer after writing as float (unsafe!)
        println!("Float bits as int: {}", u.i);
    }
}
```

## Safety Guidelines

### Best Practices
```rust
// 1. Minimize unsafe code
// 2. Encapsulate unsafe in safe APIs
// 3. Document safety invariants
// 4. Use tools like Miri for testing

/// # Safety
/// 
/// The caller must ensure that:
/// - `ptr` points to valid memory
/// - The memory pointed to by `ptr` is properly aligned
/// - The memory is valid for reads
unsafe fn read_raw<T>(ptr: *const T) -> T {
    ptr::read(ptr)
}

// Safe wrapper
fn safe_read<T>(value: &T) -> T 
where 
    T: Copy 
{
    unsafe {
        read_raw(value as *const T)
    }
}
```

Unsafe Rust is a powerful tool but should be used sparingly and with great care. Always prefer safe abstractions when possible.
