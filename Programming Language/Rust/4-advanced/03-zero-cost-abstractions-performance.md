# Zero-Cost Abstractions and Performance Optimization

## Overview
Rust's zero-cost abstractions allow high-level programming without runtime overhead. Learn to write performant code that compiles to efficient machine code.

## Understanding Zero-Cost Abstractions

### Iterator Optimizations
```rust
// High-level code that compiles to efficient loops
fn process_numbers(numbers: Vec<i32>) -> i32 {
    numbers
        .iter()
        .filter(|&&x| x > 0)
        .map(|&x| x * x)
        .sum()
}

// Equivalent manual loop (similar assembly output)
fn process_numbers_manual(numbers: Vec<i32>) -> i32 {
    let mut sum = 0;
    for number in numbers {
        if number > 0 {
            sum += number * number;
        }
    }
    sum
}

// Chained operations are optimized away
fn complex_iterator_chain() -> Vec<String> {
    (1..1000)
        .filter(|&x| x % 2 == 0)
        .map(|x| x * 3)
        .filter(|&x| x > 100)
        .map(|x| format!("Number: {}", x))
        .take(10)
        .collect()
}
```

### Generic Monomorphization
```rust
// Generic function
fn add<T>(a: T, b: T) -> T
where
    T: std::ops::Add<Output = T>,
{
    a + b
}

// Compiler generates specialized versions:
// fn add_i32(a: i32, b: i32) -> i32 { a + b }
// fn add_f64(a: f64, b: f64) -> f64 { a + b }

fn usage_example() {
    let int_result = add(5i32, 10i32);       // Calls specialized i32 version
    let float_result = add(5.0f64, 10.0f64); // Calls specialized f64 version
}

// Zero-cost trait objects with dynamic dispatch
trait Drawable {
    fn draw(&self);
}

struct Circle { radius: f64 }
struct Rectangle { width: f64, height: f64 }

impl Drawable for Circle {
    fn draw(&self) {
        println!("Drawing circle with radius {}", self.radius);
    }
}

impl Drawable for Rectangle {
    fn draw(&self) {
        println!("Drawing rectangle {}x{}", self.width, self.height);
    }
}

// Static dispatch (zero-cost)
fn draw_static<T: Drawable>(drawable: &T) {
    drawable.draw(); // Monomorphized at compile time
}

// Dynamic dispatch (small runtime cost)
fn draw_dynamic(drawable: &dyn Drawable) {
    drawable.draw(); // Virtual function call
}
```

## Memory Layout Optimization

### Struct Packing and Alignment
```rust
use std::mem;

// Unoptimized struct (24 bytes due to padding)
#[derive(Debug)]
struct Unoptimized {
    a: u8,     // 1 byte + 7 padding
    b: u64,    // 8 bytes
    c: u8,     // 1 byte + 7 padding
}

// Optimized struct (16 bytes)
#[derive(Debug)]
struct Optimized {
    b: u64,    // 8 bytes
    a: u8,     // 1 byte
    c: u8,     // 1 byte + 6 padding
}

// Manual alignment control
#[repr(C)]
struct ManualLayout {
    x: u8,
    y: u8,
    z: u16,
}

// Packed struct (no padding)
#[repr(packed)]
struct Packed {
    a: u8,
    b: u64,
    c: u8,
}

fn memory_layout_demo() {
    println!("Unoptimized size: {}", mem::size_of::<Unoptimized>());
    println!("Optimized size: {}", mem::size_of::<Optimized>());
    println!("Packed size: {}", mem::size_of::<Packed>());
    
    // Alignment information
    println!("u64 alignment: {}", mem::align_of::<u64>());
    println!("Optimized alignment: {}", mem::align_of::<Optimized>());
}
```

### Enum Optimization
```rust
// Rust optimizes enum layouts automatically
#[derive(Debug)]
enum OptimizedEnum {
    None,
    Some(String),
}

// Similar to Option<String> - uses null pointer optimization
// Takes only 24 bytes (size of String), not 32 bytes

// Tagged unions with manual discrimination
#[repr(u8)]
enum ManualTag {
    A = 1,
    B = 2,
    C = 3,
}

// Box optimization for recursive types
enum List<T> {
    Nil,
    Cons(T, Box<List<T>>), // Box prevents infinite size
}

fn enum_optimization_demo() {
    use std::mem;
    
    println!("Option<String> size: {}", mem::size_of::<Option<String>>());
    println!("String size: {}", mem::size_of::<String>());
    println!("OptimizedEnum size: {}", mem::size_of::<OptimizedEnum>());
}
```

## Performance Profiling

### CPU Profiling
```rust
use std::time::{Duration, Instant};

// Timing utilities
struct Timer {
    start: Instant,
    name: String,
}

impl Timer {
    fn new(name: &str) -> Self {
        Timer {
            start: Instant::now(),
            name: name.to_string(),
        }
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        println!("{}: {:?}", self.name, elapsed);
    }
}

// Macro for easy timing
macro_rules! time_it {
    ($name:expr, $block:block) => {
        {
            let _timer = Timer::new($name);
            $block
        }
    };
}

// Cache-friendly data structures
struct SoaData {
    positions_x: Vec<f32>,
    positions_y: Vec<f32>,
    velocities_x: Vec<f32>,
    velocities_y: Vec<f32>,
}

struct AosData {
    entities: Vec<Entity>,
}

#[derive(Clone)]
struct Entity {
    position_x: f32,
    position_y: f32,
    velocity_x: f32,
    velocity_y: f32,
}

impl SoaData {
    fn update_physics(&mut self, dt: f32) {
        // Cache-friendly: processes arrays sequentially
        for i in 0..self.positions_x.len() {
            self.positions_x[i] += self.velocities_x[i] * dt;
            self.positions_y[i] += self.velocities_y[i] * dt;
        }
    }
}

impl AosData {
    fn update_physics(&mut self, dt: f32) {
        // Less cache-friendly: jumps between struct fields
        for entity in &mut self.entities {
            entity.position_x += entity.velocity_x * dt;
            entity.position_y += entity.velocity_y * dt;
        }
    }
}

fn cache_performance_demo() {
    const COUNT: usize = 1_000_000;
    
    // Setup SoA data
    let mut soa = SoaData {
        positions_x: vec![0.0; COUNT],
        positions_y: vec![0.0; COUNT],
        velocities_x: vec![1.0; COUNT],
        velocities_y: vec![1.0; COUNT],
    };
    
    // Setup AoS data
    let mut aos = AosData {
        entities: vec![Entity {
            position_x: 0.0,
            position_y: 0.0,
            velocity_x: 1.0,
            velocity_y: 1.0,
        }; COUNT],
    };
    
    time_it!("SoA update", {
        soa.update_physics(0.016);
    });
    
    time_it!("AoS update", {
        aos.update_physics(0.016);
    });
}
```

### Memory Profiling
```rust
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

// Custom allocator for tracking
struct TrackingAllocator;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ret = System.alloc(layout);
        if !ret.is_null() {
            ALLOCATED.fetch_add(layout.size(), Ordering::SeqCst);
        }
        ret
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        ALLOCATED.fetch_sub(layout.size(), Ordering::SeqCst);
    }
}

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

fn current_memory_usage() -> usize {
    ALLOCATED.load(Ordering::SeqCst)
}

// Memory-efficient data structures
fn efficient_string_storage() {
    // Inefficient: many small allocations
    let mut inefficient: Vec<String> = Vec::new();
    for i in 0..1000 {
        inefficient.push(format!("String {}", i));
    }
    
    // More efficient: fewer allocations
    let mut efficient = String::with_capacity(10000);
    let mut offsets = Vec::with_capacity(1000);
    
    for i in 0..1000 {
        offsets.push(efficient.len());
        efficient.push_str(&format!("String {}", i));
        efficient.push('\0'); // delimiter
    }
}
```

## SIMD and Vectorization

### Manual SIMD Operations
```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// SIMD-accelerated dot product
#[target_feature(enable = "avx2")]
unsafe fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    assert!(a.len() % 8 == 0);
    
    let mut sum = _mm256_setzero_ps();
    
    for i in (0..a.len()).step_by(8) {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let mult = _mm256_mul_ps(a_vec, b_vec);
        sum = _mm256_add_ps(sum, mult);
    }
    
    // Horizontal sum
    let sum_high = _mm256_extractf128_ps(sum, 1);
    let sum_low = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(sum_high, sum_low);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x55));
    
    _mm_cvtss_f32(sum32)
}

// Auto-vectorization friendly code
fn dot_product_auto_vec(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum()
}

// Using portable SIMD (requires nightly)
#[cfg(feature = "portable_simd")]
use std::simd::*;

#[cfg(feature = "portable_simd")]
fn simd_add_arrays(a: &mut [f32], b: &[f32]) {
    let (prefix, middle, suffix) = a.as_simd_mut::<8>();
    let (_, b_middle, _) = b.as_simd::<8>();
    
    // Handle non-SIMD prefix/suffix
    for (a, b) in prefix.iter_mut().zip(b.iter()) {
        *a += *b;
    }
    
    // SIMD operations on middle
    for (a, b) in middle.iter_mut().zip(b_middle.iter()) {
        *a += *b;
    }
    
    // Handle remaining elements
    for (a, b) in suffix.iter_mut().zip(b.iter().skip(prefix.len() + middle.len() * 8)) {
        *a += *b;
    }
}
```

### Loop Optimizations
```rust
// Branch prediction optimization
fn optimized_conditional_sum(data: &[i32], threshold: i32) -> i32 {
    let mut sum = 0;
    
    // Separate hot and cold paths
    let (small, large): (Vec<_>, Vec<_>) = data
        .iter()
        .partition(|&&x| x < threshold);
    
    // Process predictable branches separately
    sum += small.iter().sum::<i32>();
    sum += large.iter().map(|&x| x * 2).sum::<i32>();
    
    sum
}

// Loop unrolling for better performance
fn unrolled_sum(data: &[i32]) -> i32 {
    let mut sum = 0;
    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();
    
    // Process 4 elements at a time
    for chunk in chunks {
        sum += chunk[0] + chunk[1] + chunk[2] + chunk[3];
    }
    
    // Handle remaining elements
    for &value in remainder {
        sum += value;
    }
    
    sum
}
```

## Compile-Time Optimizations

### Const Evaluation
```rust
// Computed at compile time
const fn fibonacci(n: u32) -> u32 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

const FIB_10: u32 = fibonacci(10); // Computed at compile time

// Const generics for compile-time array operations
fn array_sum<const N: usize>(arr: [i32; N]) -> i32 {
    let mut sum = 0;
    let mut i = 0;
    while i < N {
        sum += arr[i];
        i += 1;
    }
    sum
}

// Compile-time string operations
const fn const_string_len(s: &str) -> usize {
    s.len()
}

const MESSAGE: &str = "Hello, World!";
const MESSAGE_LEN: usize = const_string_len(MESSAGE);
```

### Inline Assembly for Critical Paths
```rust
use std::arch::asm;

// Direct assembly for maximum control
#[inline(always)]
unsafe fn fast_popcount(x: u64) -> u32 {
    let result: u64;
    asm!(
        "popcnt {}, {}",
        out(reg) result,
        in(reg) x,
        options(nomem, nostack, pure)
    );
    result as u32
}

// Compiler intrinsics (preferred when available)
fn intrinsic_popcount(x: u64) -> u32 {
    x.count_ones()
}
```

## Benchmarking and Testing

### Criterion Benchmarks
```rust
// In benches/performance.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn fibonacci_recursive(n: u64) -> u64 {
    match n {
        0 => 1,
        1 => 1,
        n => fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2),
    }
}

fn fibonacci_iterative(n: u64) -> u64 {
    let mut a = 0;
    let mut b = 1;
    for _ in 0..n {
        let temp = a + b;
        a = b;
        b = temp;
    }
    b
}

fn bench_fibonacci(c: &mut Criterion) {
    c.bench_function("fibonacci recursive", |b| {
        b.iter(|| fibonacci_recursive(black_box(20)))
    });
    
    c.bench_function("fibonacci iterative", |b| {
        b.iter(|| fibonacci_iterative(black_box(20)))
    });
}

criterion_group!(benches, bench_fibonacci);
criterion_main!(benches);
```

### Performance Testing
```rust
#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn test_performance_regression() {
        let data: Vec<i32> = (0..1_000_000).collect();
        
        let start = Instant::now();
        let _result = optimized_sum(&data);
        let duration = start.elapsed();
        
        // Ensure performance doesn't regress
        assert!(duration.as_millis() < 100, "Performance regression detected");
    }
    
    fn optimized_sum(data: &[i32]) -> i64 {
        data.iter().map(|&x| x as i64).sum()
    }
}
```

## Best Practices

### Performance Guidelines
1. **Profile First**: Measure before optimizing
2. **Hot Path Focus**: Optimize the 20% that matters
3. **Cache Locality**: Arrange data for sequential access
4. **Avoid Allocations**: Reuse memory when possible
5. **Use Iterators**: Let compiler optimize loops
6. **Const Everything**: Move computation to compile time
7. **SIMD When Appropriate**: Use for data-parallel operations

### Common Anti-patterns
```rust
// Avoid: Unnecessary allocations in loops
fn bad_string_processing(items: &[&str]) -> String {
    let mut result = String::new();
    for item in items {
        result = result + item; // Creates new string each time
    }
    result
}

// Better: Efficient string building
fn good_string_processing(items: &[&str]) -> String {
    let total_len: usize = items.iter().map(|s| s.len()).sum();
    let mut result = String::with_capacity(total_len);
    for item in items {
        result.push_str(item);
    }
    result
}

// Avoid: Unnecessary bounds checking
fn bad_array_access(data: &[i32], indices: &[usize]) -> Vec<i32> {
    indices.iter().map(|&i| data[i]).collect() // Bounds check each access
}

// Better: Use iterators or unsafe when appropriate
fn good_array_access(data: &[i32], indices: &[usize]) -> Vec<i32> {
    indices.iter()
        .filter_map(|&i| data.get(i).copied())
        .collect()
}
```

## Key Takeaways

1. **Zero-Cost Abstractions**: High-level code compiles to efficient machine code
2. **Memory Layout**: Structure data for cache efficiency
3. **SIMD**: Use vectorization for data-parallel operations
4. **Const Evaluation**: Move computation to compile time
5. **Profiling**: Measure to identify bottlenecks
6. **Iterators**: Leverage compiler optimizations
7. **Benchmarking**: Validate performance improvements
