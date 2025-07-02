# Collections (Vec, HashMap, etc.)

## Vectors (Vec<T>)

### Creating and Using Vectors
```rust
fn main() {
    // Creating vectors
    let mut v: Vec<i32> = Vec::new();
    let mut v2 = vec![1, 2, 3, 4, 5];  // vec! macro
    
    // Adding elements
    v.push(5);
    v.push(6);
    v.push(7);
    v.push(8);
    
    // Accessing elements
    let third: &i32 = &v2[2];          // Panics if out of bounds
    println!("The third element is {}", third);
    
    match v2.get(2) {                  // Returns Option<&T>
        Some(third) => println!("The third element is {}", third),
        None => println!("There is no third element."),
    }
}
```

### Iterating Over Vectors
```rust
fn main() {
    let v = vec![100, 32, 57];
    
    // Immutable references
    for i in &v {
        println!("{}", i);
    }
    
    // Mutable references
    let mut v2 = vec![100, 32, 57];
    for i in &mut v2 {
        *i += 50;  // Dereference to change value
    }
    
    // Taking ownership
    for i in v2 {
        println!("{}", i);
    }
    // v2 is no longer accessible here
}
```

### Vector Methods
```rust
fn main() {
    let mut v = vec![1, 2, 3, 4, 5];
    
    // Length and capacity
    println!("Length: {}", v.len());
    println!("Capacity: {}", v.capacity());
    println!("Is empty: {}", v.is_empty());
    
    // Removing elements
    let last = v.pop();                // Returns Option<T>
    println!("Popped: {:?}", last);
    
    v.remove(1);                       // Remove by index
    v.retain(|&x| x > 2);             // Keep elements matching predicate
    
    // Other useful methods
    v.insert(0, 0);                   // Insert at index
    v.clear();                        // Remove all elements
    v.reverse();                      // Reverse in place
    v.sort();                         // Sort in place
}
```

### Storing Different Types with Enums
```rust
enum SpreadsheetCell {
    Int(i32),
    Float(f64),
    Text(String),
}

fn main() {
    let row = vec![
        SpreadsheetCell::Int(3),
        SpreadsheetCell::Text(String::from("blue")),
        SpreadsheetCell::Float(10.12),
    ];
    
    for cell in &row {
        match cell {
            SpreadsheetCell::Int(i) => println!("Integer: {}", i),
            SpreadsheetCell::Float(f) => println!("Float: {}", f),
            SpreadsheetCell::Text(s) => println!("Text: {}", s),
        }
    }
}
```

## Hash Maps (HashMap<K, V>)

### Creating and Using Hash Maps
```rust
use std::collections::HashMap;

fn main() {
    // Creating hash maps
    let mut scores = HashMap::new();
    scores.insert(String::from("Blue"), 10);
    scores.insert(String::from("Yellow"), 50);
    
    // From vectors
    let teams = vec![String::from("Blue"), String::from("Yellow")];
    let initial_scores = vec![10, 50];
    let scores2: HashMap<_, _> = teams.into_iter()
        .zip(initial_scores.into_iter())
        .collect();
    
    // Accessing values
    let team_name = String::from("Blue");
    let score = scores.get(&team_name);  // Returns Option<&V>
    
    match score {
        Some(s) => println!("Score: {}", s),
        None => println!("Team not found"),
    }
}
```

### Ownership and Hash Maps
```rust
use std::collections::HashMap;

fn main() {
    let field_name = String::from("Favorite color");
    let field_value = String::from("Blue");
    
    let mut map = HashMap::new();
    map.insert(field_name, field_value);
    // field_name and field_value are invalid at this point
    
    // For types that implement Copy (like i32), values are copied
    let mut map2 = HashMap::new();
    let key = 1;
    let value = 42;
    map2.insert(key, value);
    // key and value are still valid
}
```

### Updating Hash Maps
```rust
use std::collections::HashMap;

fn main() {
    let mut scores = HashMap::new();
    
    // Overwriting values
    scores.insert(String::from("Blue"), 10);
    scores.insert(String::from("Blue"), 25);  // Overwrites
    
    // Only insert if key doesn't exist
    scores.entry(String::from("Yellow")).or_insert(50);
    scores.entry(String::from("Blue")).or_insert(100);  // Won't overwrite
    
    // Update based on old value
    let text = "hello world wonderful world";
    let mut map = HashMap::new();
    
    for word in text.split_whitespace() {
        let count = map.entry(word).or_insert(0);
        *count += 1;  // Dereference the mutable reference
    }
    
    println!("{:?}", map);
}
```

### Iterating Over Hash Maps
```rust
use std::collections::HashMap;

fn main() {
    let mut scores = HashMap::new();
    scores.insert(String::from("Blue"), 10);
    scores.insert(String::from("Yellow"), 50);
    
    // Iterate over key-value pairs
    for (key, value) in &scores {
        println!("{}: {}", key, value);
    }
    
    // Iterate over keys
    for key in scores.keys() {
        println!("Key: {}", key);
    }
    
    // Iterate over values
    for value in scores.values() {
        println!("Value: {}", value);
    }
    
    // Mutable iteration
    for (key, value) in &mut scores {
        *value *= 2;
    }
}
```

## Other Collections

### VecDeque (Double-ended queue)
```rust
use std::collections::VecDeque;

fn main() {
    let mut deque = VecDeque::new();
    
    // Add to both ends
    deque.push_back(1);
    deque.push_front(0);
    deque.push_back(2);
    
    // Remove from both ends
    println!("Front: {:?}", deque.pop_front());
    println!("Back: {:?}", deque.pop_back());
    
    println!("Remaining: {:?}", deque);
}
```

### HashSet
```rust
use std::collections::HashSet;

fn main() {
    let mut books = HashSet::new();
    
    books.insert("A Game of Thrones");
    books.insert("To Kill a Mockingbird");
    books.insert("The Lord of the Rings");
    
    // Check for membership
    if books.contains("To Kill a Mockingbird") {
        println!("We have that book!");
    }
    
    // Set operations
    let fiction: HashSet<&str> = ["1984", "Animal Farm"].iter().cloned().collect();
    let scifi: HashSet<&str> = ["1984", "Dune"].iter().cloned().collect();
    
    // Union
    let all_books: HashSet<_> = fiction.union(&scifi).collect();
    println!("All books: {:?}", all_books);
    
    // Intersection
    let common: HashSet<_> = fiction.intersection(&scifi).collect();
    println!("Common books: {:?}", common);
}
```

### BTreeMap and BTreeSet (Ordered collections)
```rust
use std::collections::{BTreeMap, BTreeSet};

fn main() {
    // BTreeMap keeps keys sorted
    let mut map = BTreeMap::new();
    map.insert(3, "three");
    map.insert(1, "one");
    map.insert(2, "two");
    
    for (key, value) in &map {
        println!("{}: {}", key, value);  // Prints in order: 1, 2, 3
    }
    
    // BTreeSet keeps elements sorted
    let mut set = BTreeSet::new();
    set.insert(3);
    set.insert(1);
    set.insert(2);
    
    for value in &set {
        println!("{}", value);  // Prints in order: 1, 2, 3
    }
}
```

## Common Collection Patterns

### Building Collections from Iterators
```rust
fn main() {
    let numbers = vec![1, 2, 3, 4, 5];
    
    // Filter and collect
    let evens: Vec<i32> = numbers.iter()
        .filter(|&&x| x % 2 == 0)
        .cloned()
        .collect();
    
    // Map and collect
    let doubled: Vec<i32> = numbers.iter()
        .map(|&x| x * 2)
        .collect();
    
    // Collect into different types
    use std::collections::HashSet;
    let unique_numbers: HashSet<i32> = numbers.into_iter().collect();
}
```

### Error Handling with Collections
```rust
fn main() {
    let strings = vec!["1", "2", "not_a_number", "4"];
    
    // Collect Results - stops at first error
    let numbers: Result<Vec<i32>, _> = strings.iter()
        .map(|s| s.parse::<i32>())
        .collect();
    
    match numbers {
        Ok(nums) => println!("All parsed: {:?}", nums),
        Err(e) => println!("Parse error: {}", e),
    }
    
    // Filter successful parses
    let valid_numbers: Vec<i32> = strings.iter()
        .filter_map(|s| s.parse().ok())
        .collect();
    
    println!("Valid numbers: {:?}", valid_numbers);
}
```

Understanding these collections and their performance characteristics is crucial for writing efficient Rust programs.
