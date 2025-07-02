# Structs, Enums, and Pattern Matching

## Structs

### Basic Struct Definition
```rust
struct User {
    username: String,
    email: String,
    sign_in_count: u64,
    active: bool,
}

fn main() {
    let user1 = User {
        email: String::from("user@example.com"),
        username: String::from("someuser"),
        active: true,
        sign_in_count: 1,
    };
    
    println!("User: {}", user1.username);
}
```

### Mutable Structs and Field Access
```rust
fn main() {
    let mut user1 = User {
        email: String::from("user@example.com"),
        username: String::from("someuser"),
        active: true,
        sign_in_count: 1,
    };
    
    user1.email = String::from("newemail@example.com");
}
```

### Struct Update Syntax
```rust
fn main() {
    let user1 = User {
        email: String::from("user@example.com"),
        username: String::from("someuser"),
        active: true,
        sign_in_count: 1,
    };
    
    let user2 = User {
        email: String::from("another@example.com"),
        ..user1  // Copy remaining fields from user1
    };
}
```

### Tuple Structs
```rust
struct Color(i32, i32, i32);
struct Point(i32, i32, i32);

fn main() {
    let black = Color(0, 0, 0);
    let origin = Point(0, 0, 0);
    
    println!("Red component: {}", black.0);
}
```

### Unit-Like Structs
```rust
struct AlwaysEqual;

fn main() {
    let subject = AlwaysEqual;
}
```

### Methods and Associated Functions
```rust
impl User {
    // Associated function (constructor)
    fn new(email: String, username: String) -> User {
        User {
            email,
            username,
            active: true,
            sign_in_count: 1,
        }
    }
    
    // Method
    fn is_active(&self) -> bool {
        self.active
    }
    
    // Mutating method
    fn deactivate(&mut self) {
        self.active = false;
    }
    
    // Method that takes ownership
    fn delete(self) -> String {
        format!("User {} has been deleted", self.username)
    }
}

fn main() {
    let mut user = User::new(
        String::from("user@example.com"), 
        String::from("username")
    );
    
    println!("Active: {}", user.is_active());
    user.deactivate();
    let msg = user.delete(); // user is moved here
}
```

## Enums

### Basic Enum Definition
```rust
enum IpAddrKind {
    V4,
    V6,
}

fn main() {
    let four = IpAddrKind::V4;
    let six = IpAddrKind::V6;
    
    route(four);
    route(six);
}

fn route(ip_kind: IpAddrKind) {
    // Process the IP address kind
}
```

### Enums with Data
```rust
enum IpAddr {
    V4(u8, u8, u8, u8),
    V6(String),
}

enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(i32, i32, i32),
}

fn main() {
    let home = IpAddr::V4(127, 0, 0, 1);
    let loopback = IpAddr::V6(String::from("::1"));
    
    let msg = Message::Write(String::from("hello"));
}
```

### Enum Methods
```rust
impl Message {
    fn call(&self) {
        match self {
            Message::Quit => println!("Quit"),
            Message::Move { x, y } => println!("Move to ({}, {})", x, y),
            Message::Write(text) => println!("Write: {}", text),
            Message::ChangeColor(r, g, b) => println!("Color: ({}, {}, {})", r, g, b),
        }
    }
}

fn main() {
    let msg = Message::Write(String::from("hello"));
    msg.call();
}
```

## Pattern Matching

### Basic Match Expression
```rust
enum Coin {
    Penny,
    Nickel,
    Dime,
    Quarter,
}

fn value_in_cents(coin: Coin) -> u8 {
    match coin {
        Coin::Penny => 1,
        Coin::Nickel => 5,
        Coin::Dime => 10,
        Coin::Quarter => 25,
    }
}
```

### Match with Data Extraction
```rust
#[derive(Debug)]
enum UsState {
    Alabama,
    Alaska,
    // ... etc
}

enum Coin {
    Penny,
    Nickel,
    Dime,
    Quarter(UsState),
}

fn value_in_cents(coin: Coin) -> u8 {
    match coin {
        Coin::Penny => {
            println!("Lucky penny!");
            1
        }
        Coin::Nickel => 5,
        Coin::Dime => 10,
        Coin::Quarter(state) => {
            println!("State quarter from {:?}!", state);
            25
        }
    }
}
```

### if let Syntax
```rust
fn main() {
    let some_u8_value = Some(3);
    
    // Verbose match
    match some_u8_value {
        Some(3) => println!("three"),
        _ => (),
    }
    
    // Concise if let
    if let Some(3) = some_u8_value {
        println!("three");
    }
}
```

### Match Guards
```rust
fn main() {
    let num = Some(4);
    
    match num {
        Some(x) if x < 5 => println!("less than five: {}", x),
        Some(x) => println!("{}", x),
        None => (),
    }
}
```

### Destructuring Patterns
```rust
struct Point {
    x: i32,
    y: i32,
}

fn main() {
    let p = Point { x: 0, y: 7 };
    
    // Destructuring in match
    match p {
        Point { x, y: 0 } => println!("On the x axis at {}", x),
        Point { x: 0, y } => println!("On the y axis at {}", y),
        Point { x, y } => println!("On neither axis: ({}, {})", x, y),
    }
    
    // Destructuring in let
    let Point { x, y } = p;
    println!("x: {}, y: {}", x, y);
}
```

### Pattern Matching with References
```rust
fn main() {
    let points = vec![
        Point { x: 0, y: 0 },
        Point { x: 1, y: 5 },
        Point { x: 10, y: -3 },
    ];
    
    for &Point { x, y } in &points {
        println!("Point: ({}, {})", x, y);
    }
}
```

This covers the essential concepts of structs, enums, and pattern matching that form the backbone of Rust's type system.
