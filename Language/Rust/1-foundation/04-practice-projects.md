# Practice Projects - Foundation Phase

## Project 1: Simple Calculator

### Requirements
Create a command-line calculator that performs basic arithmetic operations.

### Implementation
```rust
use std::io;

fn main() {
    println!("Simple Calculator");
    println!("Enter operation (e.g., 5 + 3): ");
    
    let mut input = String::new();
    io::stdin().read_line(&mut input)
        .expect("Failed to read line");
    
    let parts: Vec<&str> = input.trim().split_whitespace().collect();
    
    if parts.len() != 3 {
        println!("Invalid format! Use: number operator number");
        return;
    }
    
    let num1: f64 = parts[0].parse().expect("Invalid first number");
    let operator = parts[1];
    let num2: f64 = parts[2].parse().expect("Invalid second number");
    
    let result = match operator {
        "+" => num1 + num2,
        "-" => num1 - num2,
        "*" => num1 * num2,
        "/" => {
            if num2 == 0.0 {
                println!("Error: Division by zero!");
                return;
            }
            num1 / num2
        },
        _ => {
            println!("Unknown operator: {}", operator);
            return;
        }
    };
    
    println!("{} {} {} = {}", num1, operator, num2, result);
}
```

### Key Learning Points
- User input handling with `std::io`
- String parsing and error handling
- Pattern matching with `match`
- Basic arithmetic operations

## Project 2: Temperature Converter

### Requirements
Convert between Celsius, Fahrenheit, and Kelvin temperatures.

### Implementation
```rust
use std::io;

fn main() {
    loop {
        println!("\nTemperature Converter");
        println!("1. Celsius to Fahrenheit");
        println!("2. Fahrenheit to Celsius");
        println!("3. Celsius to Kelvin");
        println!("4. Kelvin to Celsius");
        println!("5. Exit");
        print!("Choose an option: ");
        
        let mut choice = String::new();
        io::stdin().read_line(&mut choice)
            .expect("Failed to read line");
        
        let choice: u32 = match choice.trim().parse() {
            Ok(num) => num,
            Err(_) => {
                println!("Please enter a valid number!");
                continue;
            }
        };
        
        match choice {
            1 => celsius_to_fahrenheit(),
            2 => fahrenheit_to_celsius(),
            3 => celsius_to_kelvin(),
            4 => kelvin_to_celsius(),
            5 => {
                println!("Goodbye!");
                break;
            },
            _ => println!("Invalid choice! Please try again."),
        }
    }
}

fn celsius_to_fahrenheit() {
    let celsius = get_temperature("Celsius");
    let fahrenheit = celsius * 9.0 / 5.0 + 32.0;
    println!("{:.2}°C = {:.2}°F", celsius, fahrenheit);
}

fn fahrenheit_to_celsius() {
    let fahrenheit = get_temperature("Fahrenheit");
    let celsius = (fahrenheit - 32.0) * 5.0 / 9.0;
    println!("{:.2}°F = {:.2}°C", fahrenheit, celsius);
}

fn celsius_to_kelvin() {
    let celsius = get_temperature("Celsius");
    let kelvin = celsius + 273.15;
    println!("{:.2}°C = {:.2}K", celsius, kelvin);
}

fn kelvin_to_celsius() {
    let kelvin = get_temperature("Kelvin");
    if kelvin < 0.0 {
        println!("Invalid temperature! Kelvin cannot be negative.");
        return;
    }
    let celsius = kelvin - 273.15;
    println!("{:.2}K = {:.2}°C", kelvin, celsius);
}

fn get_temperature(unit: &str) -> f64 {
    loop {
        println!("Enter temperature in {}: ", unit);
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)
            .expect("Failed to read line");
        
        match input.trim().parse::<f64>() {
            Ok(temp) => return temp,
            Err(_) => println!("Please enter a valid number!"),
        }
    }
}
```

### Key Learning Points
- Menu-driven program structure
- Function organization and reuse
- Input validation and error handling
- Loops and control flow

## Project 3: Number Guessing Game

### Requirements
Classic guessing game where user tries to guess a random number.

### Implementation
```rust
use std::io;
use std::cmp::Ordering;
use rand::Rng;

fn main() {
    println!("Guess the number!");
    
    let secret_number = rand::thread_rng().gen_range(1..=100);
    let mut attempts = 0;
    
    loop {
        println!("Please input your guess (1-100): ");
        
        let mut guess = String::new();
        
        io::stdin()
            .read_line(&mut guess)
            .expect("Failed to read line");
        
        let guess: u32 = match guess.trim().parse() {
            Ok(num) => {
                if num < 1 || num > 100 {
                    println!("Please enter a number between 1 and 100!");
                    continue;
                }
                num
            },
            Err(_) => {
                println!("Please enter a valid number!");
                continue;
            }
        };
        
        attempts += 1;
        
        match guess.cmp(&secret_number) {
            Ordering::Less => println!("Too small!"),
            Ordering::Greater => println!("Too big!"),
            Ordering::Equal => {
                println!("You win! The number was {} and you guessed it in {} attempts!", 
                        secret_number, attempts);
                break;
            }
        }
        
        // Give hints after 5 attempts
        if attempts == 5 {
            let difference = if secret_number > guess {
                secret_number - guess
            } else {
                guess - secret_number
            };
            
            if difference <= 5 {
                println!("Hint: You're very close!");
            } else if difference <= 15 {
                println!("Hint: You're getting warmer!");
            }
        }
    }
}
```

### Cargo.toml for rand dependency
```toml
[package]
name = "guessing_game"
version = "0.1.0"
edition = "2021"

[dependencies]
rand = "0.8"
```

### Key Learning Points
- External crate usage (rand)
- Comparison operations and `Ordering`
- Loop control with `continue` and `break`
- Input validation patterns

## Project 4: Word Counter

### Requirements
Count words, characters, and lines in text input.

### Implementation
```rust
use std::io::{self, Read};
use std::collections::HashMap;

fn main() {
    println!("Word Counter - Enter text (Ctrl+D/Ctrl+Z to finish):");
    
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)
        .expect("Failed to read input");
    
    let stats = analyze_text(&input);
    print_statistics(&stats, &input);
}

struct TextStats {
    word_count: usize,
    char_count: usize,
    line_count: usize,
    word_frequency: HashMap<String, usize>,
}

fn analyze_text(text: &str) -> TextStats {
    let char_count = text.chars().count();
    let line_count = text.lines().count();
    
    let words: Vec<String> = text
        .split_whitespace()
        .map(|word| word.to_lowercase().trim_matches(|c: char| !c.is_alphabetic()).to_string())
        .filter(|word| !word.is_empty())
        .collect();
    
    let word_count = words.len();
    
    let mut word_frequency = HashMap::new();
    for word in words {
        *word_frequency.entry(word).or_insert(0) += 1;
    }
    
    TextStats {
        word_count,
        char_count,
        line_count,
        word_frequency,
    }
}

fn print_statistics(stats: &TextStats, original_text: &str) {
    println!("\n--- Text Statistics ---");
    println!("Characters: {}", stats.char_count);
    println!("Words: {}", stats.word_count);
    println!("Lines: {}", stats.line_count);
    
    if !original_text.trim().is_empty() {
        let avg_word_length = original_text
            .split_whitespace()
            .map(|word| word.len())
            .sum::<usize>() as f64 / stats.word_count as f64;
        
        println!("Average word length: {:.2}", avg_word_length);
    }
    
    // Show most frequent words
    let mut freq_vec: Vec<_> = stats.word_frequency.iter().collect();
    freq_vec.sort_by(|a, b| b.1.cmp(a.1));
    
    println!("\nMost frequent words:");
    for (word, count) in freq_vec.iter().take(5) {
        println!("  {}: {}", word, count);
    }
}
```

### Key Learning Points
- Text processing and string manipulation
- HashMap for frequency counting
- Struct organization
- Iterator methods and functional programming concepts

## Running the Projects

```bash
# Create each project
cargo new calculator
cargo new temperature_converter
cargo new guessing_game
cargo new word_counter

# Add dependencies to Cargo.toml where needed
# Copy code into src/main.rs
# Run with:
cargo run
```

These projects reinforce the fundamental concepts while building practical, useful programs.
