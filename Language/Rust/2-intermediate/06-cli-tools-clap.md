# Command-Line Tools with Clap

## Setting Up Clap

### Cargo.toml Configuration
```toml
[package]
name = "cli_tool"
version = "0.1.0"
edition = "2021"

[dependencies]
clap = { version = "4.0", features = ["derive"] }
```

## Basic CLI with Derive API

### Simple Command-Line Tool
```rust
use clap::Parser;

#[derive(Parser)]
#[command(name = "mytool")]
#[command(about = "A simple CLI tool")]
#[command(version)]
struct Cli {
    /// Input file to process
    #[arg(short, long)]
    input: String,
    
    /// Output file (optional)
    #[arg(short, long)]
    output: Option<String>,
    
    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
    
    /// Number of iterations
    #[arg(short, long, default_value = "1")]
    count: u32,
}

fn main() {
    let cli = Cli::parse();
    
    if cli.verbose {
        println!("Processing {} with {} iterations", cli.input, cli.count);
    }
    
    // Process the file
    process_file(&cli.input, cli.output.as_deref(), cli.count);
}

fn process_file(input: &str, output: Option<&str>, count: u32) {
    println!("Processing {} {} times", input, count);
    if let Some(out) = output {
        println!("Output will go to: {}", out);
    }
}
```

## Advanced CLI Features

### Subcommands
```rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "git-like")]
#[command(about = "A git-like CLI tool")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Add files to staging
    Add {
        /// Files to add
        files: Vec<String>,
        /// Add all files
        #[arg(short, long)]
        all: bool,
    },
    /// Commit changes
    Commit {
        /// Commit message
        #[arg(short, long)]
        message: String,
        /// Amend last commit
        #[arg(short, long)]
        amend: bool,
    },
    /// Show status
    Status {
        /// Show short format
        #[arg(short, long)]
        short: bool,
    },
}

fn main() {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Add { files, all } => {
            if all {
                println!("Adding all files");
            } else {
                println!("Adding files: {:?}", files);
            }
        }
        Commands::Commit { message, amend } => {
            if amend {
                println!("Amending commit with: {}", message);
            } else {
                println!("Creating commit: {}", message);
            }
        }
        Commands::Status { short } => {
            if short {
                println!("Short status");
            } else {
                println!("Full status");
            }
        }
    }
}
```

### Validation and Custom Types
```rust
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
struct Cli {
    /// Input file (must exist)
    #[arg(short, long, value_parser = validate_file_exists)]
    input: PathBuf,
    
    /// Port number
    #[arg(short, long, value_parser = clap::value_parser!(u16).range(1024..))]
    port: u16,
    
    /// Log level
    #[arg(short, long, value_enum, default_value = "info")]
    level: LogLevel,
}

#[derive(clap::ValueEnum, Clone)]
enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

fn validate_file_exists(s: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(s);
    if path.exists() {
        Ok(path)
    } else {
        Err(format!("File '{}' does not exist", s))
    }
}

fn main() {
    let cli = Cli::parse();
    println!("Input: {:?}", cli.input);
    println!("Port: {}", cli.port);
    println!("Level: {:?}", cli.level);
}
```

## Real-World CLI Example

### Text Processor Tool
```rust
use clap::Parser;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "textproc")]
#[command(about = "Process text files")]
struct Cli {
    /// Input file
    input: PathBuf,
    
    /// Output file (stdout if not specified)
    #[arg(short, long)]
    output: Option<PathBuf>,
    
    /// Convert to uppercase
    #[arg(short, long)]
    uppercase: bool,
    
    /// Convert to lowercase
    #[arg(short, long)]
    lowercase: bool,
    
    /// Count lines, words, characters
    #[arg(short, long)]
    count: bool,
    
    /// Remove empty lines
    #[arg(long)]
    remove_empty: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    
    // Read input file
    let content = fs::read_to_string(&cli.input)
        .map_err(|e| format!("Failed to read {}: {}", cli.input.display(), e))?;
    
    // Process content
    let processed = process_text(&content, &cli);
    
    // Write output
    match cli.output {
        Some(output_path) => {
            fs::write(&output_path, &processed)
                .map_err(|e| format!("Failed to write {}: {}", output_path.display(), e))?;
            println!("Output written to {}", output_path.display());
        }
        None => {
            print!("{}", processed);
        }
    }
    
    Ok(())
}

fn process_text(content: &str, cli: &Cli) -> String {
    let mut result = content.to_string();
    
    // Apply transformations
    if cli.uppercase {
        result = result.to_uppercase();
    } else if cli.lowercase {
        result = result.to_lowercase();
    }
    
    if cli.remove_empty {
        result = result
            .lines()
            .filter(|line| !line.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n");
    }
    
    if cli.count {
        let lines = result.lines().count();
        let words = result.split_whitespace().count();
        let chars = result.chars().count();
        
        println!("Statistics:");
        println!("  Lines: {}", lines);
        println!("  Words: {}", words);
        println!("  Characters: {}", chars);
        println!();
    }
    
    result
}
```

## Environment Variables and Config

### Combining CLI Args with Environment Variables
```rust
use clap::Parser;
use std::env;

#[derive(Parser)]
struct Cli {
    /// Database URL
    #[arg(long, env = "DATABASE_URL")]
    database_url: String,
    
    /// API Key
    #[arg(long, env = "API_KEY")]
    api_key: Option<String>,
    
    /// Debug mode
    #[arg(long, env = "DEBUG")]
    debug: bool,
}

fn main() {
    let cli = Cli::parse();
    
    println!("Database: {}", cli.database_url);
    println!("Debug: {}", cli.debug);
    
    if let Some(key) = cli.api_key {
        println!("API Key provided");
    }
}
```

## Error Handling in CLI Apps

### Graceful Error Handling
```rust
use clap::Parser;
use std::process;

#[derive(Parser)]
struct Cli {
    file: String,
}

fn main() {
    let cli = Cli::parse();
    
    if let Err(e) = run(&cli) {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

fn run(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(&cli.file)?;
    println!("File contains {} characters", content.len());
    Ok(())
}
```

This provides a solid foundation for building robust command-line tools with Rust and clap.
