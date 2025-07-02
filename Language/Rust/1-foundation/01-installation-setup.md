# Rust Installation and Setup

## Installing Rust

### Using rustup (Recommended)
```bash
# Install rustup (Rust installer and version manager)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# On Windows, download and run rustup-init.exe from https://rustup.rs/
```

### Verify Installation
```bash
rustc --version    # Rust compiler version
cargo --version    # Cargo package manager version
rustup --version   # Rustup version manager
```

## Development Environment Setup

### VS Code Extensions
- **rust-analyzer**: Language server for Rust
- **CodeLLDB**: Debugger for Rust
- **Better TOML**: TOML file syntax highlighting
- **Error Lens**: Inline error display

### Basic Configuration
```json
// .vscode/settings.json
{
    "rust-analyzer.checkOnSave.command": "cargo check",
    "rust-analyzer.cargo.loadOutDirsFromCheck": true,
    "rust-analyzer.procMacro.enable": true
}
```

## First Rust Project

### Creating a New Project
```bash
cargo new hello_world    # Creates new binary project
cd hello_world
cargo run               # Compiles and runs the project
```

### Project Structure
```
hello_world/
├── Cargo.toml          # Project configuration
├── src/
│   └── main.rs         # Main source file
└── target/             # Compiled artifacts (generated)
```

### Hello World Program
```rust
// src/main.rs
fn main() {
    println!("Hello, world!");
}
```

## Cargo Basics

### Essential Commands
```bash
cargo new project_name      # Create new project
cargo build                 # Compile project
cargo run                   # Compile and run
cargo check                 # Check for errors without building
cargo test                  # Run tests
cargo doc --open            # Generate and open documentation
```

### Cargo.toml Structure
```toml
[package]
name = "hello_world"
version = "0.1.0"
edition = "2021"

[dependencies]
# External crates go here
```

## Rust Toolchain

### Components
- **rustc**: The Rust compiler
- **cargo**: Package manager and build tool
- **rustfmt**: Code formatter
- **clippy**: Linter for catching common mistakes

### Useful Commands
```bash
rustfmt src/main.rs         # Format code
cargo fmt                   # Format entire project
cargo clippy                # Run linter
cargo clean                 # Clean build artifacts
```

## Development Workflow

1. **Create Project**: `cargo new project_name`
2. **Write Code**: Edit files in `src/`
3. **Check Syntax**: `cargo check`
4. **Test**: `cargo test`
5. **Run**: `cargo run`
6. **Format**: `cargo fmt`
7. **Lint**: `cargo clippy`

## Common Issues

### PATH Issues
```bash
# Add to shell profile (.bashrc, .zshrc, etc.)
export PATH="$HOME/.cargo/bin:$PATH"
```

### Update Rust
```bash
rustup update              # Update to latest stable
rustup show                # Show installed toolchains
```

### Switching Versions
```bash
rustup install nightly     # Install nightly version
rustup default stable      # Set default to stable
```

This foundation sets you up for productive Rust development with proper tooling and workflow.
