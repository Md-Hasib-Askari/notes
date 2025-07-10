# File I/O and Text Processing

## Basic File Operations

### Reading Files
```rust
use std::fs;
use std::io::{self, Read};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read entire file to string
    let content = fs::read_to_string("input.txt")?;
    println!("File content:\n{}", content);
    
    // Read file as bytes
    let bytes = fs::read("input.txt")?;
    println!("File size: {} bytes", bytes.len());
    
    // Read with explicit File handle
    let mut file = fs::File::open("input.txt")?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    
    Ok(())
}
```

### Writing Files
```rust
use std::fs;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Write string to file (overwrites)
    fs::write("output.txt", "Hello, world!")?;
    
    // Append to file
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("log.txt")?;
    
    writeln!(file, "New log entry")?;
    
    // Write bytes
    let data = b"Binary data";
    fs::write("binary.dat", data)?;
    
    Ok(())
}
```

### File Metadata and Operations
```rust
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new("example.txt");
    
    // Check if file exists
    if path.exists() {
        println!("File exists");
        
        // Get metadata
        let metadata = fs::metadata(path)?;
        println!("File size: {} bytes", metadata.len());
        println!("Is file: {}", metadata.is_file());
        println!("Is directory: {}", metadata.is_dir());
        println!("Read-only: {}", metadata.permissions().readonly());
        
        // Get modification time
        if let Ok(modified) = metadata.modified() {
            println!("Modified: {:?}", modified);
        }
    }
    
    // Copy file
    fs::copy("source.txt", "destination.txt")?;
    
    // Rename/move file
    fs::rename("old_name.txt", "new_name.txt")?;
    
    // Remove file
    fs::remove_file("temp.txt").ok(); // Ignore error if file doesn't exist
    
    Ok(())
}
```

## Directory Operations

### Working with Directories
```rust
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create directory
    fs::create_dir("new_dir")?;
    fs::create_dir_all("path/to/nested/dir")?; // Create parent dirs too
    
    // Read directory contents
    let entries = fs::read_dir(".")?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_dir() {
            println!("Directory: {}", path.display());
        } else {
            println!("File: {}", path.display());
        }
    }
    
    // Remove directory
    fs::remove_dir("empty_dir")?;           // Must be empty
    fs::remove_dir_all("dir_with_contents")?; // Recursive removal
    
    Ok(())
}
```

### Recursive Directory Walking
```rust
use std::fs;
use std::path::Path;

fn walk_dir(dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_dir() {
                println!("Directory: {}", path.display());
                walk_dir(&path)?; // Recursive call
            } else {
                println!("File: {}", path.display());
            }
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    walk_dir(Path::new("."))?;
    Ok(())
}
```

## Text Processing Patterns

### Line-by-Line Processing
```rust
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open("large_file.txt")?;
    let reader = BufReader::new(file);
    
    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        
        // Process each line
        if line.contains("ERROR") {
            println!("Error on line {}: {}", line_num + 1, line);
        }
    }
    
    Ok(())
}
```

### Text Analysis and Transformation
```rust
use std::collections::HashMap;
use std::fs;

fn analyze_text(content: &str) -> TextStats {
    let lines: Vec<&str> = content.lines().collect();
    let words: Vec<&str> = content.split_whitespace().collect();
    
    let mut word_freq = HashMap::new();
    for word in &words {
        let cleaned = word.to_lowercase()
            .trim_matches(|c: char| !c.is_alphabetic())
            .to_string();
        
        if !cleaned.is_empty() {
            *word_freq.entry(cleaned).or_insert(0) += 1;
        }
    }
    
    TextStats {
        lines: lines.len(),
        words: words.len(),
        characters: content.chars().count(),
        word_frequency: word_freq,
    }
}

struct TextStats {
    lines: usize,
    words: usize,
    characters: usize,
    word_frequency: HashMap<String, usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let content = fs::read_to_string("document.txt")?;
    let stats = analyze_text(&content);
    
    println!("Lines: {}", stats.lines);
    println!("Words: {}", stats.words);
    println!("Characters: {}", stats.characters);
    
    // Top 10 words
    let mut freq_vec: Vec<_> = stats.word_frequency.iter().collect();
    freq_vec.sort_by(|a, b| b.1.cmp(a.1));
    
    println!("\nTop 10 words:");
    for (word, count) in freq_vec.iter().take(10) {
        println!("{}: {}", word, count);
    }
    
    Ok(())
}
```

### CSV Processing
```rust
use std::fs::File;
use std::io::{BufRead, BufReader};

fn process_csv(filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    
    // Read header
    if let Some(header_line) = lines.next() {
        let header = header_line?;
        let headers: Vec<&str> = header.split(',').collect();
        println!("Headers: {:?}", headers);
        
        // Process data rows
        for (row_num, line) in lines.enumerate() {
            let line = line?;
            let fields: Vec<&str> = line.split(',').collect();
            
            if fields.len() != headers.len() {
                eprintln!("Warning: Row {} has {} fields, expected {}", 
                         row_num + 2, fields.len(), headers.len());
                continue;
            }
            
            // Create record map
            let record: Vec<(&str, &str)> = headers.iter()
                .zip(fields.iter())
                .map(|(h, f)| (*h, *f))
                .collect();
            
            println!("Row {}: {:?}", row_num + 2, record);
        }
    }
    
    Ok(())
}
```

## File Watching and Real-time Processing

### Simple File Monitor
```rust
use std::fs;
use std::time::{Duration, SystemTime};
use std::thread;

fn monitor_file(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut last_modified = SystemTime::UNIX_EPOCH;
    
    loop {
        if let Ok(metadata) = fs::metadata(path) {
            if let Ok(modified) = metadata.modified() {
                if modified > last_modified {
                    println!("File {} changed!", path);
                    last_modified = modified;
                    
                    // Process the changed file
                    let content = fs::read_to_string(path)?;
                    println!("New content length: {}", content.len());
                }
            }
        }
        
        thread::sleep(Duration::from_secs(1));
    }
}
```

## Error Handling for File Operations

### Robust File Processing
```rust
use std::fs;
use std::io;
use std::path::Path;

#[derive(Debug)]
enum FileError {
    NotFound(String),
    PermissionDenied(String),
    InvalidFormat(String),
    Other(String),
}

impl From<io::Error> for FileError {
    fn from(error: io::Error) -> Self {
        match error.kind() {
            io::ErrorKind::NotFound => FileError::NotFound(error.to_string()),
            io::ErrorKind::PermissionDenied => FileError::PermissionDenied(error.to_string()),
            _ => FileError::Other(error.to_string()),
        }
    }
}

fn safe_read_file(path: &Path) -> Result<String, FileError> {
    if !path.exists() {
        return Err(FileError::NotFound(format!("File {} does not exist", path.display())));
    }
    
    let content = fs::read_to_string(path)?;
    
    if content.is_empty() {
        return Err(FileError::InvalidFormat("File is empty".to_string()));
    }
    
    Ok(content)
}

fn main() {
    match safe_read_file(Path::new("example.txt")) {
        Ok(content) => println!("Successfully read {} characters", content.len()),
        Err(e) => eprintln!("Error reading file: {:?}", e),
    }
}
```

## Temporary Files and Cleanup

### Working with Temporary Files
```rust
use std::fs;
use std::io::Write;
use std::env;

fn create_temp_file() -> Result<(), Box<dyn std::error::Error>> {
    // Create temporary file
    let temp_dir = env::temp_dir();
    let temp_path = temp_dir.join("my_temp_file.txt");
    
    {
        let mut temp_file = fs::File::create(&temp_path)?;
        writeln!(temp_file, "Temporary data")?;
        writeln!(temp_file, "More temporary data")?;
    } // File is closed here
    
    // Use the temporary file
    let content = fs::read_to_string(&temp_path)?;
    println!("Temp file content: {}", content);
    
    // Clean up
    fs::remove_file(&temp_path)?;
    
    Ok(())
}
```

These patterns provide a solid foundation for file I/O and text processing tasks in Rust applications.
