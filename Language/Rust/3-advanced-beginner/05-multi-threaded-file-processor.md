# Multi-threaded File Processor

## Project Overview
Build a file processor that can handle multiple files concurrently using threads and channels.

## Core Features
- Process multiple files simultaneously
- Thread pool for efficient resource management
- Progress reporting and error handling
- Various processing operations (counting lines, word frequency, etc.)

## Basic Implementation

### Main Structure
```rust
use std::sync::mpsc;
use std::thread;
use std::fs;
use std::path::Path;
use std::collections::HashMap;

#[derive(Debug)]
pub enum Task {
    ProcessFile(String),
    Shutdown,
}

#[derive(Debug)]
pub enum Result {
    FileProcessed { path: String, lines: usize, words: usize },
    Error { path: String, error: String },
}

pub struct FileProcessor {
    workers: Vec<thread::JoinHandle<()>>,
    sender: mpsc::Sender<Task>,
}

impl FileProcessor {
    pub fn new(num_workers: usize) -> Self {
        let (sender, receiver) = mpsc::channel();
        let receiver = std::sync::Arc::new(std::sync::Mutex::new(receiver));
        
        let mut workers = Vec::with_capacity(num_workers);
        
        for id in 0..num_workers {
            let receiver = std::sync::Arc::clone(&receiver);
            
            let handle = thread::spawn(move || {
                loop {
                    let task = receiver.lock().unwrap().recv().unwrap();
                    
                    match task {
                        Task::ProcessFile(path) => {
                            println!("Worker {} processing {}", id, path);
                            // Process file here
                            process_file(&path);
                        }
                        Task::Shutdown => {
                            println!("Worker {} shutting down", id);
                            break;
                        }
                    }
                }
            });
            
            workers.push(handle);
        }
        
        FileProcessor { workers, sender }
    }
    
    pub fn process(&self, file_path: String) {
        self.sender.send(Task::ProcessFile(file_path)).unwrap();
    }
    
    pub fn shutdown(self) {
        for _ in &self.workers {
            self.sender.send(Task::Shutdown).unwrap();
        }
        
        for worker in self.workers {
            worker.join().unwrap();
        }
    }
}

fn process_file(path: &str) -> Result<(usize, usize), std::io::Error> {
    let content = fs::read_to_string(path)?;
    let lines = content.lines().count();
    let words = content.split_whitespace().count();
    
    println!("File: {} - Lines: {}, Words: {}", path, lines, words);
    Ok((lines, words))
}
```

### Enhanced Version with Results Channel
```rust
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::fs;
use std::time::Instant;

pub struct FileProcessorWithResults {
    workers: Vec<thread::JoinHandle<()>>,
    task_sender: mpsc::Sender<Task>,
    result_receiver: mpsc::Receiver<ProcessResult>,
}

#[derive(Debug)]
pub struct ProcessResult {
    pub path: String,
    pub lines: usize,
    pub words: usize,
    pub processing_time: std::time::Duration,
}

impl FileProcessorWithResults {
    pub fn new(num_workers: usize) -> Self {
        let (task_sender, task_receiver) = mpsc::channel();
        let (result_sender, result_receiver) = mpsc::channel();
        
        let task_receiver = Arc::new(Mutex::new(task_receiver));
        let mut workers = Vec::with_capacity(num_workers);
        
        for id in 0..num_workers {
            let task_receiver = Arc::clone(&task_receiver);
            let result_sender = result_sender.clone();
            
            let handle = thread::spawn(move || {
                loop {
                    let task = match task_receiver.lock().unwrap().recv() {
                        Ok(task) => task,
                        Err(_) => break, // Channel closed
                    };
                    
                    match task {
                        Task::ProcessFile(path) => {
                            let start = Instant::now();
                            match process_file_detailed(&path) {
                                Ok((lines, words)) => {
                                    let result = ProcessResult {
                                        path,
                                        lines,
                                        words,
                                        processing_time: start.elapsed(),
                                    };
                                    let _ = result_sender.send(result);
                                }
                                Err(e) => {
                                    eprintln!("Error processing {}: {}", path, e);
                                }
                            }
                        }
                        Task::Shutdown => break,
                    }
                }
            });
            
            workers.push(handle);
        }
        
        FileProcessorWithResults {
            workers,
            task_sender,
            result_receiver,
        }
    }
    
    pub fn process_files(&self, file_paths: Vec<String>) {
        for path in file_paths {
            self.task_sender.send(Task::ProcessFile(path)).unwrap();
        }
    }
    
    pub fn get_results(&self) -> impl Iterator<Item = ProcessResult> + '_ {
        self.result_receiver.try_iter()
    }
    
    pub fn shutdown(self) {
        drop(self.task_sender); // Close the channel
        
        for worker in self.workers {
            worker.join().unwrap();
        }
    }
}

fn process_file_detailed(path: &str) -> std::io::Result<(usize, usize)> {
    let content = fs::read_to_string(path)?;
    let lines = content.lines().count();
    let words = content.split_whitespace().count();
    Ok((lines, words))
}
```

### Usage Example
```rust
fn main() -> std::io::Result<()> {
    let files = vec![
        "file1.txt".to_string(),
        "file2.txt".to_string(),
        "file3.txt".to_string(),
    ];
    
    let processor = FileProcessorWithResults::new(4);
    processor.process_files(files);
    
    // Collect results
    thread::sleep(std::time::Duration::from_millis(100));
    
    for result in processor.get_results() {
        println!("Processed: {} in {:?}", result.path, result.processing_time);
    }
    
    processor.shutdown();
    Ok(())
}
```

## Advanced Features

### Progress Tracking
```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub struct ProgressTracker {
    total: Arc<AtomicUsize>,
    completed: Arc<AtomicUsize>,
}

impl ProgressTracker {
    pub fn new(total: usize) -> Self {
        Self {
            total: Arc::new(AtomicUsize::new(total)),
            completed: Arc::new(AtomicUsize::new(0)),
        }
    }
    
    pub fn increment(&self) {
        self.completed.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn progress(&self) -> f64 {
        let total = self.total.load(Ordering::Relaxed);
        let completed = self.completed.load(Ordering::Relaxed);
        if total == 0 { 0.0 } else { completed as f64 / total as f64 }
    }
}
```

### CLI Interface with Clap
```rust
use clap::{App, Arg};

fn main() {
    let matches = App::new("File Processor")
        .version("1.0")
        .about("Multi-threaded file processor")
        .arg(Arg::with_name("files")
            .help("Files to process")
            .required(true)
            .multiple(true))
        .arg(Arg::with_name("workers")
            .short("w")
            .long("workers")
            .value_name("NUMBER")
            .help("Number of worker threads")
            .default_value("4"))
        .get_matches();
    
    let files: Vec<String> = matches.values_of("files")
        .unwrap()
        .map(|s| s.to_string())
        .collect();
    
    let num_workers: usize = matches.value_of("workers")
        .unwrap()
        .parse()
        .expect("Invalid number of workers");
    
    println!("Processing {} files with {} workers", files.len(), num_workers);
    
    let processor = FileProcessorWithResults::new(num_workers);
    processor.process_files(files);
    
    // Wait and collect results
    thread::sleep(std::time::Duration::from_secs(1));
    
    for result in processor.get_results() {
        println!("{:?}", result);
    }
    
    processor.shutdown();
}
```

## Key Learning Points

1. **Thread Pool Pattern**: Efficient resource management
2. **Channel Communication**: Safe data passing between threads
3. **Shared State**: Using Arc and Mutex for shared data
4. **Error Handling**: Graceful error propagation in concurrent code
5. **Resource Cleanup**: Proper shutdown procedures

## Extensions
- Add different processing operations (hash calculation, compression)
- Implement recursive directory processing
- Add configuration file support
- Create a web interface for monitoring progress
- Benchmark different thread pool sizes
