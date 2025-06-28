# Phase 3: Intermediate Development (Weeks 5-8)

## Overview
Advance your Tauri skills by mastering advanced APIs, backend development with Rust, and building more sophisticated applications. Learn system integration, window management, and event-driven architecture.

## Advanced Tauri APIs

### File System Advanced Operations

#### File Watchers
```javascript
import { watch } from '@tauri-apps/api/fs';

// Watch for file changes
const stopWatching = await watch(
  '/path/to/directory',
  (event) => {
    console.log('File changed:', event);
  },
  { recursive: true }
);

// Stop watching
stopWatching();
```

#### Path Manipulation
```javascript
import { 
  appDir, 
  resolve, 
  dirname, 
  basename, 
  extname 
} from '@tauri-apps/api/path';

// Get app directory
const appDirPath = await appDir();

// Resolve paths
const fullPath = await resolve('relative', 'path', 'to', 'file.txt');

// Path components
const dir = await dirname('/path/to/file.txt');
const name = await basename('/path/to/file.txt');
const ext = await extname('/path/to/file.txt');
```

#### Advanced File Operations
```rust
// Backend: Custom file operations
use std::fs;
use std::path::Path;

#[tauri::command]
async fn copy_file(source: String, destination: String) -> Result<(), String> {
    fs::copy(&source, &destination)
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
async fn get_file_metadata(path: String) -> Result<FileMetadata, String> {
    let metadata = fs::metadata(&path)
        .map_err(|e| e.to_string())?;
    
    Ok(FileMetadata {
        size: metadata.len(),
        modified: metadata.modified()
            .map_err(|e| e.to_string())?
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        is_dir: metadata.is_dir(),
    })
}
```

### System Integration

#### Shell Commands
```javascript
import { Command } from '@tauri-apps/api/shell';

// Execute shell command
const command = new Command('ls', ['-la']);
const output = await command.execute();

console.log('stdout:', output.stdout);
console.log('stderr:', output.stderr);
console.log('status:', output.code);
```

```rust
// Backend: Advanced shell operations
use std::process::Command;

#[tauri::command]
async fn run_system_command(cmd: String, args: Vec<String>) -> Result<String, String> {
    let output = Command::new(cmd)
        .args(args)
        .output()
        .map_err(|e| e.to_string())?;
    
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).to_string())
    }
}
```

#### Process Management
```rust
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

#[tauri::command]
async fn start_background_task(
    window: tauri::Window,
    app_handle: tauri::AppHandle,
) -> Result<String, String> {
    let window_clone = window.clone();
    
    thread::spawn(move || {
        for i in 0..10 {
            thread::sleep(Duration::from_secs(1));
            let _ = window_clone.emit("progress", i);
        }
        let _ = window_clone.emit("task-complete", "Task finished!");
    });
    
    Ok("Task started".to_string())
}
```

#### Global Shortcuts
```javascript
import { register, unregister } from '@tauri-apps/api/globalShortcut';

// Register global shortcut
await register('CommandOrControl+Shift+K', () => {
    console.log('Global shortcut triggered!');
});

// Unregister shortcut
await unregister('CommandOrControl+Shift+K');
```

#### System Tray Integration
```rust
// In main.rs
use tauri::{CustomMenuItem, SystemTray, SystemTrayMenu, SystemTrayEvent};

fn main() {
    let quit = CustomMenuItem::new("quit".to_string(), "Quit");
    let hide = CustomMenuItem::new("hide".to_string(), "Hide");
    let tray_menu = SystemTrayMenu::new()
        .add_item(quit)
        .add_item(hide);

    let system_tray = SystemTray::new()
        .with_menu(tray_menu);

    tauri::Builder::default()
        .system_tray(system_tray)
        .on_system_tray_event(|app, event| match event {
            SystemTrayEvent::LeftClick { .. } => {
                let window = app.get_window("main").unwrap();
                window.show().unwrap();
            }
            SystemTrayEvent::MenuItemClick { id, .. } => match id.as_str() {
                "quit" => std::process::exit(0),
                "hide" => {
                    let window = app.get_window("main").unwrap();
                    window.hide().unwrap();
                }
                _ => {}
            }
            _ => {}
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

#### Menu Creation
```rust
use tauri::{Menu, MenuItem, Submenu, CustomMenuItem};

fn create_menu() -> Menu {
    let file_menu = Submenu::new(
        "File",
        Menu::new()
            .add_item(CustomMenuItem::new("new", "New"))
            .add_item(CustomMenuItem::new("open", "Open"))
            .add_separator()
            .add_item(CustomMenuItem::new("quit", "Quit"))
    );

    Menu::new()
        .add_submenu(file_menu)
        .add_native_item(MenuItem::Edit)
        .add_native_item(MenuItem::View)
}
```

### Window Management

#### Multiple Windows
```javascript
import { WebviewWindow } from '@tauri-apps/api/window';

// Create new window
const webview = new WebviewWindow('settings', {
  url: 'settings.html',
  title: 'Settings',
  width: 600,
  height: 400,
  resizable: false,
  center: true
});

// Listen for window events
webview.once('tauri://created', () => {
  console.log('Settings window created');
});

webview.once('tauri://error', (e) => {
  console.error('Error creating window:', e);
});
```

#### Window Decorations
```javascript
import { appWindow } from '@tauri-apps/api/window';

// Remove window decorations
await appWindow.setDecorations(false);

// Set window to always on top
await appWindow.setAlwaysOnTop(true);

// Set window transparency
await appWindow.setOpacity(0.9);
```

#### Advanced Window Configuration
```json
// tauri.conf.json
{
  "tauri": {
    "windows": [
      {
        "label": "main",
        "title": "My App",
        "width": 1200,
        "height": 800,
        "minWidth": 800,
        "minHeight": 600,
        "decorations": true,
        "transparent": false,
        "alwaysOnTop": false,
        "skipTaskbar": false
      }
    ]
  }
}
```

## Backend Development (Rust)

### Tauri Commands

#### Creating Custom Commands
```rust
// Simple command
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}

// Command with complex types
#[derive(serde::Serialize, serde::Deserialize)]
struct User {
    id: u32,
    name: String,
    email: String,
}

#[tauri::command]
fn create_user(name: String, email: String) -> Result<User, String> {
    if email.contains('@') {
        Ok(User {
            id: 1,
            name,
            email,
        })
    } else {
        Err("Invalid email format".to_string())
    }
}
```

#### Command Return Types
```rust
// Return Result for error handling
#[tauri::command]
fn risky_operation() -> Result<String, String> {
    // Operation that might fail
    Ok("Success".to_string())
}

// Return Option for nullable values
#[tauri::command]
fn find_user(id: u32) -> Option<User> {
    // Return Some(user) or None
    None
}

// Return custom error types
#[derive(Debug, thiserror::Error)]
enum CustomError {
    #[error("User not found")]
    UserNotFound,
    #[error("Database error: {0}")]
    DatabaseError(String),
}

impl serde::Serialize for CustomError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        serializer.serialize_str(self.to_string().as_ref())
    }
}

#[tauri::command]
fn get_user(id: u32) -> Result<User, CustomError> {
    Err(CustomError::UserNotFound)
}
```

#### Async Commands
```rust
use tokio::time::{sleep, Duration};

#[tauri::command]
async fn async_operation() -> Result<String, String> {
    sleep(Duration::from_secs(2)).await;
    Ok("Async operation completed".to_string())
}

#[tauri::command]
async fn fetch_data_from_api() -> Result<String, String> {
    let client = reqwest::Client::new();
    let response = client
        .get("https://api.example.com/data")
        .send()
        .await
        .map_err(|e| e.to_string())?;
    
    let body = response
        .text()
        .await
        .map_err(|e| e.to_string())?;
    
    Ok(body)
}
```

### State Management

#### Application State
```rust
use std::sync::Mutex;
use tauri::State;

struct AppState {
    counter: Mutex<i32>,
}

#[tauri::command]
fn increment_counter(state: State<AppState>) -> i32 {
    let mut counter = state.counter.lock().unwrap();
    *counter += 1;
    *counter
}

#[tauri::command]
fn get_counter(state: State<AppState>) -> i32 {
    *state.counter.lock().unwrap()
}

fn main() {
    tauri::Builder::default()
        .manage(AppState {
            counter: Mutex::new(0),
        })
        .invoke_handler(tauri::generate_handler![
            increment_counter,
            get_counter
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

#### Shared State Between Commands
```rust
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

type SharedData = Arc<Mutex<HashMap<String, String>>>;

#[tauri::command]
fn set_data(key: String, value: String, data: State<SharedData>) -> Result<(), String> {
    let mut map = data.lock().map_err(|e| e.to_string())?;
    map.insert(key, value);
    Ok(())
}

#[tauri::command]
fn get_data(key: String, data: State<SharedData>) -> Result<Option<String>, String> {
    let map = data.lock().map_err(|e| e.to_string())?;
    Ok(map.get(&key).cloned())
}
```

### Events

#### Frontend to Backend Events
```javascript
// Frontend: Emit event
import { emit } from '@tauri-apps/api/event';

await emit('user-action', {
  action: 'click',
  element: 'button',
  timestamp: Date.now()
});
```

```rust
// Backend: Listen for events
use tauri::{Manager, Window};

#[tauri::command]
async fn setup_event_listeners(window: Window) {
    let window_clone = window.clone();
    window.listen("user-action", move |event| {
        println!("Received user action: {:?}", event.payload());
        // Process the event
        let _ = window_clone.emit("action-processed", "Event handled");
    });
}
```

#### Backend to Frontend Events
```rust
// Backend: Emit events
#[tauri::command]
async fn start_process(window: Window) -> Result<(), String> {
    // Emit progress updates
    for i in 0..=100 {
        let _ = window.emit("progress", i);
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }
    
    let _ = window.emit("process-complete", "Process finished!");
    Ok(())
}
```

```javascript
// Frontend: Listen for events
import { listen } from '@tauri-apps/api/event';

const unlisten = await listen('progress', (event) => {
    console.log('Progress:', event.payload);
    updateProgressBar(event.payload);
});

await listen('process-complete', (event) => {
    console.log('Process completed:', event.payload);
    showNotification('Process finished!');
});
```

## Intermediate Projects

### Project 1: Note-taking App
**Features**: Rich text editing with file persistence

#### Core Features
- Rich text editor with formatting
- File-based note storage
- Note organization with folders
- Search functionality
- Auto-save capability
- Export to different formats

#### Technical Implementation
- Use a rich text editor library (Quill, TinyMCE, etc.)
- Implement file watcher for external changes
- Create custom commands for file operations
- Add search indexing for fast searches

### Project 2: Media Player
**Features**: Audio/video playbook with playlist

#### Core Features
- Audio/video playback
- Playlist management
- Media library organization
- Equalizer and effects
- Keyboard shortcuts
- Media information display

#### Technical Implementation
- Use HTML5 media elements
- Implement custom controls
- Add media metadata reading
- Create playlist persistence
- Integrate with system media keys

### Project 3: System Monitor
**Features**: Real-time system statistics

#### Core Features
- CPU, memory, disk usage
- Process listing and management
- Network monitoring
- Performance graphs
- Alert thresholds
- Historical data storage

#### Technical Implementation
- Use system monitoring libraries in Rust
- Implement real-time data streaming
- Create responsive charts and graphs
- Add background monitoring
- Implement data persistence

## Learning Objectives

By the end of Phase 3, you should be able to:
- [ ] Implement advanced file system operations
- [ ] Integrate with system features (tray, shortcuts, menus)
- [ ] Manage multiple windows effectively
- [ ] Create complex Rust backend logic
- [ ] Handle application state properly
- [ ] Implement event-driven architecture
- [ ] Build sophisticated applications with rich features

## Best Practices

### Error Handling Strategy
```rust
// Define application-specific errors
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Custom error: {message}")]
    Custom { message: String },
}

// Implement Serialize for frontend
impl serde::Serialize for AppError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        serializer.serialize_str(self.to_string().as_ref())
    }
}
```

### Performance Considerations
- Use async operations for I/O
- Implement proper error boundaries
- Optimize bundle size
- Use efficient data structures
- Implement proper cleanup

### Security Guidelines
- Validate all input from frontend
- Use principle of least privilege
- Implement proper CSP headers
- Sanitize file paths
- Validate file operations

## Next Steps

After completing Phase 3:
- Move to Phase 4: Advanced Features
- Learn database integration
- Explore plugin development
- Master performance optimization

## Resources

### Advanced Topics
- [Tauri System Tray Guide](https://tauri.app/v1/guides/features/system-tray)
- [Window Management](https://tauri.app/v1/guides/features/window)
- [Global Shortcuts](https://tauri.app/v1/guides/features/global-shortcut)

### Rust Resources
- [Tokio Async Runtime](https://tokio.rs/)
- [Serde Serialization](https://serde.rs/)
- [Error Handling](https://doc.rust-lang.org/book/ch09-00-error-handling.html)
