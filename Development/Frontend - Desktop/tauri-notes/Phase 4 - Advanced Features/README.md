# Phase 4: Advanced Features (Weeks 9-12)

## Overview
Master advanced Tauri capabilities including database integration, security configuration, plugin development, and performance optimization. Build production-ready applications with professional features.

## Database Integration

### SQLite with Tauri

#### Setup tauri-plugin-sql
```bash
# Add to Cargo.toml
[dependencies]
tauri-plugin-sql = { version = "1.0", features = ["sqlite"] }
```

```rust
// In main.rs
use tauri_plugin_sql::{Migration, MigrationKind};

fn main() {
    let migrations = vec![
        Migration {
            version: 1,
            description: "create_initial_tables",
            sql: "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT);",
            kind: MigrationKind::Up,
        }
    ];

    tauri::Builder::default()
        .plugin(tauri_plugin_sql::Builder::default()
            .add_migrations("sqlite:mydatabase.db", migrations)
            .build())
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

#### Database Operations
```javascript
import Database from "tauri-plugin-sql-api";

// Initialize database
const db = await Database.load("sqlite:mydatabase.db");

// Create records
await db.execute(
    "INSERT INTO users (name, email) VALUES (?, ?)",
    ["John Doe", "john@example.com"]
);

// Read records
const result = await db.select(
    "SELECT * FROM users WHERE name = ?",
    ["John Doe"]
);

// Update records
await db.execute(
    "UPDATE users SET email = ? WHERE id = ?",
    ["newemail@example.com", 1]
);

// Delete records
await db.execute("DELETE FROM users WHERE id = ?", [1]);
```

#### Advanced Database Features
```javascript
// Transactions
await db.execute("BEGIN TRANSACTION");
try {
    await db.execute("INSERT INTO users (name, email) VALUES (?, ?)", ["User1", "user1@example.com"]);
    await db.execute("INSERT INTO users (name, email) VALUES (?, ?)", ["User2", "user2@example.com"]);
    await db.execute("COMMIT");
} catch (error) {
    await db.execute("ROLLBACK");
    throw error;
}

// Prepared statements for better performance
const stmt = await db.prepare("SELECT * FROM users WHERE created_at > ?");
const recentUsers = await stmt.select([new Date().toISOString()]);
```

#### Database Migrations
```rust
// Advanced migration example
let migrations = vec![
    Migration {
        version: 1,
        description: "create_users_table",
        sql: r#"
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        "#,
        kind: MigrationKind::Up,
    },
    Migration {
        version: 2,
        description: "add_user_preferences",
        sql: r#"
            CREATE TABLE user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(user_id, key)
            );
        "#,
        kind: MigrationKind::Up,
    }
];
```

### Alternative Databases

#### Embedded Databases (sled)
```toml
# Cargo.toml
[dependencies]
sled = "0.34"
serde_json = "1.0"
```

```rust
use sled::{Db, Tree};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct User {
    id: u32,
    name: String,
    email: String,
}

struct DatabaseManager {
    db: Db,
    users: Tree,
}

impl DatabaseManager {
    fn new(path: &str) -> Result<Self, sled::Error> {
        let db = sled::open(path)?;
        let users = db.open_tree("users")?;
        Ok(Self { db, users })
    }

    fn insert_user(&self, user: &User) -> Result<(), Box<dyn std::error::Error>> {
        let key = user.id.to_be_bytes();
        let value = serde_json::to_vec(user)?;
        self.users.insert(key, value)?;
        Ok(())
    }

    fn get_user(&self, id: u32) -> Result<Option<User>, Box<dyn std::error::Error>> {
        let key = id.to_be_bytes();
        if let Some(value) = self.users.get(key)? {
            let user: User = serde_json::from_slice(&value)?;
            Ok(Some(user))
        } else {
            Ok(None)
        }
    }
}
```

#### File-based Storage Solutions
```rust
use std::fs;
use std::path::Path;
use serde_json;

#[tauri::command]
async fn save_data_to_file<T: Serialize>(
    data: T,
    file_path: String,
) -> Result<(), String> {
    let json = serde_json::to_string_pretty(&data)
        .map_err(|e| e.to_string())?;
    
    fs::write(file_path, json)
        .map_err(|e| e.to_string())?;
    
    Ok(())
}

#[tauri::command]
async fn load_data_from_file<T: for<'de> Deserialize<'de>>(
    file_path: String,
) -> Result<T, String> {
    let content = fs::read_to_string(file_path)
        .map_err(|e| e.to_string())?;
    
    let data: T = serde_json::from_str(&content)
        .map_err(|e| e.to_string())?;
    
    Ok(data)
}
```

## Security & Configuration

### CSP (Content Security Policy)

#### Configuring CSP Headers
```json
// tauri.conf.json
{
  "tauri": {
    "security": {
      "csp": {
        "default-src": "'self'",
        "script-src": "'self' 'unsafe-inline'",
        "style-src": "'self' 'unsafe-inline'",
        "img-src": "'self' data: https:",
        "connect-src": "'self' https: wss:",
        "font-src": "'self'",
        "object-src": "'none'",
        "base-uri": "'self'",
        "form-action": "'self'"
      }
    }
  }
}
```

#### Advanced CSP Configuration
```json
{
  "tauri": {
    "security": {
      "csp": {
        "default-src": "'self'",
        "script-src": "'self' 'nonce-{NONCE}'",
        "style-src": "'self' 'nonce-{NONCE}'",
        "connect-src": "'self' https://api.myapp.com",
        "upgrade-insecure-requests": true
      }
    }
  }
}
```

#### Allowlist Configuration
```json
{
  "tauri": {
    "allowlist": {
      "all": false,
      "fs": {
        "readFile": true,
        "writeFile": true,
        "readDir": true,
        "createDir": true,
        "removeDir": false,
        "removeFile": false,
        "scope": ["$APPDATA/*", "$DOCUMENT/*"]
      },
      "shell": {
        "execute": true,
        "scope": [
          {
            "name": "git",
            "cmd": "git",
            "args": ["status", "add", "commit"]
          }
        ]
      },
      "protocol": {
        "asset": true,
        "assetScope": ["$APPDATA/**"]
      }
    }
  }
}
```

#### API Permissions
```rust
// Runtime permission checks
#[tauri::command]
async fn sensitive_operation(
    app_handle: tauri::AppHandle,
    window: tauri::Window,
) -> Result<String, String> {
    // Check if operation is allowed
    if !is_operation_permitted(&app_handle) {
        return Err("Operation not permitted".to_string());
    }
    
    // Perform sensitive operation
    Ok("Operation completed".to_string())
}

fn is_operation_permitted(app_handle: &tauri::AppHandle) -> bool {
    // Implement your permission logic
    true
}
```

### Code Signing

#### Certificate Management
```bash
# Windows: Create self-signed certificate for testing
New-SelfSignedCertificate -DnsName "MyApp" -Type CodeSigning -CertStoreLocation "Cert:\CurrentUser\My"

# Export certificate
$cert = Get-ChildItem -Path "Cert:\CurrentUser\My" -CodeSigningCert
Export-Certificate -Cert $cert -FilePath "MyAppCert.cer"
```

#### Signing Configuration
```json
// tauri.conf.json
{
  "tauri": {
    "bundle": {
      "windows": {
        "certificateThumbprint": "YOUR_CERTIFICATE_THUMBPRINT",
        "digestAlgorithm": "sha256",
        "timestampUrl": "http://timestamp.digicert.com"
      }
    }
  }
}
```

#### macOS Notarization
```bash
# Build and sign
cargo tauri build --target universal-apple-darwin

# Notarize (requires Apple Developer account)
xcrun notarytool submit \
  "target/universal-apple-darwin/release/bundle/dmg/MyApp_1.0.0_universal.dmg" \
  --keychain-profile "AC_PASSWORD" \
  --wait

# Staple the notarization
xcrun stapler staple "target/universal-apple-darwin/release/bundle/dmg/MyApp_1.0.0_universal.dmg"
```

## Plugin Development

### Using Existing Plugins

#### File System Extended
```bash
npm install @tauri-apps/plugin-fs-extra
```

```javascript
import { copy, move, remove } from '@tauri-apps/plugin-fs-extra';

// Copy files with options
await copy('source/path', 'destination/path', {
  overwrite: true,
  preserveTimestamps: true
});

// Move files
await move('old/path', 'new/path');

// Remove with options
await remove('path/to/remove', { recursive: true });
```

#### HTTP Client
```bash
npm install @tauri-apps/plugin-http
```

```javascript
import { fetch } from '@tauri-apps/plugin-http';

// Make HTTP requests
const response = await fetch('https://api.example.com/data', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({ key: 'value' })
});

const data = await response.json();
```

#### Store (Key-Value Storage)
```bash
npm install @tauri-apps/plugin-store
```

```javascript
import { Store } from '@tauri-apps/plugin-store';

// Create store
const store = new Store('.settings.dat');

// Set values
await store.set('user-preferences', {
  theme: 'dark',
  language: 'en'
});

// Get values
const preferences = await store.get('user-preferences');

// Save store
await store.save();
```

#### Updater
```bash
npm install @tauri-apps/plugin-updater
```

```javascript
import { check } from '@tauri-apps/plugin-updater';

// Check for updates
const update = await check();
if (update?.available) {
  console.log(`Update available: ${update.version}`);
  
  // Download and install
  await update.downloadAndInstall();
  
  // Restart app
  await relaunch();
}
```

### Creating Custom Plugins

#### Plugin Architecture
```rust
// plugins/my-plugin/src/lib.rs
use tauri::{
    plugin::{Builder, TauriPlugin},
    Runtime,
};

#[tauri::command]
async fn my_custom_command(input: String) -> Result<String, String> {
    // Plugin logic here
    Ok(format!("Processed: {}", input))
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("my-plugin")
        .invoke_handler(tauri::generate_handler![my_custom_command])
        .build()
}
```

```toml
# plugins/my-plugin/Cargo.toml
[package]
name = "tauri-plugin-my-plugin"
version = "0.1.0"
edition = "2021"

[dependencies]
tauri = { version = "1.0", features = ["api-all"] }
serde = { version = "1.0", features = ["derive"] }
```

#### Mobile Support Considerations
```rust
// plugins/my-plugin/src/mobile.rs
use tauri::{
    plugin::{PluginApi, PluginHandle},
    Runtime,
};

#[cfg(target_os = "ios")]
tauri::ios_plugin_binding!(init_plugin_my_plugin);

pub fn init<R: Runtime, C: PluginApi<R>>() -> tauri::plugin::TauriPlugin<R, C> {
    tauri::plugin::Builder::new("my-plugin")
        .setup(|app, api| {
            #[cfg(mobile)]
            {
                // Mobile-specific setup
            }
            Ok(())
        })
        .build()
}
```

#### Publishing Plugins
```toml
# Plugin Cargo.toml for publishing
[package]
name = "tauri-plugin-awesome-feature"
version = "1.0.0"
description = "An awesome Tauri plugin"
authors = ["Your Name <your.email@example.com>"]
license = "MIT"
repository = "https://github.com/username/tauri-plugin-awesome-feature"
keywords = ["tauri", "plugin"]
categories = ["gui", "api-bindings"]

[dependencies]
tauri = { version = "1.0", features = ["api-all"] }
```

## Performance Optimization

### Bundle Size Optimization

#### Tree Shaking Configuration
```javascript
// vite.config.js
import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          tauri: ['@tauri-apps/api']
        }
      }
    },
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true
      }
    }
  }
});
```

#### Code Splitting
```javascript
// Lazy loading components
import { lazy, Suspense } from 'react';

const HeavyComponent = lazy(() => import('./HeavyComponent'));

function App() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <HeavyComponent />
    </Suspense>
  );
}
```

#### Asset Optimization
```json
// tauri.conf.json
{
  "tauri": {
    "bundle": {
      "resources": ["assets/*"],
      "icon": ["icons/32x32.png", "icons/128x128.png"],
      "externalBin": []
    }
  }
}
```

### Runtime Performance

#### Efficient IPC Usage
```javascript
// Batch operations instead of individual calls
const batchOperations = async (operations) => {
  return await invoke('batch_process', { operations });
};

// Use proper error boundaries
const safeInvoke = async (command, args) => {
  try {
    return await invoke(command, args);
  } catch (error) {
    console.error(`Command ${command} failed:`, error);
    throw error;
  }
};
```

#### Background Task Management
```rust
use tokio::sync::mpsc;
use std::sync::Arc;

#[tauri::command]
async fn start_heavy_computation(
    window: tauri::Window,
) -> Result<String, String> {
    let (tx, mut rx) = mpsc::channel(100);
    let window_clone = window.clone();
    
    // Spawn background task
    tokio::spawn(async move {
        for i in 0..1000 {
            // Simulate heavy work
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            
            // Send progress update
            if i % 100 == 0 {
                let _ = tx.send(i).await;
            }
        }
    });
    
    // Handle progress updates
    tokio::spawn(async move {
        while let Some(progress) = rx.recv().await {
            let _ = window_clone.emit("computation-progress", progress);
        }
    });
    
    Ok("Computation started".to_string())
}
```

#### Memory Management
```rust
use std::sync::Weak;
use std::collections::HashMap;

// Use weak references to prevent memory leaks
struct ResourceManager {
    resources: HashMap<String, Weak<Resource>>,
}

impl ResourceManager {
    fn cleanup_expired(&mut self) {
        self.resources.retain(|_, weak_ref| weak_ref.strong_count() > 0);
    }
}

// Implement proper Drop for cleanup
struct MyResource {
    data: Vec<u8>,
}

impl Drop for MyResource {
    fn drop(&mut self) {
        println!("Cleaning up resource with {} bytes", self.data.len());
    }
}
```

## Advanced Projects

### Project 1: Chat Application
**Features**: Real-time messaging with WebSockets

#### Core Features
- Real-time messaging
- User authentication
- Chat rooms/channels
- File sharing
- Message history
- Offline message sync
- Push notifications

#### Technical Architecture
```rust
// WebSocket connection management
use tokio_tungstenite::{connect_async, tungstenite::Message};

#[tauri::command]
async fn connect_to_chat(
    url: String,
    window: tauri::Window,
) -> Result<(), String> {
    let (ws_stream, _) = connect_async(&url)
        .await
        .map_err(|e| e.to_string())?;
    
    let (write, read) = ws_stream.split();
    
    // Handle incoming messages
    tokio::spawn(async move {
        read.for_each(|message| async {
            if let Ok(Message::Text(text)) = message {
                let _ = window.emit("chat-message", text);
            }
        }).await;
    });
    
    Ok(())
}
```

### Project 2: Code Editor
**Features**: Syntax highlighting and file management

#### Core Features
- Syntax highlighting for multiple languages
- File tree navigation
- Search and replace
- Plugin system
- Git integration
- Terminal integration
- Theme customization

#### Implementation Highlights
```javascript
// Monaco Editor integration
import * as monaco from 'monaco-editor';

const editor = monaco.editor.create(document.getElementById('editor'), {
  value: code,
  language: 'typescript',
  theme: 'vs-dark',
  automaticLayout: true,
  minimap: { enabled: true }
});

// File watcher integration
import { watch } from '@tauri-apps/api/fs';

const stopWatching = await watch('/project/path', (event) => {
  if (event.type === 'write') {
    // Reload file content
    reloadFile(event.path);
  }
});
```

### Project 3: Desktop Widget
**Features**: Always-on-top utility with system integration

#### Core Features
- Weather information
- System statistics
- Quick notes
- Calendar integration
- Customizable layout
- Transparency effects
- Auto-positioning

#### System Integration
```rust
// Always on top window with transparency
#[tauri::command]
async fn create_widget_window(app: tauri::AppHandle) -> Result<(), String> {
    let widget_window = tauri::WindowBuilder::new(
        &app,
        "widget",
        tauri::WindowUrl::App("widget.html".into())
    )
    .title("Desktop Widget")
    .inner_size(300.0, 200.0)
    .decorations(false)
    .always_on_top(true)
    .transparent(true)
    .skip_taskbar(true)
    .build()
    .map_err(|e| e.to_string())?;
    
    widget_window.set_opacity(0.9)
        .map_err(|e| e.to_string())?;
    
    Ok(())
}
```

## Learning Objectives

By the end of Phase 4, you should be able to:
- [ ] Integrate and manage databases effectively
- [ ] Configure security settings and CSP
- [ ] Create and publish custom plugins
- [ ] Optimize application performance
- [ ] Build production-ready applications
- [ ] Handle code signing and distribution
- [ ] Implement advanced architectural patterns

## Next Steps

After completing Phase 4:
- Move to Phase 5: Expert Level
- Learn cross-platform development
- Master mobile development with Tauri
- Explore advanced deployment strategies

## Resources

### Database Resources
- [SQLite Documentation](https://sqlite.org/docs.html)
- [Sled Database](https://docs.rs/sled/)
- [Database Design Patterns](https://database.guide/)

### Security Resources
- [Content Security Policy](https://developer.mozilla.org/en-US/docs/Web/HTTP/CSP)
- [Code Signing Guide](https://docs.microsoft.com/en-us/windows/win32/seccrypto/cryptography-tools)
- [Security Best Practices](https://owasp.org/www-project-top-ten/)

### Performance Resources
- [Web Performance](https://web.dev/performance/)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [Bundle Analysis Tools](https://bundlephobia.com/)
