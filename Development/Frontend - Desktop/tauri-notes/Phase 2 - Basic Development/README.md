# Phase 2: Basic Development (Weeks 3-4)

## Overview
Build on your foundation by integrating frontend frameworks with Tauri and learning the basic APIs. You'll create your first functional applications and understand how to structure Tauri projects.

## Frontend Integration

### Choose and Setup a Frontend Framework

#### React + Vite
```bash
npm create tauri-app@latest my-app -- --template react
```
- **Benefits**: Large ecosystem, extensive documentation
- **Key concepts**: Components, hooks, state management
- **Tauri integration**: Easy with @tauri-apps/api

#### Vue + Vite
```bash
npm create tauri-app@latest my-app -- --template vue
```
- **Benefits**: Gentle learning curve, excellent tooling
- **Key concepts**: Single File Components, Composition API
- **Tauri integration**: Seamless with TypeScript support

#### Svelte/SvelteKit
```bash
npm create tauri-app@latest my-app -- --template svelte
```
- **Benefits**: Minimal boilerplate, compile-time optimizations
- **Key concepts**: Reactive declarations, stores
- **Tauri integration**: Lightweight and fast

#### Vanilla JS/TypeScript
```bash
npm create tauri-app@latest my-app -- --template vanilla
```
- **Benefits**: No framework overhead, full control
- **Key concepts**: Direct DOM manipulation, module systems
- **Tauri integration**: Direct API usage

### Project Structure Best Practices
```
src/
├── components/     # Reusable UI components
├── pages/         # Application pages/views
├── stores/        # State management
├── utils/         # Helper functions
├── types/         # TypeScript type definitions
└── assets/        # Static assets
```

## Tauri API Basics

### @tauri-apps/api Package
```bash
npm install @tauri-apps/api
```

#### Core Modules
- `tauri`: Invoke commands and handle events
- `window`: Window management
- `fs`: File system operations
- `dialog`: Native dialogs
- `notification`: System notifications

### Window Management

#### Basic Window Operations
```javascript
import { appWindow } from '@tauri-apps/api/window';

// Minimize window
await appWindow.minimize();

// Maximize window
await appWindow.maximize();

// Close window
await appWindow.close();

// Set window title
await appWindow.setTitle('My App');
```

#### Window Events
```javascript
import { appWindow } from '@tauri-apps/api/window';

// Listen for window events
const unlisten = await appWindow.listen('tauri://close-requested', () => {
  console.log('Window is about to close');
});

// Window focus events
await appWindow.onFocusChanged(({ payload: focused }) => {
  console.log('Window focused:', focused);
});
```

### File System Operations

#### Reading Files
```javascript
import { readTextFile, readBinaryFile } from '@tauri-apps/api/fs';

// Read text file
const content = await readTextFile('path/to/file.txt');

// Read binary file
const buffer = await readBinaryFile('path/to/image.png');
```

#### Writing Files
```javascript
import { writeTextFile, writeBinaryFile } from '@tauri-apps/api/fs';

// Write text file
await writeTextFile('path/to/output.txt', 'Hello, World!');

// Write binary file
await writeBinaryFile('path/to/output.bin', new Uint8Array([1, 2, 3]));
```

#### Directory Operations
```javascript
import { readDir, createDir, removeDir } from '@tauri-apps/api/fs';

// Read directory contents
const entries = await readDir('path/to/directory');

// Create directory
await createDir('path/to/new-directory', { recursive: true });

// Remove directory
await removeDir('path/to/directory', { recursive: true });
```

### Notifications
```javascript
import { sendNotification } from '@tauri-apps/api/notification';

// Send system notification
sendNotification({
  title: 'Tauri App',
  body: 'Hello from Tauri!',
  icon: 'path/to/icon.png'
});
```

### Dialog Boxes

#### File Dialogs
```javascript
import { open, save } from '@tauri-apps/api/dialog';

// Open file dialog
const selected = await open({
  multiple: false,
  filters: [{
    name: 'Text Files',
    extensions: ['txt', 'md']
  }]
});

// Save file dialog
const savePath = await save({
  defaultPath: 'document.txt',
  filters: [{
    name: 'Text Files',
    extensions: ['txt']
  }]
});
```

#### Message Dialogs
```javascript
import { ask, confirm, message } from '@tauri-apps/api/dialog';

// Ask dialog
const yes = await ask('Are you sure?', 'Confirmation');

// Confirm dialog
const confirmed = await confirm('Delete file?', { title: 'Warning', type: 'warning' });

// Message dialog
await message('Operation completed!', { title: 'Success', type: 'info' });
```

## Simple Projects

### Project 1: Todo App
**Features**: Local storage with file operations

#### Core Functionality
- Add, edit, delete todos
- Mark todos as complete
- Save/load from local file
- Basic filtering (all, active, completed)

#### Key Learning Points
- File system API usage
- State management in frontend
- Data persistence
- Error handling

#### Implementation Steps
1. Create todo data structure
2. Implement CRUD operations
3. Add file save/load functionality
4. Create responsive UI
5. Add keyboard shortcuts

### Project 2: System Info App
**Features**: Display OS and hardware information

#### Core Functionality
- Show OS details (name, version, architecture)
- Display hardware info (CPU, memory, disk)
- Real-time system statistics
- Export information to file

#### Key Learning Points
- System API integration
- Real-time data updates
- Information display techniques
- Export functionality

#### Implementation Steps
1. Create Rust commands for system info
2. Design information display UI
3. Implement real-time updates
4. Add export functionality
5. Style with responsive design

### Project 3: File Manager
**Features**: Basic file browsing and operations

#### Core Functionality
- Browse directories
- View file details (size, modified date)
- Basic file operations (copy, move, delete)
- File preview for text files
- Navigation history

#### Key Learning Points
- Directory traversal
- File operations
- Navigation patterns
- File type handling

#### Implementation Steps
1. Implement directory reading
2. Create file browser UI
3. Add file operation commands
4. Implement navigation
5. Add file preview capability

## Development Best Practices

### Error Handling
```javascript
// Frontend error handling
try {
  const result = await invoke('risky_operation');
  // Handle success
} catch (error) {
  console.error('Operation failed:', error);
  // Show user-friendly error message
}
```

```rust
// Backend error handling
#[tauri::command]
fn risky_operation() -> Result<String, String> {
    // Some operation that might fail
    if some_condition {
        Ok("Success".to_string())
    } else {
        Err("Operation failed".to_string())
    }
}
```

### Configuration Management
```json
// tauri.conf.json allowlist configuration
{
  "tauri": {
    "allowlist": {
      "fs": {
        "readFile": true,
        "writeFile": true,
        "readDir": true,
        "createDir": true
      },
      "dialog": {
        "open": true,
        "save": true,
        "message": true,
        "ask": true,
        "confirm": true
      },
      "notification": {
        "all": true
      }
    }
  }
}
```

### State Management Patterns
- Use appropriate state management for your framework
- Keep state close to where it's used
- Consider using stores for global state
- Implement proper state synchronization

## Learning Objectives

By the end of Phase 2, you should be able to:
- [ ] Set up and configure a frontend framework with Tauri
- [ ] Use core Tauri APIs effectively
- [ ] Handle file system operations
- [ ] Create and manage dialogs and notifications
- [ ] Build complete, functional applications
- [ ] Implement proper error handling
- [ ] Structure Tauri projects effectively

## Practice Exercises

### Exercise 1: API Explorer
Create a simple app that demonstrates each Tauri API:
- Window management buttons
- File operations
- Dialog examples
- Notification testing

### Exercise 2: Settings Manager
Build an app that:
- Saves user preferences to a file
- Loads settings on startup
- Provides a settings UI
- Validates user input

### Exercise 3: Log Viewer
Create an application that:
- Reads log files from a directory
- Displays logs in a readable format
- Filters logs by level/date
- Exports filtered results

## Common Issues and Solutions

### API Permission Errors
- Check allowlist configuration in tauri.conf.json
- Ensure proper API permissions are enabled
- Validate file paths and permissions

### State Synchronization
- Use proper async/await patterns
- Handle loading states in UI
- Implement optimistic updates where appropriate

### File Path Issues
- Use path API for cross-platform compatibility
- Handle relative vs absolute paths correctly
- Validate file existence before operations

## Next Steps

After completing Phase 2:
- Move to Phase 3: Intermediate Development
- Explore advanced Tauri APIs
- Learn backend development with Rust
- Build more complex applications

## Resources

### Documentation
- [Tauri API Reference](https://tauri.app/v1/api/js/)
- [Frontend Framework Docs](https://reactjs.org/docs/) (or Vue, Svelte)
- [File System API Guide](https://tauri.app/v1/api/js/fs)

### Example Projects
- [Tauri Examples Repository](https://github.com/tauri-apps/tauri/tree/dev/examples)
- Community showcase applications
- Starter templates and boilerplates

### Tools
- VS Code with Tauri extension
- Browser DevTools for frontend debugging
- Rust analyzer for backend development
