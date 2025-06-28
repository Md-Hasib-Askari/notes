### 2. Create Your First Tauri App

#### Method 1: Using create-tauri-app
```bash
npm create tauri-app@latest
# or
yarn create tauri-app
# or
pnpm create tauri-app
```

#### Method 2: Using cargo tauri init
```bash
# In an existing frontend project
cargo tauri init
```

#### Understanding Project Structure
```
my-tauri-app/
├── src-tauri/          # Rust backend
│   ├── src/
│   │   └── main.rs     # Entry point
│   ├── Cargo.toml      # Rust dependencies
│   └── tauri.conf.json # Tauri configuration
├── src/                # Frontend source
├── dist/               # Built frontend
└── package.json        # Frontend dependencies
```

#### Development Workflow
- **Run development server**: `cargo tauri dev`
- **Build for production**: `cargo tauri build`
- **Hot reload**: Automatic on frontend changes
