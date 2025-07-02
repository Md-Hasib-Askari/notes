# Specialized Domains - Web Development

## Overview
Deep dive into web development using modern Rust frameworks like Axum and Actix-Web, covering everything from basic APIs to full-stack applications.

## Axum Web Framework

### Basic Axum Setup
```rust
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{Html, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::net::TcpListener;

#[derive(Serialize, Deserialize)]
struct User {
    id: u64,
    name: String,
    email: String,
}

#[derive(Deserialize)]
struct CreateUser {
    name: String,
    email: String,
}

#[derive(Deserialize)]
struct Pagination {
    page: Option<u64>,
    limit: Option<u64>,
}

// Application state
#[derive(Clone)]
struct AppState {
    users: std::sync::Arc<tokio::sync::Mutex<HashMap<u64, User>>>,
}

// Handlers
async fn root() -> Html<&'static str> {
    Html("<h1>Welcome to Rust Web API</h1>")
}

async fn get_users(
    Query(pagination): Query<Pagination>,
    State(state): State<AppState>,
) -> Json<Vec<User>> {
    let users = state.users.lock().await;
    let page = pagination.page.unwrap_or(1);
    let limit = pagination.limit.unwrap_or(10);
    
    let users: Vec<User> = users
        .values()
        .skip(((page - 1) * limit) as usize)
        .take(limit as usize)
        .cloned()
        .collect();
    
    Json(users)
}

async fn get_user(
    Path(id): Path<u64>,
    State(state): State<AppState>,
) -> Result<Json<User>, StatusCode> {
    let users = state.users.lock().await;
    
    match users.get(&id) {
        Some(user) => Ok(Json(user.clone())),
        None => Err(StatusCode::NOT_FOUND),
    }
}

async fn create_user(
    State(state): State<AppState>,
    Json(payload): Json<CreateUser>,
) -> Result<Json<User>, StatusCode> {
    let mut users = state.users.lock().await;
    let id = users.len() as u64 + 1;
    
    let user = User {
        id,
        name: payload.name,
        email: payload.email,
    };
    
    users.insert(id, user.clone());
    Ok(Json(user))
}

#[tokio::main]
async fn main() {
    let state = AppState {
        users: std::sync::Arc::new(tokio::sync::Mutex::new(HashMap::new())),
    };
    
    let app = Router::new()
        .route("/", get(root))
        .route("/users", get(get_users).post(create_user))
        .route("/users/:id", get(get_user))
        .with_state(state);
    
    let listener = TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("Server running on http://localhost:3000");
    
    axum::serve(listener, app).await.unwrap();
}
```

### Advanced Axum Features
```rust
use axum::{
    extract::{Request, FromRequest},
    http::{header, HeaderMap, HeaderValue},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    Json, RequestExt,
};
use axum_extra::extract::cookie::{Cookie, CookieJar};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// JWT Claims
#[derive(Debug, Serialize, Deserialize)]
struct Claims {
    sub: String,
    exp: usize,
    iat: usize,
}

// Custom extractor for authenticated user
struct AuthenticatedUser {
    user_id: String,
}

#[axum::async_trait]
impl<S> FromRequest<S> for AuthenticatedUser
where
    S: Send + Sync,
{
    type Rejection = (StatusCode, Json<serde_json::Value>);
    
    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let headers = req.headers();
        
        let auth_header = headers
            .get(header::AUTHORIZATION)
            .and_then(|header| header.to_str().ok())
            .ok_or_else(|| {
                (
                    StatusCode::UNAUTHORIZED,
                    Json(serde_json::json!({
                        "error": "Missing authorization header"
                    })),
                )
            })?;
        
        let token = auth_header
            .strip_prefix("Bearer ")
            .ok_or_else(|| {
                (
                    StatusCode::UNAUTHORIZED,
                    Json(serde_json::json!({
                        "error": "Invalid authorization header format"
                    })),
                )
            })?;
        
        // Decode JWT token
        let secret = "your-secret-key";
        let token_data = decode::<Claims>(
            token,
            &DecodingKey::from_secret(secret.as_ref()),
            &Validation::default(),
        )
        .map_err(|_| {
            (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({
                    "error": "Invalid token"
                })),
            )
        })?;
        
        Ok(AuthenticatedUser {
            user_id: token_data.claims.sub,
        })
    }
}

// Middleware for request logging
async fn logging_middleware(req: Request, next: Next) -> Response {
    let method = req.method().clone();
    let uri = req.uri().clone();
    let start_time = std::time::Instant::now();
    
    let response = next.run(req).await;
    
    let duration = start_time.elapsed();
    let status = response.status();
    
    println!(
        "{} {} {} - {:?}",
        method, uri, status, duration
    );
    
    response
}

// CORS middleware
async fn cors_middleware(req: Request, next: Next) -> Response {
    let mut response = next.run(req).await;
    
    let headers = response.headers_mut();
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_ORIGIN,
        HeaderValue::from_static("*"),
    );
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_METHODS,
        HeaderValue::from_static("GET, POST, PUT, DELETE, OPTIONS"),
    );
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_HEADERS,
        HeaderValue::from_static("Content-Type, Authorization"),
    );
    
    response
}

// Protected route handler
async fn protected_route(user: AuthenticatedUser) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "message": "Access granted",
        "user_id": user.user_id
    }))
}

// Login handler
#[derive(Deserialize)]
struct LoginRequest {
    username: String,
    password: String,
}

async fn login(Json(credentials): Json<LoginRequest>) -> Result<Json<serde_json::Value>, StatusCode> {
    // Validate credentials (in real app, check against database)
    if credentials.username == "admin" && credentials.password == "password" {
        let claims = Claims {
            sub: credentials.username,
            exp: (chrono::Utc::now() + chrono::Duration::hours(24)).timestamp() as usize,
            iat: chrono::Utc::now().timestamp() as usize,
        };
        
        let secret = "your-secret-key";
        let token = encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(secret.as_ref()),
        )
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        
        Ok(Json(serde_json::json!({
            "token": token,
            "expires_in": 86400
        })))
    } else {
        Err(StatusCode::UNAUTHORIZED)
    }
}

fn create_advanced_app() -> Router {
    Router::new()
        .route("/login", post(login))
        .route("/protected", get(protected_route))
        .layer(middleware::from_fn(cors_middleware))
        .layer(middleware::from_fn(logging_middleware))
}
```

### Real-time Features with WebSockets
```rust
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::Response,
};
use futures::{sink::SinkExt, stream::StreamExt};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};

// Chat application state
type Rooms = Arc<RwLock<HashMap<String, broadcast::Sender<String>>>>;

#[derive(Clone)]
struct ChatState {
    rooms: Rooms,
}

// WebSocket handler
async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<ChatState>,
) -> Response {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: ChatState) {
    let (mut sender, mut receiver) = socket.split();
    
    // Handle incoming messages
    let mut room_name = String::new();
    let mut rx = None;
    
    while let Some(msg) = receiver.next().await {
        if let Ok(msg) = msg {
            match msg {
                Message::Text(text) => {
                    let parts: Vec<&str> = text.splitn(2, ':').collect();
                    match parts[0] {
                        "join" => {
                            if parts.len() == 2 {
                                room_name = parts[1].to_string();
                                
                                let mut rooms = state.rooms.write().await;
                                let tx = rooms
                                    .entry(room_name.clone())
                                    .or_insert_with(|| {
                                        let (tx, _) = broadcast::channel(100);
                                        tx
                                    })
                                    .clone();
                                
                                rx = Some(tx.subscribe());
                                
                                let _ = sender
                                    .send(Message::Text(format!("Joined room: {}", room_name)))
                                    .await;
                            }
                        }
                        "message" => {
                            if parts.len() == 2 && !room_name.is_empty() {
                                let rooms = state.rooms.read().await;
                                if let Some(tx) = rooms.get(&room_name) {
                                    let _ = tx.send(parts[1].to_string());
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Message::Close(_) => break,
                _ => {}
            }
        }
    }
}

// Chat room broadcast handler
async fn room_broadcast(
    mut rx: broadcast::Receiver<String>,
    mut sender: futures::stream::SplitSink<WebSocket, Message>,
) {
    while let Ok(msg) = rx.recv().await {
        if sender.send(Message::Text(msg)).await.is_err() {
            break;
        }
    }
}
```

## Server-Side Rendering with Templates

### Using Tera Templates
```rust
// Cargo.toml
// tera = "1.19"
// axum = { version = "0.7", features = ["macros"] }

use axum::{
    extract::{Path, Query, State},
    response::Html,
    routing::get,
    Router,
};
use serde::{Deserialize, Serialize};
use tera::{Context, Tera};
use std::sync::Arc;

#[derive(Clone)]
struct TemplateState {
    tera: Arc<Tera>,
}

#[derive(Serialize)]
struct Post {
    id: u64,
    title: String,
    content: String,
    author: String,
    created_at: String,
}

#[derive(Deserialize)]
struct PostQuery {
    category: Option<String>,
    tag: Option<String>,
}

async fn index(State(state): State<TemplateState>) -> Html<String> {
    let mut context = Context::new();
    context.insert("title", "Welcome to My Blog");
    context.insert("posts", &get_recent_posts());
    
    let rendered = state.tera.render("index.html", &context).unwrap();
    Html(rendered)
}

async fn post_detail(
    Path(id): Path<u64>,
    State(state): State<TemplateState>,
) -> Html<String> {
    let mut context = Context::new();
    
    if let Some(post) = get_post_by_id(id) {
        context.insert("post", &post);
        context.insert("title", &post.title);
        
        let rendered = state.tera.render("post.html", &context).unwrap();
        Html(rendered)
    } else {
        let rendered = state.tera.render("404.html", &context).unwrap();
        Html(rendered)
    }
}

async fn posts_list(
    Query(query): Query<PostQuery>,
    State(state): State<TemplateState>,
) -> Html<String> {
    let mut context = Context::new();
    context.insert("title", "All Posts");
    
    let posts = filter_posts(query.category, query.tag);
    context.insert("posts", &posts);
    
    let rendered = state.tera.render("posts.html", &context).unwrap();
    Html(rendered)
}

fn get_recent_posts() -> Vec<Post> {
    vec![
        Post {
            id: 1,
            title: "Getting Started with Rust".to_string(),
            content: "Rust is an amazing language...".to_string(),
            author: "John Doe".to_string(),
            created_at: "2024-01-15".to_string(),
        },
        // More posts...
    ]
}

fn get_post_by_id(id: u64) -> Option<Post> {
    get_recent_posts().into_iter().find(|p| p.id == id)
}

fn filter_posts(category: Option<String>, tag: Option<String>) -> Vec<Post> {
    // Filter logic here
    get_recent_posts()
}

#[tokio::main]
async fn main() {
    let tera = Tera::new("templates/**/*").unwrap();
    let state = TemplateState {
        tera: Arc::new(tera),
    };
    
    let app = Router::new()
        .route("/", get(index))
        .route("/posts", get(posts_list))
        .route("/posts/:id", get(post_detail))
        .with_state(state);
    
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

## Database Integration Patterns

### Repository Pattern with SQLx
```rust
use sqlx::{PgPool, Row};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
struct User {
    id: i64,
    email: String,
    name: String,
    created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Deserialize)]
struct CreateUserRequest {
    email: String,
    name: String,
}

// Repository trait
#[async_trait]
trait UserRepository {
    async fn create(&self, user: CreateUserRequest) -> Result<User, sqlx::Error>;
    async fn find_by_id(&self, id: i64) -> Result<Option<User>, sqlx::Error>;
    async fn find_by_email(&self, email: &str) -> Result<Option<User>, sqlx::Error>;
    async fn list(&self, limit: i64, offset: i64) -> Result<Vec<User>, sqlx::Error>;
    async fn update(&self, id: i64, user: CreateUserRequest) -> Result<Option<User>, sqlx::Error>;
    async fn delete(&self, id: i64) -> Result<bool, sqlx::Error>;
}

// PostgreSQL implementation
struct PostgresUserRepository {
    pool: PgPool,
}

impl PostgresUserRepository {
    fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl UserRepository for PostgresUserRepository {
    async fn create(&self, user: CreateUserRequest) -> Result<User, sqlx::Error> {
        let user = sqlx::query_as::<_, User>(
            "INSERT INTO users (email, name) VALUES ($1, $2) 
             RETURNING id, email, name, created_at"
        )
        .bind(&user.email)
        .bind(&user.name)
        .fetch_one(&self.pool)
        .await?;
        
        Ok(user)
    }
    
    async fn find_by_id(&self, id: i64) -> Result<Option<User>, sqlx::Error> {
        let user = sqlx::query_as::<_, User>(
            "SELECT id, email, name, created_at FROM users WHERE id = $1"
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;
        
        Ok(user)
    }
    
    async fn find_by_email(&self, email: &str) -> Result<Option<User>, sqlx::Error> {
        let user = sqlx::query_as::<_, User>(
            "SELECT id, email, name, created_at FROM users WHERE email = $1"
        )
        .bind(email)
        .fetch_optional(&self.pool)
        .await?;
        
        Ok(user)
    }
    
    async fn list(&self, limit: i64, offset: i64) -> Result<Vec<User>, sqlx::Error> {
        let users = sqlx::query_as::<_, User>(
            "SELECT id, email, name, created_at FROM users 
             ORDER BY created_at DESC LIMIT $1 OFFSET $2"
        )
        .bind(limit)
        .bind(offset)
        .fetch_all(&self.pool)
        .await?;
        
        Ok(users)
    }
    
    async fn update(&self, id: i64, user: CreateUserRequest) -> Result<Option<User>, sqlx::Error> {
        let user = sqlx::query_as::<_, User>(
            "UPDATE users SET email = $2, name = $3 WHERE id = $1 
             RETURNING id, email, name, created_at"
        )
        .bind(id)
        .bind(&user.email)
        .bind(&user.name)
        .fetch_optional(&self.pool)
        .await?;
        
        Ok(user)
    }
    
    async fn delete(&self, id: i64) -> Result<bool, sqlx::Error> {
        let result = sqlx::query("DELETE FROM users WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await?;
        
        Ok(result.rows_affected() > 0)
    }
}

// Service layer
struct UserService {
    repository: Box<dyn UserRepository + Send + Sync>,
}

impl UserService {
    fn new(repository: Box<dyn UserRepository + Send + Sync>) -> Self {
        Self { repository }
    }
    
    async fn create_user(&self, user: CreateUserRequest) -> Result<User, String> {
        // Validation
        if user.email.is_empty() || !user.email.contains('@') {
            return Err("Invalid email".to_string());
        }
        
        if user.name.is_empty() {
            return Err("Name cannot be empty".to_string());
        }
        
        // Check if email already exists
        if let Ok(Some(_)) = self.repository.find_by_email(&user.email).await {
            return Err("Email already exists".to_string());
        }
        
        // Create user
        self.repository.create(user).await
            .map_err(|e| format!("Database error: {}", e))
    }
}
```

## Frontend Integration

### API Client Generation
```rust
// For generating TypeScript types
use serde::{Deserialize, Serialize};
use ts_rs::TS;

#[derive(Serialize, Deserialize, TS)]
#[ts(export)]
struct ApiResponse<T> {
    success: bool,
    data: Option<T>,
    error: Option<String>,
}

#[derive(Serialize, Deserialize, TS)]
#[ts(export)]
struct UserDto {
    id: u64,
    name: String,
    email: String,
    #[ts(type = "string")]
    created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Serialize, Deserialize, TS)]
#[ts(export)]
struct CreateUserDto {
    name: String,
    email: String,
}

// Build script to generate TypeScript types
// build.rs
use std::path::Path;

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let bindings_path = Path::new(&out_dir).join("bindings.ts");
    
    UserDto::export_to(&bindings_path).unwrap();
    CreateUserDto::export_to(&bindings_path).unwrap();
    ApiResponse::<UserDto>::export_to(&bindings_path).unwrap();
}
```

### Static File Serving
```rust
use axum::{
    routing::{get_service, MethodRouter},
    Router,
};
use tower_http::services::{ServeDir, ServeFile};

fn create_app_with_static() -> Router {
    Router::new()
        .route("/api/users", get(get_users))
        .nest_service("/static", get_service(ServeDir::new("static")))
        .fallback_service(get_service(ServeFile::new("static/index.html")))
}
```

## Testing Web Applications

### Integration Testing
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use tower::ServiceExt;
    
    #[tokio::test]
    async fn test_get_users() {
        let app = create_test_app().await;
        
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/users")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        
        assert_eq!(response.status(), StatusCode::OK);
        
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        
        let users: Vec<User> = serde_json::from_slice(&body).unwrap();
        assert!(!users.is_empty());
    }
    
    #[tokio::test]
    async fn test_create_user() {
        let app = create_test_app().await;
        
        let new_user = CreateUser {
            name: "Test User".to_string(),
            email: "test@example.com".to_string(),
        };
        
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/users")
                    .method("POST")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&new_user).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        
        assert_eq!(response.status(), StatusCode::CREATED);
    }
    
    async fn create_test_app() -> Router {
        // Setup test database and return configured app
        todo!()
    }
}
```

## Deployment and Production

### Docker Configuration
```dockerfile
# Dockerfile
FROM rust:1.75 as builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/my-web-app /usr/local/bin/my-web-app
COPY static ./static
COPY templates ./templates

EXPOSE 3000

CMD ["my-web-app"]
```

### Configuration Management
```rust
use serde::Deserialize;

#[derive(Deserialize, Clone)]
pub struct Config {
    pub database_url: String,
    pub redis_url: String,
    pub jwt_secret: String,
    pub port: u16,
    pub log_level: String,
    pub cors_origins: Vec<String>,
}

impl Config {
    pub fn from_env() -> Result<Self, config::ConfigError> {
        let settings = config::Config::builder()
            .add_source(config::Environment::with_prefix("APP"))
            .add_source(config::File::with_name("config/default").required(false))
            .add_source(config::File::with_name("config/production").required(false))
            .build()?;
        
        settings.try_deserialize()
    }
}
```

## Key Takeaways

1. **Modern Frameworks**: Axum provides ergonomic APIs with excellent performance
2. **Type Safety**: Leverage Rust's type system for API design
3. **Async Programming**: Master async/await for scalable web services
4. **Database Integration**: Use repositories and services for clean architecture
5. **Authentication**: Implement JWT and middleware for security
6. **Real-time Features**: WebSockets for interactive applications
7. **Testing**: Comprehensive testing strategies for web applications
8. **Deployment**: Containerization and configuration management
