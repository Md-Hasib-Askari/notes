# Ecosystem Mastery - Popular Crates and Tools

## Overview
Master the essential Rust ecosystem including popular crates for serialization, HTTP, databases, web frameworks, and advanced project organization.

## Serialization with Serde

### Basic Serde Usage
```rust
use serde::{Deserialize, Serialize};
use serde_json;

#[derive(Serialize, Deserialize, Debug)]
struct User {
    id: u32,
    name: String,
    email: String,
    active: bool,
}

#[derive(Serialize, Deserialize, Debug)]
struct Config {
    database_url: String,
    port: u16,
    debug: bool,
    #[serde(default)]
    workers: u32,
}

fn serde_basics() -> Result<(), Box<dyn std::error::Error>> {
    let user = User {
        id: 1,
        name: "Alice".to_string(),
        email: "alice@example.com".to_string(),
        active: true,
    };
    
    // Serialize to JSON
    let json = serde_json::to_string(&user)?;
    println!("JSON: {}", json);
    
    // Deserialize from JSON
    let deserialized: User = serde_json::from_str(&json)?;
    println!("Deserialized: {:?}", deserialized);
    
    // Pretty printing
    let pretty_json = serde_json::to_string_pretty(&user)?;
    println!("Pretty JSON:\n{}", pretty_json);
    
    Ok(())
}
```

### Advanced Serde Features
```rust
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;

// Custom serialization
#[derive(Debug)]
struct CustomStruct {
    data: HashMap<String, i32>,
}

impl Serialize for CustomStruct {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.data.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CustomStruct {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let data = HashMap::deserialize(deserializer)?;
        Ok(CustomStruct { data })
    }
}

// Serde attributes and renaming
#[derive(Serialize, Deserialize)]
struct ApiResponse {
    #[serde(rename = "user_id")]
    id: u32,
    
    #[serde(rename = "full_name")]
    name: String,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    avatar: Option<String>,
    
    #[serde(with = "chrono::serde::ts_seconds")]
    created_at: chrono::DateTime<chrono::Utc>,
    
    #[serde(flatten)]
    metadata: HashMap<String, String>,
}

// Custom deserializer functions
fn string_or_number<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::{self, Visitor};
    use std::fmt;
    
    struct StringOrNumberVisitor;
    
    impl<'de> Visitor<'de> for StringOrNumberVisitor {
        type Value = String;
        
        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a string or number")
        }
        
        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(v.to_string())
        }
        
        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(v.to_string())
        }
        
        fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(v.to_string())
        }
    }
    
    deserializer.deserialize_any(StringOrNumberVisitor)
}

#[derive(Deserialize)]
struct FlexibleData {
    #[serde(deserialize_with = "string_or_number")]
    value: String,
}
```

## HTTP Client with Reqwest

### Basic HTTP Operations
```rust
use reqwest::{Client, Error, Response};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Debug)]
struct Post {
    id: Option<u32>,
    title: String,
    body: String,
    #[serde(rename = "userId")]
    user_id: u32,
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let client = Client::new();
    
    // GET request
    let posts: Vec<Post> = client
        .get("https://jsonplaceholder.typicode.com/posts")
        .send()
        .await?
        .json()
        .await?;
    
    println!("Fetched {} posts", posts.len());
    
    // POST request
    let new_post = Post {
        id: None,
        title: "My Post".to_string(),
        body: "This is my post content".to_string(),
        user_id: 1,
    };
    
    let response = client
        .post("https://jsonplaceholder.typicode.com/posts")
        .json(&new_post)
        .send()
        .await?;
    
    println!("Post response status: {}", response.status());
    
    Ok(())
}
```

### Advanced Reqwest Features
```rust
use reqwest::{header, ClientBuilder, Proxy};
use std::time::Duration;

async fn advanced_client_example() -> Result<(), reqwest::Error> {
    // Custom client configuration
    let client = ClientBuilder::new()
        .timeout(Duration::from_secs(30))
        .user_agent("MyApp/1.0")
        .default_headers({
            let mut headers = header::HeaderMap::new();
            headers.insert(
                header::AUTHORIZATION,
                header::HeaderValue::from_static("Bearer token123"),
            );
            headers.insert(
                header::CONTENT_TYPE,
                header::HeaderValue::from_static("application/json"),
            );
            headers
        })
        .proxy(Proxy::all("http://proxy.example.com:8080")?)
        .build()?;
    
    // File upload
    let form = reqwest::multipart::Form::new()
        .text("field1", "value1")
        .file("file", "/path/to/file.txt").await?;
    
    let response = client
        .post("https://httpbin.org/post")
        .multipart(form)
        .send()
        .await?;
    
    // Stream response
    let mut stream = response.bytes_stream();
    use futures::StreamExt;
    
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        println!("Received {} bytes", chunk.len());
    }
    
    Ok(())
}

// Retry logic and error handling
use tokio::time::{sleep, Duration};

async fn retry_request(url: &str, max_retries: u32) -> Result<String, reqwest::Error> {
    let client = Client::new();
    let mut attempts = 0;
    
    loop {
        match client.get(url).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    return response.text().await;
                } else if attempts < max_retries {
                    attempts += 1;
                    let delay = Duration::from_millis(1000 * attempts as u64);
                    sleep(delay).await;
                    continue;
                } else {
                    return Err(reqwest::Error::from(response.error_for_status().unwrap_err()));
                }
            }
            Err(e) if attempts < max_retries => {
                attempts += 1;
                let delay = Duration::from_millis(1000 * attempts as u64);
                sleep(delay).await;
                continue;
            }
            Err(e) => return Err(e),
        }
    }
}
```

## Database Integration with Diesel

### Diesel Setup and Basic Usage
```rust
// diesel.toml
// [print_schema]
// file = "src/schema.rs"

// Cargo.toml
// [dependencies]
// diesel = { version = "2.0", features = ["postgres", "chrono"] }

// schema.rs (generated by diesel)
table! {
    users (id) {
        id -> Int4,
        name -> Varchar,
        email -> Varchar,
        created_at -> Timestamp,
    }
}

table! {
    posts (id) {
        id -> Int4,
        title -> Varchar,
        body -> Text,
        published -> Bool,
        user_id -> Int4,
        created_at -> Timestamp,
    }
}

// models.rs
use diesel::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Queryable, Selectable, Serialize, Deserialize, Debug)]
#[diesel(table_name = users)]
pub struct User {
    pub id: i32,
    pub name: String,
    pub email: String,
    pub created_at: chrono::NaiveDateTime,
}

#[derive(Insertable, Deserialize)]
#[diesel(table_name = users)]
pub struct NewUser {
    pub name: String,
    pub email: String,
}

#[derive(Queryable, Selectable, Serialize, Associations, Debug)]
#[diesel(belongs_to(User))]
#[diesel(table_name = posts)]
pub struct Post {
    pub id: i32,
    pub title: String,
    pub body: String,
    pub published: bool,
    pub user_id: i32,
    pub created_at: chrono::NaiveDateTime,
}

#[derive(Insertable)]
#[diesel(table_name = posts)]
pub struct NewPost {
    pub title: String,
    pub body: String,
    pub user_id: i32,
}
```

### Advanced Diesel Queries
```rust
use diesel::prelude::*;
use diesel::pg::PgConnection;
use diesel::connection::Connection;

pub fn establish_connection() -> PgConnection {
    let database_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL must be set");
    
    PgConnection::establish(&database_url)
        .unwrap_or_else(|_| panic!("Error connecting to {}", database_url))
}

// CRUD operations
pub fn create_user(conn: &mut PgConnection, name: &str, email: &str) -> User {
    use crate::schema::users;
    
    let new_user = NewUser {
        name: name.to_string(),
        email: email.to_string(),
    };
    
    diesel::insert_into(users::table)
        .values(&new_user)
        .returning(User::as_returning())
        .get_result(conn)
        .expect("Error saving new user")
}

pub fn get_user_by_id(conn: &mut PgConnection, user_id: i32) -> Option<User> {
    use crate::schema::users::dsl::*;
    
    users
        .filter(id.eq(user_id))
        .select(User::as_select())
        .first(conn)
        .optional()
        .expect("Error loading user")
}

pub fn get_users_with_posts(conn: &mut PgConnection) -> Vec<(User, Vec<Post>)> {
    use crate::schema::{users, posts};
    
    let users_data: Vec<User> = users::table
        .select(User::as_select())
        .load(conn)
        .expect("Error loading users");
    
    let posts_data = Post::belonging_to(&users_data)
        .select(Post::as_select())
        .load(conn)
        .expect("Error loading posts");
    
    let posts_grouped = posts_data.grouped_by(&users_data);
    
    users_data.into_iter().zip(posts_grouped).collect()
}

// Complex queries
pub fn search_posts(
    conn: &mut PgConnection,
    query: &str,
    limit: i64,
    offset: i64,
) -> Vec<Post> {
    use crate::schema::posts::dsl::*;
    
    posts
        .filter(title.ilike(format!("%{}%", query))
            .or(body.ilike(format!("%{}%", query))))
        .filter(published.eq(true))
        .order(created_at.desc())
        .limit(limit)
        .offset(offset)
        .select(Post::as_select())
        .load(conn)
        .expect("Error searching posts")
}

// Transactions
pub fn transfer_posts(
    conn: &mut PgConnection,
    from_user: i32,
    to_user: i32,
    post_ids: Vec<i32>,
) -> Result<(), diesel::result::Error> {
    use crate::schema::posts::dsl::*;
    
    conn.transaction(|conn| {
        // Verify all posts belong to from_user
        let count: i64 = posts
            .filter(id.eq_any(&post_ids))
            .filter(user_id.eq(from_user))
            .count()
            .get_result(conn)?;
        
        if count != post_ids.len() as i64 {
            return Err(diesel::result::Error::RollbackTransaction);
        }
        
        // Transfer posts
        diesel::update(posts.filter(id.eq_any(&post_ids)))
            .set(user_id.eq(to_user))
            .execute(conn)?;
        
        Ok(())
    })
}
```

## Web Framework with Actix-Web

### Basic Actix-Web Setup
```rust
use actix_web::{web, App, HttpResponse, HttpServer, Result, middleware::Logger};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct CreateUserRequest {
    name: String,
    email: String,
}

#[derive(Serialize)]
struct UserResponse {
    id: u32,
    name: String,
    email: String,
}

// Handlers
async fn get_users() -> Result<HttpResponse> {
    let users = vec![
        UserResponse {
            id: 1,
            name: "Alice".to_string(),
            email: "alice@example.com".to_string(),
        },
        UserResponse {
            id: 2,
            name: "Bob".to_string(),
            email: "bob@example.com".to_string(),
        },
    ];
    
    Ok(HttpResponse::Ok().json(users))
}

async fn get_user(path: web::Path<u32>) -> Result<HttpResponse> {
    let user_id = path.into_inner();
    
    let user = UserResponse {
        id: user_id,
        name: format!("User {}", user_id),
        email: format!("user{}@example.com", user_id),
    };
    
    Ok(HttpResponse::Ok().json(user))
}

async fn create_user(user_data: web::Json<CreateUserRequest>) -> Result<HttpResponse> {
    let new_user = UserResponse {
        id: 123, // Would be generated by database
        name: user_data.name.clone(),
        email: user_data.email.clone(),
    };
    
    Ok(HttpResponse::Created().json(new_user))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init();
    
    HttpServer::new(|| {
        App::new()
            .wrap(Logger::default())
            .route("/users", web::get().to(get_users))
            .route("/users", web::post().to(create_user))
            .route("/users/{id}", web::get().to(get_user))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
```

### Advanced Actix-Web Features
```rust
use actix_web::{
    web, App, HttpServer, HttpResponse, Result, Error,
    middleware::{Logger, DefaultHeaders},
    guard,
};
use actix_web_httpauth::middleware::HttpAuthentication;
use actix_web_httpauth::extractors::bearer::BearerAuth;

// Application state
#[derive(Clone)]
struct AppState {
    database_pool: r2d2::Pool<r2d2_sqlite::SqliteConnectionManager>,
    jwt_secret: String,
}

// Middleware for authentication
async fn bearer_auth_validator(
    req: actix_web::dev::ServiceRequest,
    auth: BearerAuth,
) -> Result<actix_web::dev::ServiceRequest, Error> {
    // Validate JWT token here
    let token = auth.token();
    
    if token == "valid_token" {
        Ok(req)
    } else {
        Err(actix_web::error::ErrorUnauthorized("Invalid token"))
    }
}

// Custom error handler
async fn not_found() -> Result<HttpResponse> {
    Ok(HttpResponse::NotFound().json(serde_json::json!({
        "error": "Resource not found"
    })))
}

// Streaming response
async fn stream_data() -> Result<HttpResponse> {
    use actix_web::body::BodyStream;
    use futures::stream;
    
    let data_stream = stream::iter((0..1000).map(|i| {
        Ok::<_, actix_web::Error>(
            actix_web::web::Bytes::from(format!("chunk {}\n", i))
        )
    }));
    
    Ok(HttpResponse::Ok()
        .content_type("text/plain")
        .streaming(BodyStream::new(data_stream)))
}

// File upload handler
async fn upload_file(
    mut payload: actix_multipart::Multipart,
) -> Result<HttpResponse, Error> {
    use actix_multipart::Field;
    use futures::{StreamExt, TryStreamExt};
    use tokio::fs::File;
    use tokio::io::AsyncWriteExt;
    
    while let Some(mut field) = payload.try_next().await? {
        let content_disposition = field.content_disposition();
        
        if let Some(filename) = content_disposition.get_filename() {
            let filepath = format!("./uploads/{}", filename);
            let mut file = File::create(filepath).await?;
            
            while let Some(chunk) = field.try_next().await? {
                file.write_all(&chunk).await?;
            }
        }
    }
    
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "message": "File uploaded successfully"
    })))
}

fn create_app() -> App<
    impl actix_web::dev::ServiceFactory<
        actix_web::dev::ServiceRequest,
        Config = (),
        Response = actix_web::dev::ServiceResponse<impl actix_web::body::MessageBody>,
        Error = actix_web::Error,
        InitError = (),
    >,
> {
    App::new()
        .wrap(Logger::default())
        .wrap(DefaultHeaders::new().header("X-Version", "1.0"))
        .service(
            web::scope("/api/v1")
                .wrap(HttpAuthentication::bearer(bearer_auth_validator))
                .route("/users", web::get().to(get_users))
                .route("/users", web::post().to(create_user))
                .route("/users/{id}", web::get().to(get_user))
        )
        .service(
            web::scope("/public")
                .route("/stream", web::get().to(stream_data))
                .route("/upload", web::post().to(upload_file))
        )
        .default_service(web::route().to(not_found))
}
```

## Cargo Workspaces and Project Organization

### Workspace Configuration
```toml
# Cargo.toml (workspace root)
[workspace]
members = [
    "common",
    "api-server",
    "cli-tool",
    "database-migrations",
]
resolver = "2"

[workspace.dependencies]
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.0", features = ["full"] }
anyhow = "1.0"
thiserror = "1.0"

[workspace.package]
version = "0.1.0"
edition = "2021"
authors = ["Your Name <you@example.com>"]
license = "MIT OR Apache-2.0"
```

### Package Structure
```toml
# common/Cargo.toml
[package]
name = "common"
version.workspace = true
edition.workspace = true

[dependencies]
serde.workspace = true
thiserror.workspace = true

# api-server/Cargo.toml
[package]
name = "api-server"
version.workspace = true
edition.workspace = true

[dependencies]
common = { path = "../common" }
tokio.workspace = true
actix-web = "4.0"
diesel = { version = "2.0", features = ["postgres"] }

# cli-tool/Cargo.toml
[package]
name = "cli-tool"
version.workspace = true
edition.workspace = true

[[bin]]
name = "mycli"
path = "src/main.rs"

[dependencies]
common = { path = "../common" }
clap = { version = "4.0", features = ["derive"] }
anyhow.workspace = true
```

### Cross-Package Code Organization
```rust
// common/src/lib.rs
pub mod models;
pub mod errors;
pub mod utils;

pub use models::*;
pub use errors::*;

// common/src/models.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: u32,
    pub name: String,
    pub email: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Post {
    pub id: u32,
    pub title: String,
    pub content: String,
    pub author_id: u32,
}

// common/src/errors.rs
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("User not found: {id}")]
    UserNotFound { id: u32 },
    
    #[error("Database error: {0}")]
    Database(#[from] diesel::result::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),
}

pub type Result<T> = std::result::Result<T, AppError>;
```

## Publishing and Distribution

### Preparing for Publication
```toml
# Cargo.toml for published crate
[package]
name = "my-awesome-crate"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <you@example.com>"]
license = "MIT OR Apache-2.0"
description = "A short description of my crate"
documentation = "https://docs.rs/my-awesome-crate"
homepage = "https://github.com/yourusername/my-awesome-crate"
repository = "https://github.com/yourusername/my-awesome-crate"
readme = "README.md"
keywords = ["cli", "tool", "awesome"]
categories = ["command-line-utilities"]
include = [
    "src/**/*",
    "Cargo.toml",
    "README.md",
    "LICENSE-*",
]
exclude = [
    "examples/large-files/*",
    "tests/fixtures/*",
]

[dependencies]
# Only necessary dependencies

[dev-dependencies]
# Dependencies only for testing

[features]
default = ["feature1"]
feature1 = []
feature2 = ["dep:optional-dependency"]

[optional-dependencies]
optional-dependency = "1.0"
```

### Documentation and Examples
```rust
//! # My Awesome Crate
//! 
//! This crate provides awesome functionality for doing awesome things.
//! 
//! ## Quick Start
//! 
//! ```rust
//! use my_awesome_crate::AwesomeStruct;
//! 
//! let awesome = AwesomeStruct::new();
//! println!("{}", awesome.do_awesome_thing());
//! ```
//! 
//! ## Features
//! 
//! - Feature 1: Does amazing things
//! - Feature 2: Does even more amazing things

/// A struct that does awesome things.
/// 
/// # Examples
/// 
/// ```
/// use my_awesome_crate::AwesomeStruct;
/// 
/// let awesome = AwesomeStruct::new();
/// assert_eq!(awesome.get_value(), 42);
/// ```
pub struct AwesomeStruct {
    value: i32,
}

impl AwesomeStruct {
    /// Creates a new `AwesomeStruct`.
    /// 
    /// # Examples
    /// 
    /// ```
    /// use my_awesome_crate::AwesomeStruct;
    /// 
    /// let awesome = AwesomeStruct::new();
    /// ```
    pub fn new() -> Self {
        Self { value: 42 }
    }
    
    /// Gets the awesome value.
    /// 
    /// # Examples
    /// 
    /// ```
    /// use my_awesome_crate::AwesomeStruct;
    /// 
    /// let awesome = AwesomeStruct::new();
    /// assert_eq!(awesome.get_value(), 42);
    /// ```
    pub fn get_value(&self) -> i32 {
        self.value
    }
    
    /// Does an awesome thing.
    /// 
    /// # Errors
    /// 
    /// Returns an error if the thing is not awesome enough.
    /// 
    /// # Examples
    /// 
    /// ```
    /// use my_awesome_crate::AwesomeStruct;
    /// 
    /// let awesome = AwesomeStruct::new();
    /// let result = awesome.do_awesome_thing();
    /// assert!(result.is_ok());
    /// ```
    pub fn do_awesome_thing(&self) -> Result<String, &'static str> {
        if self.value > 0 {
            Ok(format!("Awesome value: {}", self.value))
        } else {
            Err("Not awesome enough")
        }
    }
}

impl Default for AwesomeStruct {
    fn default() -> Self {
        Self::new()
    }
}
```

## Key Takeaways

1. **Serde**: Master serialization for data interchange
2. **Reqwest**: Build robust HTTP clients with retries and error handling
3. **Diesel**: Use type-safe database queries and migrations
4. **Actix-Web**: Create scalable web services with middleware
5. **Workspaces**: Organize large projects with multiple crates
6. **Documentation**: Write comprehensive docs with examples
7. **Publishing**: Follow best practices for crate distribution
8. **Ecosystem**: Leverage the rich Rust crate ecosystem effectively
