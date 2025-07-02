# Simple HTTP Client/Server

## Project Overview
Build basic HTTP client and server implementations using Rust's standard library and popular crates.

## HTTP Server Implementation

### Basic Server with std::net
```rust
use std::io::prelude::*;
use std::net::{TcpListener, TcpStream};
use std::fs;
use std::thread;

fn main() {
    let listener = TcpListener::bind("127.0.0.1:7878").unwrap();
    println!("Server running on http://127.0.0.1:7878");

    for stream in listener.incoming() {
        let stream = stream.unwrap();
        thread::spawn(|| {
            handle_connection(stream);
        });
    }
}

fn handle_connection(mut stream: TcpStream) {
    let mut buffer = [0; 1024];
    stream.read(&mut buffer).unwrap();

    let get = b"GET / HTTP/1.1\r\n";
    let sleep = b"GET /sleep HTTP/1.1\r\n";

    let (status_line, filename) = if buffer.starts_with(get) {
        ("HTTP/1.1 200 OK", "hello.html")
    } else if buffer.starts_with(sleep) {
        thread::sleep(std::time::Duration::from_secs(5));
        ("HTTP/1.1 200 OK", "hello.html")
    } else {
        ("HTTP/1.1 404 NOT FOUND", "404.html")
    };

    let contents = fs::read_to_string(filename).unwrap_or_else(|_| {
        "<html><body><h1>File not found</h1></body></html>".to_string()
    });

    let response = format!(
        "{}\r\nContent-Length: {}\r\n\r\n{}",
        status_line,
        contents.len(),
        contents
    );

    stream.write(response.as_bytes()).unwrap();
    stream.flush().unwrap();
}
```

### Enhanced Server with Routing
```rust
use std::collections::HashMap;
use std::io::prelude::*;
use std::net::{TcpListener, TcpStream};
use std::thread;

type Handler = fn(&Request) -> Response;

#[derive(Debug)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub headers: HashMap<String, String>,
    pub body: String,
}

#[derive(Debug)]
pub struct Response {
    pub status: u16,
    pub headers: HashMap<String, String>,
    pub body: String,
}

impl Response {
    pub fn new(status: u16, body: String) -> Self {
        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "text/html".to_string());
        
        Self { status, headers, body }
    }
    
    pub fn json(status: u16, body: String) -> Self {
        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        
        Self { status, headers, body }
    }
    
    pub fn to_http(&self) -> String {
        let status_text = match self.status {
            200 => "OK",
            404 => "NOT FOUND",
            500 => "INTERNAL SERVER ERROR",
            _ => "UNKNOWN",
        };
        
        let mut response = format!("HTTP/1.1 {} {}\r\n", self.status, status_text);
        
        for (key, value) in &self.headers {
            response.push_str(&format!("{}: {}\r\n", key, value));
        }
        
        response.push_str(&format!("Content-Length: {}\r\n", self.body.len()));
        response.push_str("\r\n");
        response.push_str(&self.body);
        
        response
    }
}

pub struct Router {
    routes: HashMap<String, Handler>,
}

impl Router {
    pub fn new() -> Self {
        Self {
            routes: HashMap::new(),
        }
    }
    
    pub fn add_route(&mut self, path: &str, handler: Handler) {
        self.routes.insert(path.to_string(), handler);
    }
    
    pub fn handle(&self, request: &Request) -> Response {
        if let Some(handler) = self.routes.get(&request.path) {
            handler(request)
        } else {
            Response::new(404, "<h1>404 - Page Not Found</h1>".to_string())
        }
    }
}

fn parse_request(buffer: &[u8]) -> Request {
    let request_str = String::from_utf8_lossy(buffer);
    let lines: Vec<&str> = request_str.lines().collect();
    
    if lines.is_empty() {
        return Request {
            method: "GET".to_string(),
            path: "/".to_string(),
            headers: HashMap::new(),
            body: String::new(),
        };
    }
    
    let first_line = lines[0];
    let parts: Vec<&str> = first_line.split_whitespace().collect();
    
    let method = parts.get(0).unwrap_or(&"GET").to_string();
    let path = parts.get(1).unwrap_or(&"/").to_string();
    
    let mut headers = HashMap::new();
    let mut body_start = 0;
    
    for (i, line) in lines.iter().enumerate().skip(1) {
        if line.is_empty() {
            body_start = i + 1;
            break;
        }
        
        if let Some(colon_pos) = line.find(':') {
            let key = line[..colon_pos].trim().to_string();
            let value = line[colon_pos + 1..].trim().to_string();
            headers.insert(key, value);
        }
    }
    
    let body = lines[body_start..].join("\n");
    
    Request { method, path, headers, body }
}

// Route handlers
fn home_handler(_req: &Request) -> Response {
    Response::new(200, "<h1>Welcome to Rust HTTP Server!</h1>".to_string())
}

fn about_handler(_req: &Request) -> Response {
    Response::new(200, "<h1>About Page</h1><p>This is a simple Rust HTTP server.</p>".to_string())
}

fn api_handler(_req: &Request) -> Response {
    Response::json(200, r#"{"message": "Hello from API", "status": "success"}"#.to_string())
}

fn main() {
    let mut router = Router::new();
    router.add_route("/", home_handler);
    router.add_route("/about", about_handler);
    router.add_route("/api", api_handler);
    
    let listener = TcpListener::bind("127.0.0.1:7878").unwrap();
    println!("Server running on http://127.0.0.1:7878");
    
    for stream in listener.incoming() {
        let stream = stream.unwrap();
        
        thread::spawn(move || {
            handle_connection_with_router(stream, &router);
        });
    }
}

fn handle_connection_with_router(mut stream: TcpStream, router: &Router) {
    let mut buffer = [0; 1024];
    stream.read(&mut buffer).unwrap();
    
    let request = parse_request(&buffer);
    let response = router.handle(&request);
    
    stream.write(response.to_http().as_bytes()).unwrap();
    stream.flush().unwrap();
}
```

## HTTP Client Implementation

### Basic Client
```rust
use std::io::{Read, Write};
use std::net::TcpStream;

pub struct HttpClient;

impl HttpClient {
    pub fn get(url: &str) -> Result<String, Box<dyn std::error::Error>> {
        let (host, path) = Self::parse_url(url)?;
        
        let mut stream = TcpStream::connect(format!("{}:80", host))?;
        
        let request = format!(
            "GET {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
            path, host
        );
        
        stream.write_all(request.as_bytes())?;
        
        let mut response = String::new();
        stream.read_to_string(&mut response)?;
        
        Ok(response)
    }
    
    pub fn post(url: &str, body: &str) -> Result<String, Box<dyn std::error::Error>> {
        let (host, path) = Self::parse_url(url)?;
        
        let mut stream = TcpStream::connect(format!("{}:80", host))?;
        
        let request = format!(
            "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            path, host, body.len(), body
        );
        
        stream.write_all(request.as_bytes())?;
        
        let mut response = String::new();
        stream.read_to_string(&mut response)?;
        
        Ok(response)
    }
    
    fn parse_url(url: &str) -> Result<(String, String), Box<dyn std::error::Error>> {
        if !url.starts_with("http://") {
            return Err("Only HTTP URLs supported".into());
        }
        
        let url = &url[7..]; // Remove "http://"
        
        if let Some(slash_pos) = url.find('/') {
            let host = url[..slash_pos].to_string();
            let path = url[slash_pos..].to_string();
            Ok((host, path))
        } else {
            Ok((url.to_string(), "/".to_string()))
        }
    }
}

// Usage example
fn main() -> Result<(), Box<dyn std::error::Error>> {
    match HttpClient::get("http://httpbin.org/get") {
        Ok(response) => println!("Response: {}", response),
        Err(e) => eprintln!("Error: {}", e),
    }
    
    Ok(())
}
```

## Using Popular HTTP Crates

### Server with Warp
```rust
// Cargo.toml
// [dependencies]
// warp = "0.3"
// tokio = { version = "1", features = ["full"] }
// serde = { version = "1.0", features = ["derive"] }

use warp::Filter;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
struct User {
    id: u32,
    name: String,
}

#[tokio::main]
async fn main() {
    // GET /hello/warp => 200 OK with body "Hello, warp!"
    let hello = warp::path!("hello" / String)
        .map(|name| format!("Hello, {}!", name));

    // GET /users => JSON response
    let users = warp::path("users")
        .map(|| {
            let users = vec![
                User { id: 1, name: "Alice".to_string() },
                User { id: 2, name: "Bob".to_string() },
            ];
            warp::reply::json(&users)
        });

    // POST /users with JSON body
    let create_user = warp::path("users")
        .and(warp::post())
        .and(warp::body::json())
        .map(|user: User| {
            println!("Created user: {:?}", user);
            warp::reply::with_status("Created", warp::http::StatusCode::CREATED)
        });

    let routes = hello.or(users).or(create_user);

    warp::serve(routes)
        .run(([127, 0, 0, 1], 3030))
        .await;
}
```

### Client with Reqwest
```rust
// Cargo.toml
// [dependencies]
// reqwest = { version = "0.11", features = ["json"] }
// tokio = { version = "1", features = ["full"] }
// serde = { version = "1.0", features = ["derive"] }

use reqwest;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug)]
struct Post {
    id: u32,
    title: String,
    body: String,
    #[serde(rename = "userId")]
    user_id: u32,
}

#[tokio::main]
async fn main() -> Result<(), reqwest::Error> {
    let client = reqwest::Client::new();
    
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
        id: 0,
        title: "My New Post".to_string(),
        body: "This is the body of my new post".to_string(),
        user_id: 1,
    };
    
    let response = client
        .post("https://jsonplaceholder.typicode.com/posts")
        .json(&new_post)
        .send()
        .await?;
    
    println!("Response status: {}", response.status());
    
    Ok(())
}
```

### Middleware and Error Handling
```rust
use warp::{Filter, Rejection, Reply};
use serde_json;

// Custom error types
#[derive(Debug)]
struct CustomError {
    message: String,
}

impl warp::reject::Reject for CustomError {}

// Logging middleware
fn with_logging() -> impl Filter<Extract = (), Error = Rejection> + Copy {
    warp::log::custom(|info| {
        println!(
            "{} {} {} {}ms",
            info.method(),
            info.path(),
            info.status(),
            info.elapsed().as_millis()
        );
    })
}

// Authentication middleware
fn with_auth() -> impl Filter<Extract = (String,), Error = Rejection> + Clone {
    warp::header::<String>("authorization")
        .and_then(|auth_header: String| async move {
            if auth_header.starts_with("Bearer ") {
                Ok(auth_header[7..].to_string())
            } else {
                Err(warp::reject::custom(CustomError {
                    message: "Invalid authorization header".to_string(),
                }))
            }
        })
}

async fn handle_rejection(err: Rejection) -> Result<impl Reply, std::convert::Infallible> {
    if let Some(custom_error) = err.find::<CustomError>() {
        Ok(warp::reply::with_status(
            format!("Error: {}", custom_error.message),
            warp::http::StatusCode::UNAUTHORIZED,
        ))
    } else {
        Ok(warp::reply::with_status(
            "Internal Server Error".to_string(),
            warp::http::StatusCode::INTERNAL_SERVER_ERROR,
        ))
    }
}
```

## Key Learning Points

1. **HTTP Protocol Basics**: Understanding requests, responses, headers
2. **TCP Networking**: Low-level socket programming
3. **Concurrency**: Handling multiple connections
4. **Error Handling**: Robust error management in network code
5. **Async Programming**: Using async/await for efficient I/O
6. **Serialization**: JSON handling with serde
7. **Middleware**: Request/response processing pipelines

## Extensions
- Add HTTPS support with TLS
- Implement WebSocket connections
- Create REST API with database integration
- Add authentication and authorization
- Implement rate limiting and caching
