# Network Programming and Protocol Implementation

## Overview
Rust excels at network programming due to its performance, safety, and excellent async support. This guide covers low-level networking, protocol implementation, and building distributed systems.

## TCP/UDP Fundamentals

### Basic TCP Server
```rust
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

fn handle_client(mut stream: TcpStream) {
    let mut buffer = [0; 1024];
    
    loop {
        match stream.read(&mut buffer) {
            Ok(0) => break, // Connection closed
            Ok(n) => {
                println!("Received: {}", String::from_utf8_lossy(&buffer[..n]));
                
                // Echo back
                if stream.write_all(&buffer[..n]).is_err() {
                    break;
                }
            }
            Err(_) => break,
        }
    }
}

fn tcp_server() -> std::io::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:8080")?;
    println!("Server listening on 127.0.0.1:8080");
    
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                thread::spawn(|| {
                    handle_client(stream);
                });
            }
            Err(e) => eprintln!("Error: {}", e),
        }
    }
    
    Ok(())
}
```

### Async TCP with Tokio
```rust
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

async fn handle_connection(mut socket: TcpStream) {
    let mut buf = [0; 1024];
    
    loop {
        match socket.read(&mut buf).await {
            Ok(0) => return, // Connection closed
            Ok(n) => {
                if socket.write_all(&buf[0..n]).await.is_err() {
                    return;
                }
            }
            Err(_) => return,
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind("127.0.0.1:8080").await?;
    println!("Server running on 127.0.0.1:8080");
    
    loop {
        let (socket, _) = listener.accept().await?;
        tokio::spawn(async move {
            handle_connection(socket).await;
        });
    }
}
```

### UDP Socket Programming
```rust
use tokio::net::UdpSocket;
use std::io;

#[tokio::main]
async fn main() -> io::Result<()> {
    let sock = UdpSocket::bind("0.0.0.0:8080").await?;
    let mut buf = [0; 1024];
    
    loop {
        let (len, addr) = sock.recv_from(&mut buf).await?;
        println!("Received from {}: {}", addr, String::from_utf8_lossy(&buf[..len]));
        
        // Echo back
        sock.send_to(&buf[..len], addr).await?;
    }
}
```

## HTTP Protocol Implementation

### Simple HTTP Server
```rust
use std::collections::HashMap;
use std::io::prelude::*;
use std::net::{TcpListener, TcpStream};

#[derive(Debug)]
struct HttpRequest {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    body: String,
}

impl HttpRequest {
    fn parse(request: &str) -> Option<Self> {
        let lines: Vec<&str> = request.lines().collect();
        if lines.is_empty() {
            return None;
        }
        
        let first_line: Vec<&str> = lines[0].split_whitespace().collect();
        if first_line.len() < 2 {
            return None;
        }
        
        let method = first_line[0].to_string();
        let path = first_line[1].to_string();
        
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
        
        Some(HttpRequest {
            method,
            path,
            headers,
            body,
        })
    }
}

fn handle_request(mut stream: TcpStream) {
    let mut buffer = [0; 1024];
    stream.read(&mut buffer).unwrap();
    
    let request = String::from_utf8_lossy(&buffer[..]);
    
    if let Some(http_request) = HttpRequest::parse(&request) {
        let response = match http_request.path.as_str() {
            "/" => "HTTP/1.1 200 OK\r\n\r\nHello, World!",
            "/health" => "HTTP/1.1 200 OK\r\n\r\nOK",
            _ => "HTTP/1.1 404 NOT FOUND\r\n\r\nNot Found",
        };
        
        stream.write(response.as_bytes()).unwrap();
    }
    
    stream.flush().unwrap();
}
```

### HTTP Client
```rust
use std::io::{Read, Write};
use std::net::TcpStream;

fn http_get(host: &str, path: &str) -> Result<String, std::io::Error> {
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
```

## Custom Protocol Implementation

### Binary Protocol Example
```rust
use std::io::{self, Read, Write};
use std::mem;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MessageHeader {
    magic: u32,        // Protocol identifier
    version: u16,      // Protocol version
    message_type: u16, // Type of message
    length: u32,       // Length of payload
}

impl MessageHeader {
    const MAGIC: u32 = 0xDEADBEEF;
    const VERSION: u16 = 1;
    
    fn new(message_type: u16, length: u32) -> Self {
        MessageHeader {
            magic: Self::MAGIC,
            version: Self::VERSION,
            message_type,
            length,
        }
    }
    
    fn to_bytes(&self) -> [u8; 12] {
        unsafe { mem::transmute(*self) }
    }
    
    fn from_bytes(bytes: &[u8; 12]) -> Self {
        unsafe { mem::transmute(*bytes) }
    }
    
    fn is_valid(&self) -> bool {
        self.magic == Self::MAGIC && self.version == Self::VERSION
    }
}

struct Message {
    header: MessageHeader,
    payload: Vec<u8>,
}

impl Message {
    fn new(message_type: u16, payload: Vec<u8>) -> Self {
        let header = MessageHeader::new(message_type, payload.len() as u32);
        Message { header, payload }
    }
    
    fn serialize(&self) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&self.header.to_bytes());
        data.extend_from_slice(&self.payload);
        data
    }
    
    fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut header_bytes = [0u8; 12];
        reader.read_exact(&mut header_bytes)?;
        
        let header = MessageHeader::from_bytes(&header_bytes);
        if !header.is_valid() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid header"));
        }
        
        let mut payload = vec![0u8; header.length as usize];
        reader.read_exact(&mut payload)?;
        
        Ok(Message { header, payload })
    }
}
```

## WebSocket Implementation

### WebSocket Server with tungstenite
```rust
use std::net::{TcpListener, TcpStream};
use std::thread::spawn;
use tungstenite::{accept, Message, WebSocket};

fn handle_client(stream: TcpStream) {
    let mut websocket = accept(stream).unwrap();
    
    loop {
        match websocket.read_message() {
            Ok(msg) => {
                match msg {
                    Message::Text(text) => {
                        println!("Received: {}", text);
                        websocket.write_message(Message::Text(format!("Echo: {}", text))).unwrap();
                    }
                    Message::Binary(bin) => {
                        println!("Received binary data: {} bytes", bin.len());
                        websocket.write_message(Message::Binary(bin)).unwrap();
                    }
                    Message::Close(_) => {
                        println!("Client disconnected");
                        break;
                    }
                    _ => {}
                }
            }
            Err(e) => {
                println!("Error: {}", e);
                break;
            }
        }
    }
}

fn main() {
    let server = TcpListener::bind("127.0.0.1:9001").unwrap();
    
    for stream in server.incoming() {
        spawn(move || handle_client(stream.unwrap()));
    }
}
```

### Async WebSocket with tokio-tungstenite
```rust
use futures_util::{SinkExt, StreamExt};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::{accept_async, tungstenite::Message};

async fn handle_connection(stream: TcpStream) {
    let ws_stream = accept_async(stream).await.expect("Error during handshake");
    let (mut ws_sender, mut ws_receiver) = ws_stream.split();
    
    while let Some(msg) = ws_receiver.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                println!("Received: {}", text);
                if ws_sender.send(Message::Text(format!("Echo: {}", text))).await.is_err() {
                    break;
                }
            }
            Ok(Message::Close(_)) => break,
            Err(e) => {
                println!("WebSocket error: {}", e);
                break;
            }
            _ => {}
        }
    }
}

#[tokio::main]
async fn main() {
    let listener = TcpListener::bind("127.0.0.1:9001").await.unwrap();
    
    while let Ok((stream, _)) = listener.accept().await {
        tokio::spawn(handle_connection(stream));
    }
}
```

## P2P Networking

### libp2p Example
```rust
use libp2p::{
    gossipsub, mdns, noise,
    swarm::{Swarm, SwarmBuilder, SwarmEvent},
    tcp, yamux, PeerId, Transport,
    identity,
};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Duration;
use tokio::{io, select};

#[derive(NetworkBehaviour)]
#[behaviour(out_event = "MyBehaviourEvent")]
struct MyBehaviour {
    gossipsub: gossipsub::Gossipsub,
    mdns: mdns::Mdns,
}

#[derive(Debug)]
enum MyBehaviourEvent {
    Gossipsub(gossipsub::GossipsubEvent),
    Mdns(mdns::MdnsEvent),
}

impl From<gossipsub::GossipsubEvent> for MyBehaviourEvent {
    fn from(event: gossipsub::GossipsubEvent) -> Self {
        MyBehaviourEvent::Gossipsub(event)
    }
}

impl From<mdns::MdnsEvent> for MyBehaviourEvent {
    fn from(event: mdns::MdnsEvent) -> Self {
        MyBehaviourEvent::Mdns(event)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let local_key = identity::Keypair::generate_ed25519();
    let local_peer_id = PeerId::from(local_key.public());
    
    let transport = tcp::TcpConfig::new()
        .upgrade(upgrade::Version::V1)
        .authenticate(noise::NoiseConfig::xx(local_key).into_authenticated())
        .multiplex(yamux::YamuxConfig::default())
        .boxed();
    
    let topic = gossipsub::IdentTopic::new("chat");
    let gossipsub_config = gossipsub::GossipsubConfig::builder()
        .heartbeat_interval(Duration::from_secs(10))
        .validation_mode(gossipsub::ValidationMode::Strict)
        .build()?;
    
    let mut gossipsub = gossipsub::Gossipsub::new(
        gossipsub::MessageAuthenticity::Signed(local_key),
        gossipsub_config,
    )?;
    
    gossipsub.subscribe(&topic)?;
    
    let mdns = mdns::Mdns::new(mdns::MdnsConfig::default()).await?;
    let behaviour = MyBehaviour { gossipsub, mdns };
    
    let mut swarm = SwarmBuilder::new(transport, behaviour, local_peer_id)
        .build();
    
    swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;
    
    loop {
        select! {
            event = swarm.select_next_some() => {
                match event {
                    SwarmEvent::NewListenAddr { address, .. } => {
                        println!("Listening on {:?}", address);
                    }
                    SwarmEvent::Behaviour(MyBehaviourEvent::Mdns(mdns::MdnsEvent::Discovered(list))) => {
                        for (peer, _) in list {
                            swarm.behaviour_mut().gossipsub.add_explicit_peer(&peer);
                        }
                    }
                    SwarmEvent::Behaviour(MyBehaviourEvent::Gossipsub(gossipsub::GossipsubEvent::Message {
                        propagation_source: peer_id,
                        message_id: id,
                        message,
                    })) => {
                        println!("Got message: {} from {:?}", String::from_utf8_lossy(&message.data), peer_id);
                    }
                    _ => {}
                }
            }
        }
    }
}
```

## Load Balancing and Proxying

### Simple Load Balancer
```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

struct LoadBalancer {
    backends: Vec<String>,
    current: AtomicUsize,
}

impl LoadBalancer {
    fn new(backends: Vec<String>) -> Self {
        LoadBalancer {
            backends,
            current: AtomicUsize::new(0),
        }
    }
    
    fn next_backend(&self) -> &str {
        let index = self.current.fetch_add(1, Ordering::Relaxed) % self.backends.len();
        &self.backends[index]
    }
}

async fn proxy_connection(mut client: TcpStream, backend_addr: &str) {
    match TcpStream::connect(backend_addr).await {
        Ok(mut backend) => {
            let (mut client_read, mut client_write) = client.split();
            let (mut backend_read, mut backend_write) = backend.split();
            
            let client_to_backend = async {
                tokio::io::copy(&mut client_read, &mut backend_write).await
            };
            
            let backend_to_client = async {
                tokio::io::copy(&mut backend_read, &mut client_write).await
            };
            
            tokio::select! {
                _ = client_to_backend => {},
                _ = backend_to_client => {},
            }
        }
        Err(e) => {
            eprintln!("Failed to connect to backend {}: {}", backend_addr, e);
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backends = vec![
        "127.0.0.1:8081".to_string(),
        "127.0.0.1:8082".to_string(),
        "127.0.0.1:8083".to_string(),
    ];
    
    let load_balancer = Arc::new(LoadBalancer::new(backends));
    let listener = TcpListener::bind("127.0.0.1:8080").await?;
    
    println!("Load balancer listening on 127.0.0.1:8080");
    
    loop {
        let (client, _) = listener.accept().await?;
        let lb = Arc::clone(&load_balancer);
        
        tokio::spawn(async move {
            let backend = lb.next_backend().to_string();
            proxy_connection(client, &backend).await;
        });
    }
}
```

## Essential Crates

```toml
[dependencies]
# Async runtime
tokio = { version = "1.0", features = ["full"] }

# HTTP
hyper = "0.14"
reqwest = "0.11"
axum = "0.6"

# WebSocket
tungstenite = "0.18"
tokio-tungstenite = "0.18"

# P2P
libp2p = "0.45"

# Serialization
serde = { version = "1.0", features = ["derive"] }
bincode = "1.3"
```

## Performance Tips

1. **Use async/await**: For I/O-bound operations
2. **Connection pooling**: Reuse connections when possible
3. **Buffer sizes**: Tune buffer sizes for your use case
4. **Zero-copy**: Use techniques like `bytes::Bytes` for efficiency
5. **Monitoring**: Implement metrics and logging

## Project Ideas

1. **Chat Server**: Multi-client chat with WebSocket
2. **HTTP Proxy**: Caching HTTP proxy server
3. **File Transfer Protocol**: Custom P2P file sharing
4. **Distributed Database**: Simple distributed key-value store
5. **Game Server**: Real-time multiplayer game backend

This covers the fundamentals of network programming in Rust, from basic sockets to advanced P2P systems.
