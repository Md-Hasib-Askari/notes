# Basic Database Interactions

## Project Overview
Learn to interact with databases using popular Rust crates, covering both SQL and NoSQL databases.

## SQLite with Rusqlite

### Basic Setup
```toml
# Cargo.toml
[dependencies]
rusqlite = { version = "0.29", features = ["bundled"] }
serde = { version = "1.0", features = ["derive"] }
```

### Simple Database Operations
```rust
use rusqlite::{Connection, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct User {
    id: Option<i32>,
    name: String,
    email: String,
    age: i32,
}

struct Database {
    conn: Connection,
}

impl Database {
    fn new(db_path: &str) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        
        // Create users table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                age INTEGER NOT NULL
            )",
            [],
        )?;
        
        Ok(Database { conn })
    }
    
    fn create_user(&self, user: &User) -> Result<i64> {
        let mut stmt = self.conn.prepare(
            "INSERT INTO users (name, email, age) VALUES (?1, ?2, ?3)"
        )?;
        
        stmt.execute([&user.name, &user.email, &user.age.to_string()])?;
        Ok(self.conn.last_insert_rowid())
    }
    
    fn get_user(&self, id: i32) -> Result<Option<User>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, email, age FROM users WHERE id = ?1"
        )?;
        
        let user_iter = stmt.query_map([id], |row| {
            Ok(User {
                id: Some(row.get(0)?),
                name: row.get(1)?,
                email: row.get(2)?,
                age: row.get(3)?,
            })
        })?;
        
        for user in user_iter {
            return Ok(Some(user?));
        }
        
        Ok(None)
    }
    
    fn get_all_users(&self) -> Result<Vec<User>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, email, age FROM users"
        )?;
        
        let user_iter = stmt.query_map([], |row| {
            Ok(User {
                id: Some(row.get(0)?),
                name: row.get(1)?,
                email: row.get(2)?,
                age: row.get(3)?,
            })
        })?;
        
        let mut users = Vec::new();
        for user in user_iter {
            users.push(user?);
        }
        
        Ok(users)
    }
    
    fn update_user(&self, id: i32, user: &User) -> Result<usize> {
        let mut stmt = self.conn.prepare(
            "UPDATE users SET name = ?1, email = ?2, age = ?3 WHERE id = ?4"
        )?;
        
        Ok(stmt.execute([&user.name, &user.email, &user.age.to_string(), &id.to_string()])?)
    }
    
    fn delete_user(&self, id: i32) -> Result<usize> {
        let mut stmt = self.conn.prepare("DELETE FROM users WHERE id = ?1")?;
        Ok(stmt.execute([id])?)
    }
    
    fn search_users(&self, query: &str) -> Result<Vec<User>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, email, age FROM users WHERE name LIKE ?1 OR email LIKE ?1"
        )?;
        
        let search_pattern = format!("%{}%", query);
        let user_iter = stmt.query_map([search_pattern], |row| {
            Ok(User {
                id: Some(row.get(0)?),
                name: row.get(1)?,
                email: row.get(2)?,
                age: row.get(3)?,
            })
        })?;
        
        let mut users = Vec::new();
        for user in user_iter {
            users.push(user?);
        }
        
        Ok(users)
    }
}

fn main() -> Result<()> {
    let db = Database::new("users.db")?;
    
    // Create some users
    let user1 = User {
        id: None,
        name: "Alice Smith".to_string(),
        email: "alice@example.com".to_string(),
        age: 30,
    };
    
    let user2 = User {
        id: None,
        name: "Bob Johnson".to_string(),
        email: "bob@example.com".to_string(),
        age: 25,
    };
    
    let id1 = db.create_user(&user1)?;
    let id2 = db.create_user(&user2)?;
    
    println!("Created users with IDs: {}, {}", id1, id2);
    
    // Read users
    if let Some(user) = db.get_user(id1 as i32)? {
        println!("Found user: {:?}", user);
    }
    
    // List all users
    let all_users = db.get_all_users()?;
    println!("All users: {:?}", all_users);
    
    // Search users
    let search_results = db.search_users("Alice")?;
    println!("Search results: {:?}", search_results);
    
    Ok(())
}
```

## PostgreSQL with Tokio-Postgres

### Async Database Operations
```toml
# Cargo.toml
[dependencies]
tokio-postgres = "0.7"
tokio = { version = "1", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
uuid = { version = "1.0", features = ["v4"] }
```

```rust
use tokio_postgres::{Client, Error, NoTls, Row};
use uuid::Uuid;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct Product {
    id: Option<Uuid>,
    name: String,
    price: f64,
    description: Option<String>,
    in_stock: bool,
}

impl From<Row> for Product {
    fn from(row: Row) -> Self {
        Product {
            id: Some(row.get("id")),
            name: row.get("name"),
            price: row.get("price"),
            description: row.get("description"),
            in_stock: row.get("in_stock"),
        }
    }
}

struct ProductRepository {
    client: Client,
}

impl ProductRepository {
    async fn new(connection_string: &str) -> Result<Self, Error> {
        let (client, connection) = tokio_postgres::connect(connection_string, NoTls).await?;
        
        // Spawn the connection in the background
        tokio::spawn(async move {
            if let Err(e) = connection.await {
                eprintln!("Connection error: {}", e);
            }
        });
        
        // Create table if not exists
        client.execute(
            "CREATE TABLE IF NOT EXISTS products (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name VARCHAR NOT NULL,
                price DECIMAL(10,2) NOT NULL,
                description TEXT,
                in_stock BOOLEAN NOT NULL DEFAULT true,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )",
            &[],
        ).await?;
        
        Ok(ProductRepository { client })
    }
    
    async fn create_product(&self, product: &Product) -> Result<Uuid, Error> {
        let id = Uuid::new_v4();
        
        self.client.execute(
            "INSERT INTO products (id, name, price, description, in_stock) 
             VALUES ($1, $2, $3, $4, $5)",
            &[&id, &product.name, &product.price, &product.description, &product.in_stock],
        ).await?;
        
        Ok(id)
    }
    
    async fn get_product(&self, id: Uuid) -> Result<Option<Product>, Error> {
        let rows = self.client.query(
            "SELECT id, name, price, description, in_stock FROM products WHERE id = $1",
            &[&id],
        ).await?;
        
        Ok(rows.into_iter().next().map(Product::from))
    }
    
    async fn get_all_products(&self) -> Result<Vec<Product>, Error> {
        let rows = self.client.query(
            "SELECT id, name, price, description, in_stock FROM products ORDER BY name",
            &[],
        ).await?;
        
        Ok(rows.into_iter().map(Product::from).collect())
    }
    
    async fn update_product(&self, id: Uuid, product: &Product) -> Result<bool, Error> {
        let rows_affected = self.client.execute(
            "UPDATE products SET name = $2, price = $3, description = $4, in_stock = $5 
             WHERE id = $1",
            &[&id, &product.name, &product.price, &product.description, &product.in_stock],
        ).await?;
        
        Ok(rows_affected > 0)
    }
    
    async fn delete_product(&self, id: Uuid) -> Result<bool, Error> {
        let rows_affected = self.client.execute(
            "DELETE FROM products WHERE id = $1",
            &[&id],
        ).await?;
        
        Ok(rows_affected > 0)
    }
    
    async fn search_products(&self, query: &str) -> Result<Vec<Product>, Error> {
        let search_pattern = format!("%{}%", query);
        let rows = self.client.query(
            "SELECT id, name, price, description, in_stock FROM products 
             WHERE name ILIKE $1 OR description ILIKE $1
             ORDER BY name",
            &[&search_pattern],
        ).await?;
        
        Ok(rows.into_iter().map(Product::from).collect())
    }
    
    async fn get_products_by_price_range(&self, min_price: f64, max_price: f64) -> Result<Vec<Product>, Error> {
        let rows = self.client.query(
            "SELECT id, name, price, description, in_stock FROM products 
             WHERE price BETWEEN $1 AND $2 AND in_stock = true
             ORDER BY price",
            &[&min_price, &max_price],
        ).await?;
        
        Ok(rows.into_iter().map(Product::from).collect())
    }
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let repo = ProductRepository::new("postgresql://username:password@localhost/mydb").await?;
    
    // Create a product
    let product = Product {
        id: None,
        name: "Laptop".to_string(),
        price: 999.99,
        description: Some("High-performance laptop".to_string()),
        in_stock: true,
    };
    
    let product_id = repo.create_product(&product).await?;
    println!("Created product with ID: {}", product_id);
    
    // Get the product
    if let Some(retrieved_product) = repo.get_product(product_id).await? {
        println!("Retrieved: {:?}", retrieved_product);
    }
    
    // Search products
    let search_results = repo.search_products("laptop").await?;
    println!("Search results: {:?}", search_results);
    
    Ok(())
}
```

## MongoDB with MongoDB Driver

### NoSQL Database Operations
```toml
# Cargo.toml
[dependencies]
mongodb = "2.7"
tokio = { version = "1", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
bson = { version = "2.7", features = ["chrono-0_4"] }
```

```rust
use mongodb::{Client, Collection, Database};
use serde::{Deserialize, Serialize};
use bson::{doc, oid::ObjectId, DateTime};

#[derive(Debug, Serialize, Deserialize)]
struct BlogPost {
    #[serde(rename = "_id", skip_serializing_if = "Option::is_none")]
    id: Option<ObjectId>,
    title: String,
    content: String,
    author: String,
    tags: Vec<String>,
    published: bool,
    created_at: Option<DateTime>,
    updated_at: Option<DateTime>,
}

struct BlogRepository {
    collection: Collection<BlogPost>,
}

impl BlogRepository {
    async fn new(connection_string: &str, db_name: &str) -> mongodb::error::Result<Self> {
        let client = Client::with_uri_str(connection_string).await?;
        let database = client.database(db_name);
        let collection = database.collection::<BlogPost>("blog_posts");
        
        // Create indexes
        collection.create_index(
            doc! { "title": "text", "content": "text" },
            None,
        ).await?;
        
        Ok(BlogRepository { collection })
    }
    
    async fn create_post(&self, mut post: BlogPost) -> mongodb::error::Result<ObjectId> {
        post.created_at = Some(DateTime::now());
        post.updated_at = Some(DateTime::now());
        
        let result = self.collection.insert_one(post, None).await?;
        Ok(result.inserted_id.as_object_id().unwrap())
    }
    
    async fn get_post(&self, id: ObjectId) -> mongodb::error::Result<Option<BlogPost>> {
        self.collection.find_one(doc! { "_id": id }, None).await
    }
    
    async fn get_all_posts(&self) -> mongodb::error::Result<Vec<BlogPost>> {
        let mut cursor = self.collection.find(None, None).await?;
        let mut posts = Vec::new();
        
        while cursor.advance().await? {
            posts.push(cursor.deserialize_current()?);
        }
        
        Ok(posts)
    }
    
    async fn get_published_posts(&self) -> mongodb::error::Result<Vec<BlogPost>> {
        let filter = doc! { "published": true };
        let mut cursor = self.collection.find(filter, None).await?;
        let mut posts = Vec::new();
        
        while cursor.advance().await? {
            posts.push(cursor.deserialize_current()?);
        }
        
        Ok(posts)
    }
    
    async fn update_post(&self, id: ObjectId, mut post: BlogPost) -> mongodb::error::Result<bool> {
        post.updated_at = Some(DateTime::now());
        
        let filter = doc! { "_id": id };
        let update = doc! {
            "$set": {
                "title": &post.title,
                "content": &post.content,
                "author": &post.author,
                "tags": &post.tags,
                "published": post.published,
                "updated_at": post.updated_at
            }
        };
        
        let result = self.collection.update_one(filter, update, None).await?;
        Ok(result.modified_count > 0)
    }
    
    async fn delete_post(&self, id: ObjectId) -> mongodb::error::Result<bool> {
        let result = self.collection.delete_one(doc! { "_id": id }, None).await?;
        Ok(result.deleted_count > 0)
    }
    
    async fn search_posts(&self, query: &str) -> mongodb::error::Result<Vec<BlogPost>> {
        let filter = doc! { "$text": { "$search": query } };
        let mut cursor = self.collection.find(filter, None).await?;
        let mut posts = Vec::new();
        
        while cursor.advance().await? {
            posts.push(cursor.deserialize_current()?);
        }
        
        Ok(posts)
    }
    
    async fn get_posts_by_tag(&self, tag: &str) -> mongodb::error::Result<Vec<BlogPost>> {
        let filter = doc! { "tags": tag };
        let mut cursor = self.collection.find(filter, None).await?;
        let mut posts = Vec::new();
        
        while cursor.advance().await? {
            posts.push(cursor.deserialize_current()?);
        }
        
        Ok(posts)
    }
    
    async fn get_posts_by_author(&self, author: &str) -> mongodb::error::Result<Vec<BlogPost>> {
        let filter = doc! { "author": author };
        let mut cursor = self.collection.find(filter, None).await?;
        let mut posts = Vec::new();
        
        while cursor.advance().await? {
            posts.push(cursor.deserialize_current()?);
        }
        
        Ok(posts)
    }
}

#[tokio::main]
async fn main() -> mongodb::error::Result<()> {
    let repo = BlogRepository::new("mongodb://localhost:27017", "blog_db").await?;
    
    // Create a blog post
    let post = BlogPost {
        id: None,
        title: "Getting Started with Rust".to_string(),
        content: "Rust is a systems programming language...".to_string(),
        author: "Alice".to_string(),
        tags: vec!["rust".to_string(), "programming".to_string()],
        published: true,
        created_at: None,
        updated_at: None,
    };
    
    let post_id = repo.create_post(post).await?;
    println!("Created post with ID: {}", post_id);
    
    // Get the post
    if let Some(retrieved_post) = repo.get_post(post_id).await? {
        println!("Retrieved: {:?}", retrieved_post);
    }
    
    // Search posts
    let search_results = repo.search_posts("rust").await?;
    println!("Search results: {} posts found", search_results.len());
    
    Ok(())
}
```

## Connection Pooling and Transactions

### Connection Pool with r2d2
```toml
# Cargo.toml
[dependencies]
rusqlite = { version = "0.29", features = ["bundled"] }
r2d2 = "0.8"
r2d2_sqlite = "0.22"
```

```rust
use r2d2_sqlite::SqliteConnectionManager;
use r2d2::{Pool, PooledConnection};
use rusqlite::{Result, Transaction};

type SqlitePool = Pool<SqliteConnectionManager>;
type SqliteConnection = PooledConnection<SqliteConnectionManager>;

struct DatabaseService {
    pool: SqlitePool,
}

impl DatabaseService {
    fn new(db_path: &str) -> Result<Self, r2d2::Error> {
        let manager = SqliteConnectionManager::file(db_path);
        let pool = Pool::new(manager)?;
        
        // Initialize schema
        let conn = pool.get()?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL,
                total DECIMAL(10,2) NOT NULL
            )",
            [],
        ).unwrap();
        
        conn.execute(
            "CREATE TABLE IF NOT EXISTS order_items (
                id INTEGER PRIMARY KEY,
                order_id INTEGER NOT NULL,
                product_name TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price DECIMAL(10,2) NOT NULL,
                FOREIGN KEY(order_id) REFERENCES orders(id)
            )",
            [],
        ).unwrap();
        
        Ok(DatabaseService { pool })
    }
    
    fn create_order_with_items(
        &self,
        customer_id: i32,
        items: Vec<(String, i32, f64)>,
    ) -> Result<i64> {
        let mut conn = self.pool.get().unwrap();
        let tx = conn.transaction()?;
        
        // Insert order
        tx.execute(
            "INSERT INTO orders (customer_id, total) VALUES (?1, ?2)",
            [&customer_id.to_string(), &"0.00"],
        )?;
        let order_id = tx.last_insert_rowid();
        
        let mut total = 0.0;
        
        // Insert order items
        for (product_name, quantity, price) in items {
            tx.execute(
                "INSERT INTO order_items (order_id, product_name, quantity, price)
                 VALUES (?1, ?2, ?3, ?4)",
                [&order_id.to_string(), &product_name, &quantity.to_string(), &price.to_string()],
            )?;
            total += price * quantity as f64;
        }
        
        // Update order total
        tx.execute(
            "UPDATE orders SET total = ?1 WHERE id = ?2",
            [&total.to_string(), &order_id.to_string()],
        )?;
        
        tx.commit()?;
        Ok(order_id)
    }
    
    fn get_order_with_items(&self, order_id: i64) -> Result<Option<(i32, f64, Vec<(String, i32, f64)>)>> {
        let conn = self.pool.get().unwrap();
        
        // Get order
        let mut stmt = conn.prepare("SELECT customer_id, total FROM orders WHERE id = ?1")?;
        let order_iter = stmt.query_map([order_id], |row| {
            Ok((row.get::<_, i32>(0)?, row.get::<_, f64>(1)?))
        })?;
        
        let (customer_id, total) = match order_iter.collect::<Result<Vec<_>, _>>()?.into_iter().next() {
            Some(order) => order,
            None => return Ok(None),
        };
        
        // Get order items
        let mut stmt = conn.prepare(
            "SELECT product_name, quantity, price FROM order_items WHERE order_id = ?1"
        )?;
        let item_iter = stmt.query_map([order_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i32>(1)?,
                row.get::<_, f64>(2)?,
            ))
        })?;
        
        let items: Result<Vec<_>, _> = item_iter.collect();
        Ok(Some((customer_id, total, items?)))
    }
}

fn main() -> Result<()> {
    let db_service = DatabaseService::new("orders.db").unwrap();
    
    let items = vec![
        ("Laptop".to_string(), 1, 999.99),
        ("Mouse".to_string(), 2, 25.50),
    ];
    
    let order_id = db_service.create_order_with_items(123, items)?;
    println!("Created order with ID: {}", order_id);
    
    if let Some((customer_id, total, items)) = db_service.get_order_with_items(order_id)? {
        println!("Order {} for customer {}: ${:.2}", order_id, customer_id, total);
        for (name, qty, price) in items {
            println!("  {} x {} @ ${:.2}", qty, name, price);
        }
    }
    
    Ok(())
}
```

## Key Learning Points

1. **SQL vs NoSQL**: Understanding different database paradigms
2. **Connection Management**: Efficient database connections
3. **Error Handling**: Database-specific error types
4. **Async Programming**: Non-blocking database operations
5. **Transactions**: ACID properties and data consistency
6. **Query Building**: Dynamic and safe query construction
7. **Schema Design**: Proper database structure

## Best Practices
- Use connection pooling for production applications
- Handle database errors gracefully
- Use transactions for data consistency
- Implement proper indexing for performance
- Validate data before database operations
- Use prepared statements to prevent SQL injection
