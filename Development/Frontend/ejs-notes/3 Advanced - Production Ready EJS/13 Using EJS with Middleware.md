# 13. Using EJS with Middleware

## Flash Messages
Flash messages are temporary messages stored in the session and displayed to the user after a redirect.

### Example:
- Install `express-flash` and `express-session`:
  ```bash
  npm install express-flash express-session
  ```

- Set up middleware:
  ```javascript
  const session = require('express-session');
  const flash = require('express-flash');

  app.use(session({
    secret: 'your-secret-key',
    resave: false,
    saveUninitialized: true
  }));
  app.use(flash());
  ```

- Use flash messages in routes:
  ```javascript
  app.post('/login', (req, res) => {
    if (req.body.username === 'admin') {
      req.flash('success', 'Login successful!');
      res.redirect('/dashboard');
    } else {
      req.flash('error', 'Invalid credentials');
      res.redirect('/login');
    }
  });
  ```

- Display flash messages in EJS:
  ```ejs
  <% if (messages.success) { %>
    <p style="color: green;"><%= messages.success %></p>
  <% } %>
  <% if (messages.error) { %>
    <p style="color: red;"><%= messages.error %></p>
  <% } %>
  ```

## Authentication Messages
You can use middleware to display authentication-related messages (e.g., login success or error).

### Example:
- Middleware for authentication:
  ```javascript
  app.use((req, res, next) => {
    res.locals.isAuthenticated = req.session.isAuthenticated || false;
    next();
  });
  ```

- Use the variable in EJS:
  ```ejs
  <% if (isAuthenticated) { %>
    <p>Welcome back!</p>
  <% } else { %>
    <p>Please log in.</p>
  <% } %>
  ```
