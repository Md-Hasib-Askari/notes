# 7. Dynamic Routing with EJS

## Route Parameters
Dynamic routes allow you to capture values from the URL and use them in your EJS templates.

### Example:
- Define a route with a parameter:
  ```javascript
  app.get('/user/:id', (req, res) => {
    const userId = req.params.id;
    res.render('user', { id: userId });
  });
  ```

- Create the `user.ejs` template:
  ```ejs
  <h1>User Profile</h1>
  <p>User ID: <%= id %></p>
  ```

## Conditional Rendering Based on Route
You can render different content in your EJS templates based on the route parameters or query strings.

### Example:
- Pass additional data to the template:
  ```javascript
  app.get('/user/:id', (req, res) => {
    const userId = req.params.id;
    const isAdmin = userId === '1';
    res.render('user', { id: userId, isAdmin });
  });
  ```

- Use conditional rendering in `user.ejs`:
  ```ejs
  <h1>User Profile</h1>
  <p>User ID: <%= id %></p>
  <% if (isAdmin) { %>
    <p>Welcome, Admin!</p>
  <% } else { %>
    <p>Welcome, User!</p>
  <% } %>
  ```
