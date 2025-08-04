# 5. Basic Control Structures

EJS allows you to use JavaScript control structures directly in your templates.

## Conditional Statements
- `if` and `else`:
  ```ejs
  <% if (user.isLoggedIn) { %>
    <p>Welcome, <%= user.name %>!</p>
  <% } else { %>
    <p>Please log in.</p>
  <% } %>
  ```

## Loops
- `for` loop:
  ```ejs
  <ul>
    <% for (let i = 0; i < items.length; i++) { %>
      <li><%= items[i] %></li>
    <% } %>
  </ul>
  ```

- `forEach` loop:
  ```ejs
  <ul>
    <% items.forEach(item => { %>
      <li><%= item %></li>
    <% }); %>
  </ul>
  ```

## Iterating Through Arrays
You can iterate through arrays using any JavaScript loop or array method:
```ejs
<% for (let item of items) { %>
  <p><%= item %></p>
<% } %>
```
