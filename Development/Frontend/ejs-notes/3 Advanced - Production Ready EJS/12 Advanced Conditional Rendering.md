# 12. Advanced Conditional Rendering

## Nested Loops and Conditionals
EJS allows you to nest loops and conditionals to create complex templates.

### Example:
- Nested loops:
  ```ejs
  <ul>
    <% categories.forEach(category => { %>
      <li>
        <%= category.name %>
        <ul>
          <% category.items.forEach(item => { %>
            <li><%= item %></li>
          <% }); %>
        </ul>
      </li>
    <% }); %>
  </ul>
  ```

## Ternary Operations Inside EJS
Use ternary operators for concise conditional rendering.

### Example:
- Render content based on a condition:
  ```ejs
  <p><%= user.isAdmin ? 'Welcome, Admin!' : 'Welcome, User!' %></p>
  ```

- Render attributes conditionally:
  ```ejs
  <button class="<%= isActive ? 'active' : 'inactive' %>">Click Me</button>
  ```
