# 3. Basic Syntax

EJS provides several tags for embedding JavaScript into your HTML templates:

## Output Tags
- `<%= %>`: Outputs escaped content (safe for HTML).
  ```ejs
  <h1><%= user.name %></h1>
  ```
- `<%- %>`: Outputs unescaped content (use with caution).
  ```ejs
  <div><%- user.bio %></div>
  ```

## Scriptlet Tags
- `<% %>`: Executes JavaScript code without output.
  ```ejs
  <% if (user.isAdmin) { %>
    <p>Welcome, Admin!</p>
  <% } %>
  ```

## Trim-Mode Tags
- `<%_ _%>`: Removes whitespace around the tag.
  ```ejs
  <%_ for (let i = 0; i < items.length; i++) { _%>
    <li><%= items[i] %></li>
  <%_ } _%>
  ```
