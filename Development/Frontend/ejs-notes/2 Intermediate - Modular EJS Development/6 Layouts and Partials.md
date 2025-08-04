# 6. Layouts and Partials

## Creating Header and Footer Partials
Partials are reusable EJS templates that can be included in other templates. For example, you can create a `header.ejs` and `footer.ejs` for your website layout.

### Example:
- `views/partials/header.ejs`:
  ```ejs
  <header>
    <h1>My Website</h1>
    <nav>
      <a href="/">Home</a>
      <a href="/about">About</a>
    </nav>
  </header>
  ```

- `views/partials/footer.ejs`:
  ```ejs
  <footer>
    <p>&copy; 2025 My Website</p>
  </footer>
  ```

## Including Partials in Templates
Use the `<%- include() %>` syntax to include partials in your main templates.

### Example:
- `views/index.ejs`:
  ```ejs
  <%- include('partials/header') %>
  <main>
    <h2>Welcome to My Website</h2>
    <p>This is the homepage.</p>
  </main>
  <%- include('partials/footer') %>
  ```

## Building Reusable Components
You can create reusable components like navigation bars, modals, or cards as partials and include them wherever needed. This approach promotes modularity and reduces code duplication.
