# 11. Organizing Views and Partials

## Folder Structure for Scalability
Organize your views and partials into a structured folder hierarchy to make your project scalable and maintainable.

### Example:
```
project-folder/
  views/
    layouts/
      main.ejs
    partials/
      header.ejs
      footer.ejs
    pages/
      home.ejs
      about.ejs
```

## Naming Conventions
- Use descriptive names for your files (e.g., `header.ejs`, `footer.ejs`).
- Group related templates into folders (e.g., `layouts`, `partials`, `pages`).

## Using Layout Wrappers for Specific Pages
Create layout wrappers to define the overall structure of your pages.

### Example:
- `views/layouts/main.ejs`:
  ```ejs
  <html>
  <head>
    <title><%= title %></title>
  </head>
  <body>
    <%- include('../partials/header') %>
    <main>
      <%- body %>
    </main>
    <%- include('../partials/footer') %>
  </body>
  </html>
  ```

- Render a page with the layout:
  ```javascript
  app.get('/', (req, res) => {
    res.render('layouts/main', { title: 'Home', body: '<h1>Welcome to My Website</h1>' });
  });
  ```
