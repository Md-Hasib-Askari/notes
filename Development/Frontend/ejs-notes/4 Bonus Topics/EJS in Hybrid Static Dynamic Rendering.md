# EJS in Hybrid Static/Dynamic Rendering Apps

## Combining Static and Dynamic Rendering
EJS can be used to render static content at build time and dynamic content at runtime.

### Example:
- Render static content:
  ```javascript
  app.get('/static-page', (req, res) => {
    res.render('static', { title: 'Static Page', content: 'This is static content.' });
  });
  ```

  - `views/static.ejs`:
    ```ejs
    <h1><%= title %></h1>
    <p><%= content %></p>
    ```

- Add dynamic content with AJAX:
  ```javascript
  app.get('/api/dynamic-content', (req, res) => {
    res.json({ dynamicContent: 'This is dynamic content loaded via AJAX.' });
  });
  ```

- Frontend JavaScript (`public/js/dynamic.js`):
  ```javascript
  fetch('/api/dynamic-content')
    .then(response => response.json())
    .then(data => {
      const dynamicContent = document.getElementById('dynamic-content');
      dynamicContent.textContent = data.dynamicContent;
    });
  ```

  - `views/static.ejs` (updated):
    ```ejs
    <h1><%= title %></h1>
    <p><%= content %></p>
    <div id="dynamic-content">Loading dynamic content...</div>
    <script src="/js/dynamic.js"></script>
    ```
