# EJS with AJAX

## Combining EJS with Frontend JavaScript
EJS can be used to render initial HTML, while AJAX can dynamically update parts of the page without reloading.

### Example:
- Render the initial page with EJS:
  ```javascript
  app.get('/items', (req, res) => {
    res.render('items', { items: [] });
  });
  ```

  - `views/items.ejs`:
    ```ejs
    <h1>Items</h1>
    <ul id="item-list">
      <% items.forEach(item => { %>
        <li><%= item %></li>
      <% }); %>
    </ul>
    <button id="load-items">Load Items</button>
    <script src="/js/items.js"></script>
    ```

- Use AJAX to fetch data:
  ```javascript
  app.get('/api/items', (req, res) => {
    res.json(['Item 1', 'Item 2', 'Item 3']);
  });
  ```

- Frontend JavaScript (`public/js/items.js`):
  ```javascript
  document.getElementById('load-items').addEventListener('click', () => {
    fetch('/api/items')
      .then(response => response.json())
      .then(data => {
        const itemList = document.getElementById('item-list');
        data.forEach(item => {
          const li = document.createElement('li');
          li.textContent = item;
          itemList.appendChild(li);
        });
      });
  });
  ```
