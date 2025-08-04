# 9. EJS with CSS/JS

## Linking Static Assets in EJS
To serve static files like CSS, JavaScript, and images, use the `express.static` middleware.

### Example:
- Set up the `public` folder in your Express app:
  ```javascript
  app.use(express.static('public'));
  ```

- Folder structure:
  ```
  project-folder/
    public/
      css/
        styles.css
      js/
        script.js
    views/
      index.ejs
    app.js
  ```

- Link static files in your EJS template:
  ```ejs
  <link rel="stylesheet" href="/css/styles.css">
  <script src="/js/script.js"></script>
  ```

## Using Relative Paths for Assets
When linking assets, use relative paths based on the `public` folder.

### Example:
- `styles.css`:
  ```css
  body {
    font-family: Arial, sans-serif;
    background-color: #f4f4f4;
  }
  ```

- `script.js`:
  ```javascript
  console.log('JavaScript is working!');
  ```

- `index.ejs`:
  ```ejs
  <html>
  <head>
    <link rel="stylesheet" href="/css/styles.css">
  </head>
  <body>
    <h1>Welcome to My Website</h1>
    <script src="/js/script.js"></script>
  </body>
  </html>
  ```
