# 8. Form Handling

## Building Forms Using EJS
You can create forms in EJS templates to collect user input.

### Example:
- `views/form.ejs`:
  ```ejs
  <form action="/submit" method="POST">
    <label for="name">Name:</label>
    <input type="text" id="name" name="name" required>
    <button type="submit">Submit</button>
  </form>
  ```

## Submitting Forms to Express Backend
Handle form submissions in your Express routes.

### Example:
- Define a POST route:
  ```javascript
  const bodyParser = require('body-parser');
  app.use(bodyParser.urlencoded({ extended: true }));

  app.post('/submit', (req, res) => {
    const name = req.body.name;
    res.render('result', { name });
  });
  ```

- Create the `result.ejs` template:
  ```ejs
  <h1>Form Submitted</h1>
  <p>Name: <%= name %></p>
  ```

## Preserving User Input and Displaying Errors
You can preserve user input and display error messages in the form.

### Example:
- Pass errors and input data to the template:
  ```javascript
  app.post('/submit', (req, res) => {
    const name = req.body.name;
    if (!name) {
      res.render('form', { error: 'Name is required', name });
    } else {
      res.render('result', { name });
    }
  });
  ```

- Update `form.ejs`:
  ```ejs
  <% if (error) { %>
    <p style="color: red;"><%= error %></p>
  <% } %>
  <form action="/submit" method="POST">
    <label for="name">Name:</label>
    <input type="text" id="name" name="name" value="<%= name %>" required>
    <button type="submit">Submit</button>
  </form>
  ```
