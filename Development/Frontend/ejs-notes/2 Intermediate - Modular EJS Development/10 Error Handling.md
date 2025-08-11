# 10. Error Handling

## Displaying Error Messages
You can display error messages in your EJS templates by passing error data from your Express routes.

### Example:
- Pass an error message to the template:
  ```javascript
  app.get('/error', (req, res) => {
    res.render('error', { message: 'Something went wrong!' });
  });
  ```

- Create the `error.ejs` template:
  ```ejs
  <h1>Error</h1>
  <p style="color: red;"><%= message %></p>
  ```

## Using Conditional Rendering for Alerts/Toasts
You can conditionally render alerts or toasts based on the presence of error or success messages.

### Example:
- Pass success and error messages to the template:
  ```javascript
  app.get('/status', (req, res) => {
    res.render('status', { success: 'Operation successful!', error: null });
  });
  ```

- Update the `status.ejs` template:
  ```ejs
  <% if (success) { %>
    <div style="color: green;"><%= success %></div>
  <% } %>
  <% if (error) { %>
    <div style="color: red;"><%= error %></div>
  <% } %>
  ```
