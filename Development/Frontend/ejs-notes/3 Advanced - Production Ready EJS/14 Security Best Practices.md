# 14. Security Best Practices

## Avoiding XSS (Cross-Site Scripting)
Always use `<%= %>` to escape user input and prevent XSS attacks. Avoid using `<%- %>` unless you are certain the content is safe.

### Example:
- Escaped output (safe):
  ```ejs
  <p><%= userInput %></p>
  ```

- Unescaped output (unsafe):
  ```ejs
  <p><%- userInput %></p>
  ```

## Sanitizing User Input
Use libraries like `validator` or `DOMPurify` to sanitize user input before rendering it in templates.

### Example:
- Install `validator`:
  ```bash
  npm install validator
  ```

- Sanitize input:
  ```javascript
  const validator = require('validator');
  const sanitizedInput = validator.escape(userInput);
  ```

## Content Security Policy (CSP)
Implement a CSP header to restrict the sources of content that can be loaded by your application.

### Example:
- Use the `helmet` middleware:
  ```bash
  npm install helmet
  ```

- Set up CSP:
  ```javascript
  const helmet = require('helmet');
  app.use(helmet.contentSecurityPolicy({
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'", 'trusted-cdn.com']
    }
  }));
  ```
