# Creating a Custom Helper Function in EJS

## Why Use Helper Functions?
Helper functions simplify repetitive tasks in EJS templates, such as formatting dates or generating HTML snippets.

## Defining a Helper Function
You can define helper functions in your Express app and pass them to EJS templates.

### Example:
- Define a helper function:
  ```javascript
  const formatDate = (date) => {
    const options = { year: 'numeric', month: 'long', day: 'numeric' };
    return new Date(date).toLocaleDateString(undefined, options);
  };

  app.locals.formatDate = formatDate;
  ```

- Use the helper function in EJS:
  ```javascript
  app.get('/events', (req, res) => {
    const events = [
      { name: 'Event 1', date: '2025-08-01' },
      { name: 'Event 2', date: '2025-08-15' }
    ];
    res.render('events', { events });
  });
  ```

  - `views/events.ejs`:
    ```ejs
    <h1>Upcoming Events</h1>
    <ul>
      <% events.forEach(event => { %>
        <li><%= event.name %> - <%= formatDate(event.date) %></li>
      <% }); %>
    </ul>
    ```
