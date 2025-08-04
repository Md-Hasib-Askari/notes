# 15. Rendering Performance Tips

## Caching Views
Cache rendered views to improve performance, especially for pages that do not change frequently.

### Example:
- Enable view caching in Express:
  ```javascript
  app.set('view cache', true);
  ```

## Using Minimal Logic in Templates
Move complex logic to controllers or middleware instead of handling it in EJS templates.

### Example:
- Controller logic:
  ```javascript
  app.get('/dashboard', (req, res) => {
    const userData = getUserData(); // Fetch data in the controller
    res.render('dashboard', { userData });
  });
  ```

- Template:
  ```ejs
  <h1>Welcome, <%= userData.name %></h1>
  ```

## Offloading Logic to Controllers
Keep your templates clean by preparing all necessary data in the controller before rendering.

### Example:
- Controller:
  ```javascript
  app.get('/profile', (req, res) => {
    const profileData = {
      name: 'John Doe',
      age: 30,
      hobbies: ['Reading', 'Cycling', 'Hiking']
    };
    res.render('profile', { profileData });
  });
  ```

- Template:
  ```ejs
  <h1>Profile</h1>
  <p>Name: <%= profileData.name %></p>
  <p>Age: <%= profileData.age %></p>
  <ul>
    <% profileData.hobbies.forEach(hobby => { %>
      <li><%= hobby %></li>
    <% }); %>
  </ul>
  ```
