# 4. Rendering Views

## Creating `.ejs` Files
EJS templates are stored in the `views` folder by default. For example:
```
project-folder/
  views/
    index.ejs
```

## Using `res.render()`
To render an EJS file, use the `res.render()` method in your Express route:
```javascript
app.get('/', (req, res) => {
  res.render('index', { title: 'Home Page', user: { name: 'John Doe' } });
});
```

## Passing Variables to Views
You can pass data to the EJS template as an object:
```javascript
res.render('index', { title: 'Welcome', items: ['Item 1', 'Item 2', 'Item 3'] });
```
In the EJS file, access the variables:
```ejs
<h1><%= title %></h1>
<ul>
  <% items.forEach(item => { %>
    <li><%= item %></li>
  <% }); %>
</ul>
```
