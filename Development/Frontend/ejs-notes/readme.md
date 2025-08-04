## 🟢 Beginner Level: Basics of EJS

### 1. Introduction to EJS

* What is EJS?
* Why use EJS over plain HTML?
* Difference between EJS and other templating engines (Pug, Handlebars)

### 2. Setting up EJS with Express.js

* Installing `ejs`
* Configuring the view engine in Express

  ```js
  app.set('view engine', 'ejs');
  ```

### 3. Basic Syntax

* `<%= %>`: Output escaped value
* `<%- %>`: Output unescaped HTML
* `<% %>`: Scriptlet for logic (no output)
* `<%_ _%>`: Trim-mode tags (remove whitespace)

### 4. Rendering Views

* Creating `.ejs` files in the `views` folder
* Using `res.render('filename', data)`
* Passing variables to views

### 5. Basic Control Structures

* `if`, `else`
* `for` loops
* Iterating through arrays

### ✅ Mini Project:

* Build a simple blog homepage with title, author, and date.

---

## 🟡 Intermediate Level: Modular EJS Development

### 6. Layouts and Partials

* Creating header and footer partials
* `<%- include('partials/header') %>`
* Building reusable components

### 7. Dynamic Routing with EJS

* Route parameters (`/user/:id`)
* Rendering data conditionally based on route

### 8. Form Handling

* Building forms using EJS
* Submitting forms to Express backend
* Preserving user input and displaying errors

### 9. EJS with CSS/JS

* Linking static assets in EJS (`public` folder)
* Using relative paths for stylesheets, scripts, images

### 10. Error Handling

* Displaying error messages
* Using conditional rendering for alerts/toasts

### ✅ Project:

* Create a task manager with add/delete functionality using EJS views.

---

## 🔴 Advanced Level: Production-Ready EJS

### 11. Organizing Views and Partials

* Folder structure for scalability
* Naming conventions
* Using layout wrappers for specific pages

### 12. Advanced Conditional Rendering

* Nested loops and conditionals
* Ternary operations inside EJS

### 13. Using EJS with Middleware

* Flash messages (e.g., `express-flash`)
* Authentication messages (e.g., login success/error)

### 14. Security Best Practices

* Avoiding XSS: Use `<%= %>` instead of `<%- %>` unless absolutely necessary
* Sanitizing user input before rendering

### 15. Rendering Performance Tips

* Caching views
* Using minimal logic in templates
* Offloading logic to controllers

### ✅ Capstone Project:

* Build a full-stack CMS (e.g., blog or school site) with:

  * Admin login
  * EJS templating
  * Dynamic routing
  * CRUD pages with modals/forms
  * Multilingual support (if needed)

---

## 📚 Bonus Topics

* EJS with AJAX (combine with frontend JS)
* EJS in hybrid static/dynamic rendering apps
* Creating a custom helper function in EJS
