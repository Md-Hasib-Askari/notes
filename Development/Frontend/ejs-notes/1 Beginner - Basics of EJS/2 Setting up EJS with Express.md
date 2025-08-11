# 2. Setting up EJS with Express.js

## Installing EJS
To install EJS, use the following command:
```bash
npm install ejs
```

## Configuring the View Engine
To use EJS as the templating engine in an Express.js application, configure it as follows:
```javascript
const express = require('express');
const app = express();

// Set EJS as the view engine
app.set('view engine', 'ejs');

// Define the views directory (optional, default is './views')
app.set('views', './views');

app.listen(3000, () => {
  console.log('Server is running on port 3000');
});
```

## Folder Structure
Ensure your `.ejs` files are placed in the `views` folder by default. For example:
```
project-folder/
  views/
    index.ejs
  app.js
```
