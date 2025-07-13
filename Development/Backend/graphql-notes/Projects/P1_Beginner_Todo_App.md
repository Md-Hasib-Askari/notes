# Beginner Project: Todo App

## Project Overview
A simple GraphQL-powered todo application that demonstrates fundamental CRUD operations, basic authentication, and client-side integration. Perfect for beginners to apply GraphQL concepts in a real-world scenario.

## Core Features

### 1. Simple CRUD Operations
Complete Create, Read, Update, Delete functionality for todos.

### 2. Basic Authentication
User registration, login, and session management.

### 3. Client-Side Integration
Frontend application consuming the GraphQL API.

## Schema Design

```graphql
type User {
  id: ID!
  email: String!
  username: String!
  todos: [Todo!]!
  createdAt: String!
}

type Todo {
  id: ID!
  title: String!
  description: String
  completed: Boolean!
  user: User!
  createdAt: String!
  updatedAt: String!
}

type AuthPayload {
  token: String!
  user: User!
}

type Query {
  me: User
  todos: [Todo!]!
  todo(id: ID!): Todo
}

type Mutation {
  # Authentication
  register(email: String!, username: String!, password: String!): AuthPayload!
  login(email: String!, password: String!): AuthPayload!
  
  # Todo operations
  createTodo(title: String!, description: String): Todo!
  updateTodo(id: ID!, title: String, description: String, completed: Boolean): Todo!
  deleteTodo(id: ID!): Boolean!
}
```

## Server Implementation

### Setup and Dependencies
```javascript
// package.json dependencies
{
  "apollo-server-express": "^3.12.0",
  "express": "^4.18.0",
  "graphql": "^16.6.0",
  "jsonwebtoken": "^9.0.0",
  "bcryptjs": "^2.4.3",
  "mongoose": "^7.0.0",
  "dotenv": "^16.0.0"
}
```

### Server Setup
```javascript
const { ApolloServer } = require('apollo-server-express');
const express = require('express');
const mongoose = require('mongoose');
const jwt = require('jsonwebtoken');

const typeDefs = require('./schema');
const resolvers = require('./resolvers');

async function startServer() {
  const app = express();
  
  // Connect to MongoDB
  await mongoose.connect(process.env.MONGODB_URI);
  
  const server = new ApolloServer({
    typeDefs,
    resolvers,
    context: ({ req }) => {
      let user = null;
      const token = req.headers.authorization?.replace('Bearer ', '');
      
      if (token) {
        try {
          user = jwt.verify(token, process.env.JWT_SECRET);
        } catch (err) {
          console.log('Invalid token');
        }
      }
      
      return { user };
    }
  });
  
  await server.start();
  server.applyMiddleware({ app });
  
  app.listen(4000, () => {
    console.log(`Server running at http://localhost:4000${server.graphqlPath}`);
  });
}

startServer();
```

### Database Models
```javascript
// models/User.js
const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');

const userSchema = new mongoose.Schema({
  email: { type: String, required: true, unique: true },
  username: { type: String, required: true },
  password: { type: String, required: true }
}, { timestamps: true });

userSchema.pre('save', async function(next) {
  if (!this.isModified('password')) return next();
  this.password = await bcrypt.hash(this.password, 12);
  next();
});

userSchema.methods.comparePassword = async function(password) {
  return bcrypt.compare(password, this.password);
};

module.exports = mongoose.model('User', userSchema);

// models/Todo.js
const mongoose = require('mongoose');

const todoSchema = new mongoose.Schema({
  title: { type: String, required: true },
  description: String,
  completed: { type: Boolean, default: false },
  user: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true }
}, { timestamps: true });

module.exports = mongoose.model('Todo', todoSchema);
```

### Resolvers
```javascript
const jwt = require('jsonwebtoken');
const User = require('./models/User');
const Todo = require('./models/Todo');

const resolvers = {
  Query: {
    me: async (_, __, { user }) => {
      if (!user) throw new Error('Not authenticated');
      return await User.findById(user.id);
    },
    
    todos: async (_, __, { user }) => {
      if (!user) throw new Error('Not authenticated');
      return await Todo.find({ user: user.id }).populate('user');
    },
    
    todo: async (_, { id }, { user }) => {
      if (!user) throw new Error('Not authenticated');
      return await Todo.findOne({ _id: id, user: user.id }).populate('user');
    }
  },
  
  Mutation: {
    register: async (_, { email, username, password }) => {
      const existingUser = await User.findOne({ email });
      if (existingUser) throw new Error('User already exists');
      
      const user = await User.create({ email, username, password });
      const token = jwt.sign({ id: user.id }, process.env.JWT_SECRET);
      
      return { token, user };
    },
    
    login: async (_, { email, password }) => {
      const user = await User.findOne({ email });
      if (!user) throw new Error('Invalid credentials');
      
      const isValid = await user.comparePassword(password);
      if (!isValid) throw new Error('Invalid credentials');
      
      const token = jwt.sign({ id: user.id }, process.env.JWT_SECRET);
      return { token, user };
    },
    
    createTodo: async (_, { title, description }, { user }) => {
      if (!user) throw new Error('Not authenticated');
      
      const todo = await Todo.create({
        title,
        description,
        user: user.id
      });
      
      return await Todo.findById(todo.id).populate('user');
    },
    
    updateTodo: async (_, { id, ...updates }, { user }) => {
      if (!user) throw new Error('Not authenticated');
      
      const todo = await Todo.findOneAndUpdate(
        { _id: id, user: user.id },
        updates,
        { new: true }
      ).populate('user');
      
      if (!todo) throw new Error('Todo not found');
      return todo;
    },
    
    deleteTodo: async (_, { id }, { user }) => {
      if (!user) throw new Error('Not authenticated');
      
      const result = await Todo.deleteOne({ _id: id, user: user.id });
      return result.deletedCount > 0;
    }
  },
  
  User: {
    todos: async (user) => {
      return await Todo.find({ user: user.id });
    }
  }
};

module.exports = resolvers;
```

## Client-Side Integration

### React with Apollo Client
```jsx
// App.js
import { ApolloClient, InMemoryCache, ApolloProvider, createHttpLink } from '@apollo/client';
import { setContext } from '@apollo/client/link/context';
import TodoApp from './components/TodoApp';

const httpLink = createHttpLink({
  uri: 'http://localhost:4000/graphql',
});

const authLink = setContext((_, { headers }) => {
  const token = localStorage.getItem('token');
  return {
    headers: {
      ...headers,
      authorization: token ? `Bearer ${token}` : "",
    }
  }
});

const client = new ApolloClient({
  link: authLink.concat(httpLink),
  cache: new InMemoryCache()
});

function App() {
  return (
    <ApolloProvider client={client}>
      <TodoApp />
    </ApolloProvider>
  );
}

export default App;
```

### Todo Components
```jsx
// components/TodoList.js
import { useQuery, useMutation, gql } from '@apollo/client';

const GET_TODOS = gql`
  query GetTodos {
    todos {
      id
      title
      description
      completed
      createdAt
    }
  }
`;

const TOGGLE_TODO = gql`
  mutation ToggleTodo($id: ID!, $completed: Boolean!) {
    updateTodo(id: $id, completed: $completed) {
      id
      completed
    }
  }
`;

const DELETE_TODO = gql`
  mutation DeleteTodo($id: ID!) {
    deleteTodo(id: $id)
  }
`;

function TodoList() {
  const { loading, error, data, refetch } = useQuery(GET_TODOS);
  const [toggleTodo] = useMutation(TOGGLE_TODO);
  const [deleteTodo] = useMutation(DELETE_TODO, {
    refetchQueries: [{ query: GET_TODOS }]
  });

  if (loading) return <p>Loading...</p>;
  if (error) return <p>Error: {error.message}</p>;

  return (
    <div>
      <h2>My Todos</h2>
      {data.todos.map(todo => (
        <div key={todo.id} className="todo-item">
          <input
            type="checkbox"
            checked={todo.completed}
            onChange={() => toggleTodo({
              variables: { id: todo.id, completed: !todo.completed }
            })}
          />
          <span className={todo.completed ? 'completed' : ''}>
            {todo.title}
          </span>
          <button onClick={() => deleteTodo({ variables: { id: todo.id } })}>
            Delete
          </button>
        </div>
      ))}
    </div>
  );
}

export default TodoList;
```

## Learning Outcomes

### GraphQL Concepts Mastered
- Schema design and type definitions
- Query and mutation operations
- Context and authentication
- Error handling
- Client-side queries and mutations

### Development Skills
- Server setup with Apollo Server
- Database integration with MongoDB
- JWT authentication
- React integration with Apollo Client
- State management with GraphQL cache

### Best Practices
- Proper error handling
- Authentication middleware
- Database relationships
- Client-side caching
- Environment configuration

## Next Steps
1. Add input validation
2. Implement pagination
3. Add real-time updates with subscriptions
4. Deploy to production
5. Add unit and integration tests
