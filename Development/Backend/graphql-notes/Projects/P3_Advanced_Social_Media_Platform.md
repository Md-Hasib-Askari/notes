# Advanced Project: Social Media Platform

## Project Overview
A comprehensive social media platform built with GraphQL Federation architecture, featuring real-time messaging, advanced caching strategies, and mobile app integration. This project demonstrates expert-level GraphQL concepts and enterprise-scale architecture patterns.

## Architecture Overview

### Microservices with GraphQL Federation
- **User Service**: Authentication, profiles, relationships
- **Content Service**: Posts, media, comments, reactions
- **Messaging Service**: Direct messages, group chats
- **Notification Service**: Real-time notifications
- **Search Service**: Content discovery and search
- **Analytics Service**: User engagement and metrics

## Federated Schema Design

### User Service Schema
```graphql
# User Service
type User @key(fields: "id") {
  id: ID!
  username: String!
  email: String!
  profile: UserProfile!
  isOnline: Boolean!
  lastSeen: String
  followers: [User!]!
  following: [User!]!
  followerCount: Int!
  followingCount: Int!
  createdAt: String!
}

type UserProfile {
  displayName: String!
  bio: String
  avatar: Image
  coverImage: Image
  website: String
  location: String
  birthDate: String
  isPrivate: Boolean!
  isVerified: Boolean!
}

extend type Query {
  me: User
  user(username: String!): User
  searchUsers(query: String!, first: Int, after: String): UserConnection!
  suggestedFollows: [User!]!
}

extend type Mutation {
  updateProfile(input: UpdateProfileInput!): User!
  followUser(userId: ID!): FollowResult!
  unfollowUser(userId: ID!): Boolean!
  blockUser(userId: ID!): Boolean!
  reportUser(userId: ID!, reason: String!): Boolean!
}

extend type Subscription {
  userOnlineStatus(userId: ID!): User!
}
```

### Content Service Schema
```graphql
# Content Service
type Post @key(fields: "id") {
  id: ID!
  author: User! @provides(fields: "username")
  content: String!
  media: [Media!]!
  hashtags: [String!]!
  mentions: [User!]!
  reactions: [Reaction!]!
  reactionCounts: [ReactionCount!]!
  comments: [Comment!]!
  commentCount: Int!
  shares: [Share!]!
  shareCount: Int!
  visibility: PostVisibility!
  isEdited: Boolean!
  isPinned: Boolean!
  location: Location
  createdAt: String!
  updatedAt: String!
}

extend type User @key(fields: "id") {
  id: ID! @external
  posts: [Post!]!
  likedPosts: [Post!]!
  savedPosts: [Post!]!
  postCount: Int!
}

type Media {
  id: ID!
  type: MediaType!
  url: String!
  thumbnailUrl: String
  alt: String
  width: Int
  height: Int
  duration: Int # For videos
  size: Int
}

enum MediaType {
  IMAGE
  VIDEO
  GIF
  AUDIO
}

type Comment @key(fields: "id") {
  id: ID!
  post: Post!
  author: User!
  content: String!
  replies: [Comment!]!
  replyCount: Int!
  reactions: [Reaction!]!
  parent: Comment
  createdAt: String!
}

type Reaction {
  id: ID!
  user: User!
  type: ReactionType!
  createdAt: String!
}

enum ReactionType {
  LIKE
  LOVE
  LAUGH
  ANGRY
  SAD
  WOW
}

type ReactionCount {
  type: ReactionType!
  count: Int!
}

enum PostVisibility {
  PUBLIC
  FOLLOWERS
  FRIENDS
  PRIVATE
}

extend type Query {
  feed(first: Int, after: String): PostConnection!
  post(id: ID!): Post
  explorePosts(first: Int, after: String): PostConnection!
  trendingHashtags: [Hashtag!]!
  postsByHashtag(hashtag: String!, first: Int, after: String): PostConnection!
}

extend type Mutation {
  createPost(input: CreatePostInput!): Post!
  updatePost(id: ID!, input: UpdatePostInput!): Post!
  deletePost(id: ID!): Boolean!
  reactToPost(postId: ID!, reaction: ReactionType!): Reaction!
  removeReaction(postId: ID!): Boolean!
  sharePost(postId: ID!, content: String): Share!
  savePost(postId: ID!): Boolean!
  reportPost(postId: ID!, reason: String!): Boolean!
}

extend type Subscription {
  postAdded(userId: ID!): Post!
  postUpdated(postId: ID!): Post!
  reactionAdded(postId: ID!): Reaction!
  commentAdded(postId: ID!): Comment!
}
```

### Messaging Service Schema
```graphql
# Messaging Service
type Conversation @key(fields: "id") {
  id: ID!
  type: ConversationType!
  participants: [User!]!
  name: String
  avatar: Image
  lastMessage: Message
  unreadCount: Int!
  isActive: Boolean!
  createdAt: String!
  updatedAt: String!
}

enum ConversationType {
  DIRECT
  GROUP
}

type Message @key(fields: "id") {
  id: ID!
  conversation: Conversation!
  sender: User!
  content: String!
  media: [Media!]!
  replyTo: Message
  reactions: [MessageReaction!]!
  isEdited: Boolean!
  isDeleted: Boolean!
  deliveredTo: [User!]!
  readBy: [MessageRead!]!
  createdAt: String!
  updatedAt: String!
}

type MessageReaction {
  user: User!
  emoji: String!
  createdAt: String!
}

type MessageRead {
  user: User!
  readAt: String!
}

extend type User @key(fields: "id") {
  id: ID! @external
  conversations: [Conversation!]!
  blockedUsers: [User!]!
}

extend type Query {
  conversations: [Conversation!]!
  conversation(id: ID!): Conversation
  messages(conversationId: ID!, first: Int, after: String): MessageConnection!
  searchMessages(query: String!, conversationId: ID): [Message!]!
}

extend type Mutation {
  createConversation(input: CreateConversationInput!): Conversation!
  sendMessage(input: SendMessageInput!): Message!
  editMessage(id: ID!, content: String!): Message!
  deleteMessage(id: ID!): Boolean!
  reactToMessage(messageId: ID!, emoji: String!): MessageReaction!
  markAsRead(conversationId: ID!): Boolean!
  leaveConversation(conversationId: ID!): Boolean!
  addParticipants(conversationId: ID!, userIds: [ID!]!): Conversation!
}

extend type Subscription {
  messageAdded(conversationId: ID!): Message!
  messageUpdated(conversationId: ID!): Message!
  messageDeleted(conversationId: ID!): ID!
  messagingActivity(conversationId: ID!): MessagingActivity!
  conversationUpdated(userId: ID!): Conversation!
}

type MessagingActivity {
  user: User!
  type: ActivityType!
  conversationId: ID!
}

enum ActivityType {
  TYPING
  STOPPED_TYPING
  ONLINE
  OFFLINE
}
```

## Advanced Implementation

### Federation Gateway Setup
```javascript
const { ApolloGateway, IntrospectAndCompose } = require('@apollo/gateway');
const { ApolloServer } = require('apollo-server-express');

const gateway = new ApolloGateway({
  supergraphSdl: new IntrospectAndCompose({
    subgraphs: [
      { name: 'users', url: 'http://user-service:4001/graphql' },
      { name: 'content', url: 'http://content-service:4002/graphql' },
      { name: 'messaging', url: 'http://messaging-service:4003/graphql' },
      { name: 'notifications', url: 'http://notification-service:4004/graphql' },
      { name: 'search', url: 'http://search-service:4005/graphql' },
      { name: 'analytics', url: 'http://analytics-service:4006/graphql' }
    ]
  }),
  
  buildService({ url }) {
    return new RemoteGraphQLDataSource({
      url,
      willSendRequest({ request, context }) {
        // Forward authentication
        if (context.user) {
          request.http.headers.set('x-user-id', context.user.id);
          request.http.headers.set('x-user-role', context.user.role);
        }
      }
    });
  }
});

const server = new ApolloServer({
  gateway,
  context: ({ req }) => {
    // Authentication context
    const token = req.headers.authorization?.replace('Bearer ', '');
    const user = verifyToken(token);
    return { user };
  },
  subscriptions: {
    path: '/subscriptions',
    onConnect: (connectionParams) => {
      const token = connectionParams.authToken;
      const user = verifyToken(token);
      return { user };
    }
  }
});
```

### Advanced Caching with Redis
```javascript
const Redis = require('ioredis');
const { RedisCache } = require('apollo-server-cache-redis');

const redis = new Redis({
  host: process.env.REDIS_HOST,
  port: process.env.REDIS_PORT,
  retryDelayOnFailover: 100,
  maxRetriesPerRequest: 3
});

// Multi-level caching strategy
class AdvancedCacheStore {
  constructor() {
    this.redis = redis;
    this.localCache = new Map();
    this.localCacheTTL = 30 * 1000; // 30 seconds
  }
  
  async get(key) {
    // Check local cache first
    const localValue = this.localCache.get(key);
    if (localValue && localValue.expires > Date.now()) {
      return localValue.data;
    }
    
    // Check Redis
    const redisValue = await this.redis.get(key);
    if (redisValue) {
      const parsed = JSON.parse(redisValue);
      // Store in local cache
      this.localCache.set(key, {
        data: parsed,
        expires: Date.now() + this.localCacheTTL
      });
      return parsed;
    }
    
    return undefined;
  }
  
  async set(key, value, ttl = 300) {
    const serialized = JSON.stringify(value);
    
    // Store in Redis with TTL
    await this.redis.setex(key, ttl, serialized);
    
    // Store in local cache
    this.localCache.set(key, {
      data: value,
      expires: Date.now() + this.localCacheTTL
    });
  }
  
  async invalidate(pattern) {
    // Invalidate Redis keys
    const keys = await this.redis.keys(pattern);
    if (keys.length > 0) {
      await this.redis.del(...keys);
    }
    
    // Invalidate local cache
    for (const [key] of this.localCache) {
      if (key.match(pattern.replace('*', '.*'))) {
        this.localCache.delete(key);
      }
    }
  }
}

const cacheStore = new AdvancedCacheStore();

// Cached resolver example
const resolvers = {
  Query: {
    feed: async (_, { first, after }, { user, dataSources }) => {
      const cacheKey = `feed:${user.id}:${first}:${after}`;
      
      let feed = await cacheStore.get(cacheKey);
      if (!feed) {
        feed = await dataSources.contentAPI.getFeed(user.id, first, after);
        await cacheStore.set(cacheKey, feed, 60); // Cache for 1 minute
      }
      
      return feed;
    }
  },
  
  Mutation: {
    createPost: async (_, { input }, { user, dataSources }) => {
      const post = await dataSources.contentAPI.createPost(user.id, input);
      
      // Invalidate related caches
      await cacheStore.invalidate(`feed:*`);
      await cacheStore.invalidate(`user:${user.id}:posts`);
      
      return post;
    }
  }
};
```

### Real-time Features with GraphQL Subscriptions
```javascript
const { RedisPubSub } = require('graphql-redis-subscriptions');
const { withFilter } = require('graphql-subscriptions');

const pubsub = new RedisPubSub({
  connection: {
    host: process.env.REDIS_HOST,
    port: process.env.REDIS_PORT
  }
});

// Events
const MESSAGE_ADDED = 'MESSAGE_ADDED';
const USER_ONLINE_STATUS = 'USER_ONLINE_STATUS';
const POST_ADDED = 'POST_ADDED';
const NOTIFICATION_SENT = 'NOTIFICATION_SENT';

const subscriptionResolvers = {
  Subscription: {
    messageAdded: {
      subscribe: withFilter(
        () => pubsub.asyncIterator(MESSAGE_ADDED),
        (payload, variables, context) => {
          return payload.messageAdded.conversation.participants
            .some(p => p.id === context.user.id);
        }
      )
    },
    
    userOnlineStatus: {
      subscribe: withFilter(
        () => pubsub.asyncIterator(USER_ONLINE_STATUS),
        (payload, variables) => {
          return payload.userOnlineStatus.id === variables.userId;
        }
      )
    },
    
    postAdded: {
      subscribe: withFilter(
        () => pubsub.asyncIterator(POST_ADDED),
        async (payload, variables, context) => {
          // Only send to followers
          const isFollowing = await checkIfFollowing(
            context.user.id,
            payload.postAdded.author.id
          );
          return isFollowing;
        }
      )
    }
  }
};

// Connection tracking for presence
const connectionStore = new Map();

const server = new ApolloServer({
  // ... other config
  subscriptions: {
    onConnect: async (connectionParams, webSocket) => {
      const user = await authenticateUser(connectionParams.authToken);
      
      if (user) {
        connectionStore.set(webSocket, user);
        
        // Update user online status
        await updateUserOnlineStatus(user.id, true);
        pubsub.publish(USER_ONLINE_STATUS, {
          userOnlineStatus: { ...user, isOnline: true }
        });
        
        return { user };
      }
      
      throw new Error('Authentication failed');
    },
    
    onDisconnect: async (webSocket) => {
      const user = connectionStore.get(webSocket);
      if (user) {
        connectionStore.delete(webSocket);
        
        // Check if user has other active connections
        const hasOtherConnections = Array.from(connectionStore.values())
          .some(u => u.id === user.id);
        
        if (!hasOtherConnections) {
          await updateUserOnlineStatus(user.id, false);
          pubsub.publish(USER_ONLINE_STATUS, {
            userOnlineStatus: { ...user, isOnline: false, lastSeen: new Date() }
          });
        }
      }
    }
  }
});
```

### Mobile App Integration
```javascript
// React Native Apollo Client Setup
import { ApolloClient, InMemoryCache, split, HttpLink } from '@apollo/client';
import { getMainDefinition } from '@apollo/client/utilities';
import { WebSocketLink } from '@apollo/client/link/ws';
import { setContext } from '@apollo/client/link/context';
import AsyncStorage from '@react-native-async-storage/async-storage';

const httpLink = new HttpLink({
  uri: 'https://api.socialapp.com/graphql'
});

const wsLink = new WebSocketLink({
  uri: 'wss://api.socialapp.com/subscriptions',
  options: {
    reconnect: true,
    connectionParams: async () => {
      const token = await AsyncStorage.getItem('authToken');
      return {
        authToken: token
      };
    }
  }
});

const authLink = setContext(async (_, { headers }) => {
  const token = await AsyncStorage.getItem('authToken');
  return {
    headers: {
      ...headers,
      authorization: token ? `Bearer ${token}` : ""
    }
  };
});

const splitLink = split(
  ({ query }) => {
    const definition = getMainDefinition(query);
    return (
      definition.kind === 'OperationDefinition' &&
      definition.operation === 'subscription'
    );
  },
  wsLink,
  authLink.concat(httpLink)
);

const cache = new InMemoryCache({
  typePolicies: {
    User: {
      fields: {
        followers: {
          merge: false
        },
        following: {
          merge: false
        }
      }
    },
    Query: {
      fields: {
        feed: {
          keyArgs: [],
          merge(existing, incoming) {
            if (!existing) return incoming;
            return {
              ...incoming,
              edges: [...existing.edges, ...incoming.edges]
            };
          }
        },
        conversations: {
          merge: false
        }
      }
    }
  }
});

const client = new ApolloClient({
  link: splitLink,
  cache,
  defaultOptions: {
    watchQuery: {
      errorPolicy: 'all'
    }
  }
});
```

### Real-time Chat Component
```jsx
import React, { useEffect, useRef } from 'react';
import { useQuery, useSubscription, useMutation } from '@apollo/client';

const MESSAGES_QUERY = gql`
  query Messages($conversationId: ID!, $first: Int, $after: String) {
    messages(conversationId: $conversationId, first: $first, after: $after) {
      edges {
        node {
          id
          content
          sender {
            id
            username
            profile {
              avatar {
                url
              }
            }
          }
          createdAt
        }
      }
      pageInfo {
        hasNextPage
        endCursor
      }
    }
  }
`;

const MESSAGE_ADDED_SUBSCRIPTION = gql`
  subscription MessageAdded($conversationId: ID!) {
    messageAdded(conversationId: $conversationId) {
      id
      content
      sender {
        id
        username
        profile {
          avatar {
            url
          }
        }
      }
      createdAt
    }
  }
`;

const SEND_MESSAGE_MUTATION = gql`
  mutation SendMessage($input: SendMessageInput!) {
    sendMessage(input: $input) {
      id
      content
      createdAt
    }
  }
`;

function ChatRoom({ conversationId }) {
  const messagesEndRef = useRef(null);
  const [sendMessage] = useMutation(SEND_MESSAGE_MUTATION);
  
  const { data, loading, fetchMore } = useQuery(MESSAGES_QUERY, {
    variables: { conversationId, first: 50 },
    notifyOnNetworkStatusChange: true
  });
  
  useSubscription(MESSAGE_ADDED_SUBSCRIPTION, {
    variables: { conversationId },
    onSubscriptionData: ({ subscriptionData, client }) => {
      const newMessage = subscriptionData.data.messageAdded;
      
      // Add to cache
      client.cache.modify({
        fields: {
          messages(existingMessages = { edges: [] }) {
            const newEdge = {
              __typename: 'MessageEdge',
              node: newMessage
            };
            
            return {
              ...existingMessages,
              edges: [...existingMessages.edges, newEdge]
            };
          }
        }
      });
      
      // Scroll to bottom
      scrollToBottom();
    }
  });
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  
  useEffect(() => {
    scrollToBottom();
  }, [data]);
  
  const handleSendMessage = async (content) => {
    try {
      await sendMessage({
        variables: {
          input: {
            conversationId,
            content
          }
        },
        optimisticResponse: {
          sendMessage: {
            __typename: 'Message',
            id: `temp-${Date.now()}`,
            content,
            createdAt: new Date().toISOString()
          }
        }
      });
    } catch (error) {
      console.error('Failed to send message:', error);
    }
  };
  
  if (loading) return <div>Loading...</div>;
  
  return (
    <div className="chat-room">
      <div className="messages">
        {data?.messages.edges.map(({ node: message }) => (
          <div key={message.id} className="message">
            <img src={message.sender.profile.avatar?.url} alt="Avatar" />
            <div>
              <span>{message.sender.username}</span>
              <p>{message.content}</p>
              <small>{new Date(message.createdAt).toLocaleTimeString()}</small>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      
      <MessageInput onSend={handleSendMessage} />
    </div>
  );
}
```

## Performance Optimization

### Query Complexity Analysis
```javascript
const depthLimit = require('graphql-depth-limit');
const costAnalysis = require('graphql-cost-analysis');

const server = new ApolloServer({
  // ... other config
  validationRules: [
    depthLimit(10),
    costAnalysis({
      maximumCost: 1000,
      scalarCost: 1,
      objectCost: 5,
      listFactor: 10,
      introspectionCost: 1000
    })
  ]
});
```

### Database Optimization
```javascript
// Efficient pagination with cursor-based approach
const getPaginatedPosts = async (userId, first, after) => {
  const query = {
    author: { $in: await getFollowingIds(userId) },
    ...(after && { _id: { $lt: after } })
  };
  
  const posts = await Post.find(query)
    .sort({ _id: -1 })
    .limit(first + 1)
    .populate('author', 'username profile.avatar')
    .lean();
  
  const hasNextPage = posts.length > first;
  const edges = posts.slice(0, first);
  
  return {
    edges: edges.map(post => ({ node: post, cursor: post._id })),
    pageInfo: {
      hasNextPage,
      endCursor: edges[edges.length - 1]?._id
    }
  };
};
```

## Learning Outcomes

### Expert GraphQL Concepts
- GraphQL Federation and microservices architecture
- Advanced subscription patterns and real-time features
- Multi-level caching strategies
- Performance optimization and query analysis
- Mobile app integration patterns
- Production-scale error handling and monitoring

### Enterprise Architecture
- Service mesh and inter-service communication
- Event-driven architecture with message queues
- Distributed caching and state management
- Security at scale with authentication and authorization
- Monitoring, logging, and observability
- DevOps practices for GraphQL services

### Advanced Integration
- Real-time collaboration features
- Offline-first mobile applications
- Advanced UI patterns with optimistic updates
- Machine learning integration for recommendations
- Analytics and user behavior tracking
- Content delivery and media optimization

This project represents the pinnacle of GraphQL development, combining all advanced concepts into a production-ready, scalable social media platform.
