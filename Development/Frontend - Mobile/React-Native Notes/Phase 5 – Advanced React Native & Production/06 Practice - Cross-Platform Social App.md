# Practice – Cross-Platform Social App

Goal
- Build a social app with feed, post creation (native camera processing), likes/comments, push notifications, and offline-first sync.

Features
- Auth: Firebase Auth (email/password or OAuth)
- Feed: paginated list of posts with images/videos
- Create Post: capture media; Android uses a native Kotlin module for camera pre-processing (e.g., compress, auto-rotate)
- Interactions: like/comment with optimistic updates
- Notifications: FCM/Expo notifications for likes/comments
- Offline: local DB cache, queued mutations, background sync

Architecture
- State: Redux Toolkit (RTK Query optional) or Zustand
- Data: SQLite/Realm + repository layer
- Media: react-native-image-picker or Expo ImagePicker + native module for processing
- Navigation: Tabs (Feed, Create, Profile) + Stacks
- Theming: light/dark, persisted

Implementation outline
- Week 1: Auth + basic navigation + feed skeleton
- Week 2: Post creation flow + native camera processing module
- Week 3: Offline cache + sync loop + optimistic UI
- Week 4: Notifications + polish + deployment to test tracks

Stretch goals
- Deep linking to profiles/posts
- Share sheet integration
- Background uploads with retries
- CDN and image resizing on server
