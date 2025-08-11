# Deep Linking & Universal Links

React Navigation linking config
```ts
const linking = {
  prefixes: ['myapp://', 'https://app.example.com'],
  config: { screens: { Home: 'home', Profile: 'u/:id', Post: 'p/:id' } }
};
```

iOS Universal Links
- Enable Associated Domains capability: applinks:app.example.com
- Host apple-app-site-association at https://app.example.com/apple-app-site-association with JSON content (no extension)
- Ensure paths match your routes

Android App Links
- intent-filter in AndroidManifest.xml with autoVerify=true
- Host .well-known/assetlinks.json on your domain referencing your appId and certificate fingerprints

Firebase Dynamic Links (optional)
- Handles cross-platform link resolution and install flows
- Configure domain (e.g., example.page.link) and map to your routes

Testing
- iOS: use Notes app or Safari to open https links
- Android: adb shell am start -a android.intent.action.VIEW -d "myapp://p/123"

Gotchas
- Universal/app links bypass the chooser when verified; ensure you can handle 404/unknown paths
- Handle cold start vs warm start navigation paths carefully
