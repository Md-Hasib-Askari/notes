# Security

Secure storage
- Tokens/refresh tokens: use platform keystores
  - Expo: expo-secure-store
  - Bare: react-native-keychain

Expo SecureStore
```ts
import * as SecureStore from 'expo-secure-store';
await SecureStore.setItemAsync('token', jwt, { keychainService: 'auth' });
const token = await SecureStore.getItemAsync('token');
```

react-native-keychain
```ts
import * as Keychain from 'react-native-keychain';
await Keychain.setGenericPassword('auth', jwt, { service: 'auth' });
const creds = await Keychain.getGenericPassword({ service: 'auth' });
```

API keys & config
- Do not hardcode secrets in JS bundles
- Use env files and native config (Gradle resources, iOS plist) or EAS secrets

Network
- Enforce HTTPS; validate certificates
- Consider certificate pinning (native: OkHttp; JS: axios + native module)
- Handle JWT refresh securely; rotate tokens

App hardening
- Avoid logging PII or secrets
- WebView: restrict origins and enable safe settings
- Clipboard: avoid copying secrets

Compliance
- Follow OWASP MAS guidelines
- Respect platform privacy requirements and permissions descriptions
