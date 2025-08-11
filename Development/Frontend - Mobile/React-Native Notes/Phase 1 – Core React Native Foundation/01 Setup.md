# Setup

This guide is Windows-centric and uses PowerShell. Adjust paths if needed.

1) Install Node.js (LTS)
- Download from https://nodejs.org and install LTS, or run in PowerShell:
  - winget install OpenJS.NodeJS.LTS
- Verify: node -v; npm -v
- Choose one package manager (npm, yarn, or pnpm). npm is fine for now.

2) Choose RN workflow
- React Native CLI (bare): Full native access, more setup. Best for custom native modules.
- Expo: Fast DX, managed native dependencies, great for learning and rapid prototyping.

3) Android Studio + SDK
- Install Android Studio (latest stable). During setup:
  - Android SDK Platform (e.g., Android 14 / API 34+)
  - Android SDK Platform-Tools
  - Android Emulator
- Create an Android Virtual Device (AVD) via Tools → Device Manager.
- Environment variables (usually auto-set by Android Studio). Typical paths:
  - ANDROID_HOME: C:\Users\<you>\AppData\Local\Android\Sdk
  - Add to PATH: %ANDROID_HOME%\platform-tools; %ANDROID_HOME%\emulator
- Enable USB debugging on a real device if testing on hardware.

4) Create a project
- Expo (recommended for Phase 1)
  - npx create-expo-app MyFirstRNApp
  - cd MyFirstRNApp
  - npx expo start
  - Press a for Android emulator, or scan the QR with Expo Go on your phone.
- React Native CLI (bare)
  - npx react-native init MyFirstRNApp
  - cd MyFirstRNApp
  - Start Metro: npx react-native start
  - In another terminal: npx react-native run-android

5) Folder structure overview (suggested)
- App.tsx or App.js: App entry
- src/
  - screens/: Screen components (e.g., HomeScreen.tsx)
  - components/: Reusable UI (e.g., Button.tsx)
  - navigation/: Navigation setup (e.g., RootNavigator.tsx)
  - hooks/: Custom hooks (e.g., useWeather.ts)
  - services/: API clients (e.g., api.ts)
  - assets/: Images, fonts
  - theme/: Colors, spacing
  - utils/: Helpers

6) Useful VS Code extensions
- ES7+ React/Redux/JS snippets
- Prettier – Code formatter
- ESLint
- React Native Tools (optional)

7) Troubleshooting quick checks
- Android emulator doesn’t show up: Ensure AVD is running and platform-tools in PATH.
- Stuck bundling: Stop Metro, clear cache: npx react-native start --reset-cache
- Device not detected: adb devices; if empty, reconnect USB and enable USB debugging.
