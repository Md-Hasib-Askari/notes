# Deployment (Android & iOS) + OTA

Android (Play Store)
1) App signing
- Generate a release keystore (keytool) and add to android/app
- Configure signingConfigs in android/app/build.gradle
2) Build release
- ./gradlew assembleRelease (APK) or bundleRelease (AAB)
3) Upload AAB to Play Console, set up listing, content rating, testing tracks, and roll out

iOS (App Store)
1) Certificates & Profiles
- Use Xcode automatic signing or manage via Apple Developer portal
2) Versioning
- Bump CFBundleShortVersionString and CFBundleVersion
3) Archive & upload
- Product → Archive, then Distribute to App Store Connect (or use Transporter)
4) TestFlight testing, then App Review

OTA Updates
- CodePush (bare RN)
  - Integrate appcenter/codepush SDK, create app in App Center, release keys per platform
  - Release JS bundles/assets to users without store review (policy-compliant changes only)
- Expo EAS Update (Expo/Hybrid)
  - Configure channel/branch, run eas update, use runtime versioning

Release hygiene
- Enable Hermes, Proguard/Minify and shrinkResources on Android
- Source maps for crash reporting (Sentry, Bugsnag)
- Feature flags for risky features
- Pre-launch reports and automated device testing
