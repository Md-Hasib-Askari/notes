# CI/CD Pipelines (GitHub Actions, Bitrise, EAS)

GitHub Actions (Android release)
```yaml
name: android-release
on: { workflow_dispatch: {}, push: { tags: ["v*"] } }
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-java@v4
        with: { distribution: 'temurin', java-version: '17' }
      - name: Cache Gradle
        uses: actions/cache@v4
        with:
          path: ~/.gradle/caches
          key: ${{ runner.os }}-gradle-${{ hashFiles('**/*.gradle*', '**/gradle-wrapper.properties') }}
      - name: Decrypt keystore
        run: |
          echo "$SIGNING_KEYSTORE_BASE64" | base64 -d > android/app/release.keystore
        env:
          SIGNING_KEYSTORE_BASE64: ${{ secrets.SIGNING_KEYSTORE_BASE64 }}
      - name: Build AAB
        run: cd android && ./gradlew bundleRelease
      - uses: actions/upload-artifact@v4
        with: { name: app-release, path: android/app/build/outputs/bundle/release/*.aab }
```

iOS build (requires macOS runner)
```yaml
runs-on: macos-14
steps:
  - uses: actions/checkout@v4
  - uses: maxim-lobanov/setup-xcode@v1
    with: { xcode-version: '15.3' }
  - name: Install pods
    run: cd ios && pod install --repo-update
  - name: Build archive
    run: xcodebuild -workspace ios/App.xcworkspace -scheme App -configuration Release -sdk iphoneos -archivePath $PWD/build/App.xcarchive archive
```

EAS Build (Expo/Hybrid)
```yaml
- uses: actions/setup-node@v4
  with: { node-version: 20 }
- run: npm i -g eas-cli
- run: eas build --platform all --non-interactive --profile production
  env:
    EXPO_TOKEN: ${{ secrets.EXPO_TOKEN }}
```

Bitrise outline
- Steps: Git Clone → Install NPM/Yarn → Install Pods → Android Build → iOS Build → Sign → Deploy to stores/TestFlight

Best practices
- Keep secrets in CI vaults; never commit keys
- Cache node_modules and Gradle/CocoaPods
- Use separate workflows for PR checks, nightly, and releases
- Upload sourcemaps to Sentry/Bugsnag during build
