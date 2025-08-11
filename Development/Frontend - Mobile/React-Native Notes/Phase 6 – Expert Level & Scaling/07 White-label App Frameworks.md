# White-label App Frameworks

Goals
- Ship multiple branded apps from one codebase

Key techniques
- Theming: design tokens (colors, typography, spacing) per brand
- Config: brand.json files or env vars for logos, copy, features
- Assets: brand-specific icons/splash; conditional import by brand
- App IDs: Android productFlavors; iOS schemes/targets

Android flavors (build.gradle)
```gradle
android {
  productFlavors {
    brandA { applicationIdSuffix ".brandA"; resValue "string", "app_name", "MyApp A" }
    brandB { applicationIdSuffix ".brandB"; resValue "string", "app_name", "MyApp B" }
  }
}
```

iOS targets
- Duplicate target and scheme; set bundle identifiers per brand

Runtime selection
- Use a BRAND env to pick theme/config at build time (EAS profiles or CI matrix)

Automation
- Scripts to swap icons, generate splash screens, and validate brand assets
- Snapshot tests to ensure branding correctness

Docs
- Maintain a checklist for adding a new brand (IDs, icons, colors, copy, store listings)
