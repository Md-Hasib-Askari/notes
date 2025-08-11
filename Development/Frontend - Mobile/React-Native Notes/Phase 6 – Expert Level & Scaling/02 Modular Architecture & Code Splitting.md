# Modular Architecture & Code Splitting

Goals
- Scale codebases with clear boundaries and faster startup

Suggested structure
- app/ (entry, providers)
- src/
  - features/
    - auth/
    - feed/
    - payments/
  - core/ (design tokens, theme, api, analytics)
  - shared/ (ui components, hooks, utils)
  - navigation/

Principles
- Feature-first modules; keep cross-cutting concerns in core/
- Dependency inversion: features depend on interfaces, not concrete impls
- Avoid deep relative imports; use path aliases ("@/feature/...")

Startup performance
- Enable inlineRequires to lazy-load modules
- Consider RAM bundles for large apps (Android)

metro.config.js
```js
const { getDefaultConfig } = require('@react-native/metro-config');
module.exports = (async () => {
  const config = await getDefaultConfig(__dirname);
  config.transformer = { ...config.transformer, unstable_allowRequireContext: true }; // optional
  config.serializer = { ...config.serializer, getModulesRunBeforeMainModule: () => [], getPolyfills: () => [] };
  config.transformer.getTransformOptions = async () => ({ transform: { experimentalImportSupport: false, inlineRequires: true } });
  return config;
})();
```

Lazy screens
- Defer heavy screens until needed; use React.lazy only if your RN version supports it reliably; otherwise split by navigation boundaries
- In React Navigation, use lazy: true on tabs; fetch data on focus

Runtime config
- Provide env-driven feature flags to enable/disable modules per brand

Bundle analysis
- Use react-native-bundle-visualizer to inspect bundle size

Do
- Keep each feature self-contained (routes, state, API)
- Use public API pattern: expose only index.ts at feature root

Avoid
- Cross-feature imports bypassing public API
- Global singletons leaking state across tests
