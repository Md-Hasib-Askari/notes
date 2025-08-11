# Performance Optimization

Render optimization
- useMemo/useCallback to avoid re-creating objects/handlers
- React.memo for pure child components

Example
```tsx
const Item = React.memo(({ item }: { item: Tx }) => <Text>{item.note}</Text>);
const renderItem = useCallback(({ item }) => <Item item={item} />, []);
```

FlatList tuning
- Provide stable keyExtractor
- Memoize renderItem; avoid inline functions
- getItemLayout for fixed-height rows
- Adjust windowSize, initialNumToRender, maxToRenderPerBatch
- removeClippedSubviews on Android
```tsx
<FlatList
  data={data}
  keyExtractor={(i) => i.id}
  renderItem={renderItem}
  getItemLayout={(d, i) => ({ length: ROW, offset: ROW * i, index: i })}
  windowSize={10}
  initialNumToRender={12}
  maxToRenderPerBatch={12}
  removeClippedSubviews
/>
```

Images
- Use appropriate sizes; avoid rendering huge images
- Caching: Expo Image (expo-image) with cachePolicy; bare RN consider react-native-fast-image

Work scheduling
- Defer heavy work with InteractionManager.runAfterInteractions
- Split large lists with pagination/infinite scroll

JS/Engine settings
- Enable Hermes in production for better perf
- Disable dev features (dev menu, warnings) in release builds

Profiling
- Flipper: React DevTools, Network, Performance plugins
- Systrace/Android Studio Profiler for native bottlenecks

Anti-patterns
- Unbounded re-renders from context; consider selectors or store slices
- Large JSON parsing on UI thread; offload to Background/JSI/native if needed
