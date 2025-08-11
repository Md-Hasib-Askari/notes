# Styling

Style with objects; StyleSheet.create helps validate and precompute.

StyleSheet.create
```tsx
import { StyleSheet, View, Text } from 'react-native';

const s = StyleSheet.create({
  card: { padding: 16, backgroundColor: '#fff', borderRadius: 12, shadowOpacity: 0.1 },
  title: { fontSize: 18, fontWeight: '700' },
});

export default function Card() {
  return (
    <View style={s.card}>
      <Text style={s.title}>Card</Text>
    </View>
  );
}
```

Flexbox essentials
- Container: flexDirection, justifyContent, alignItems
- Children: flex, alignSelf
```tsx
<View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
  <View style={{ width: 60, height: 60, backgroundColor: '#f59e0b' }} />
  <View style={{ width: 60, height: 60, backgroundColor: '#10b981' }} />
  <View style={{ width: 60, height: 60, backgroundColor: '#3b82f6' }} />
</View>
```

Spacing and layout tips
- Use padding/margin, not absolute positioning, unless necessary.
- Prefer minWidth/minHeight over fixed sizes when possible.
- Safe areas: wrap iOS screens with SafeAreaView.
```tsx
import { SafeAreaView } from 'react-native-safe-area-context';

<SafeAreaView style={{ flex: 1, backgroundColor: '#fff' }}>
  {/* screen content */}
</SafeAreaView>
```

Responsive techniques
- useWindowDimensions for live size updates.
- Percent widths/heights for simple layouts.
- Platform-specific tweaks with Platform.select.
- Consider utility libs later: react-native-size-matters, react-native-responsive-screen.
```tsx
import { useWindowDimensions, View } from 'react-native';

function Box() {
  const { width } = useWindowDimensions();
  const box = Math.min(200, width * 0.5);
  return <View style={{ width: box, height: box, backgroundColor: '#9333ea' }} />;
}
```

Theming basics
- Centralize colors/spacing in theme files and pass via Context or a provider.
