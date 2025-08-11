# Animations (Reanimated, Gesture Handler, Animated API)

Install
- Expo: npx expo install react-native-reanimated react-native-gesture-handler
- Bare: npm i react-native-reanimated react-native-gesture-handler
  - Add Reanimated Babel plugin and ensure Gesture Handler setup per docs

Animated API (baseline)
```tsx
import { Animated, Easing } from 'react-native';

const opacity = new Animated.Value(0);
Animated.timing(opacity, { toValue: 1, duration: 500, easing: Easing.out(Easing.cubic), useNativeDriver: true }).start();
```

Reanimated 2/3 shared values
```tsx
import Animated, { useSharedValue, useAnimatedStyle, withTiming, withSpring } from 'react-native-reanimated';

export default function Box() {
  const scale = useSharedValue(1);
  const style = useAnimatedStyle(() => ({ transform: [{ scale: scale.value }] }));
  return (
    <Animated.View style={[{ width: 100, height: 100, backgroundColor: '#3b82f6' }, style]} onTouchStart={() => { scale.value = withSpring(1.1); }} onTouchEnd={() => { scale.value = withTiming(1); }} />
  );
}
```

Gestures
```tsx
import { Gesture, GestureDetector } from 'react-native-gesture-handler';
import Animated, { useSharedValue, useAnimatedStyle } from 'react-native-reanimated';

export default function PanCard() {
  const x = useSharedValue(0); const y = useSharedValue(0);
  const pan = Gesture.Pan().onChange(e => { x.value += e.changeX; y.value += e.changeY; });
  const style = useAnimatedStyle(() => ({ transform: [{ translateX: x.value }, { translateY: y.value }] }));
  return (
    <GestureDetector gesture={pan}>
      <Animated.View style={[{ width: 120, height: 120, backgroundColor: '#10b981', borderRadius: 12 }, style]} />
    </GestureDetector>
  );
}
```

Layout animations
```tsx
import Animated, { Layout, FadeIn, FadeOut } from 'react-native-reanimated';

<Animated.View entering={FadeIn} exiting={FadeOut} layout={Layout.spring()} />
```

Tips
- Prefer Reanimated for smooth 60fps and gesture-driven animations
- Keep work on the UI thread via worklets; avoid setState in animation loops
- Profile with the Performance Monitor and Flipper
