# Navigation (React Navigation)

Install
- Expo
  - npx expo install @react-navigation/native react-native-screens react-native-safe-area-context
  - npx expo install @react-navigation/native-stack
  - For tabs/drawer: npx expo install @react-navigation/bottom-tabs @react-navigation/drawer react-native-gesture-handler react-native-reanimated
- Bare React Native
  - npm i @react-navigation/native @react-navigation/native-stack
  - npm i react-native-screens react-native-safe-area-context
  - Tabs/Drawer: npm i @react-navigation/bottom-tabs @react-navigation/drawer react-native-gesture-handler react-native-reanimated
  - Follow reanimated setup (Babel plugin) per docs when you reach animations.

Stack navigator
```tsx
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import Home from '../screens/Home';
import Details from '../screens/Details';

type RootStackParamList = {
  Home: undefined;
  Details: { id: string };
};

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function RootNavigator() {
  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen name="Home" component={Home} />
        <Stack.Screen name="Details" component={Details} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
```

Passing params
```tsx
// Navigate
navigation.navigate('Details', { id: '42' });

// Read params in Details
function Details({ route }) {
  const { id } = route.params;
  // ...
}
```

Tabs and Drawer (quick start)
```tsx
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
const Tab = createBottomTabNavigator();

<Tab.Navigator>
  <Tab.Screen name="Home" component={Home} />
  <Tab.Screen name="Settings" component={Settings} />
</Tab.Navigator>
```

Useful APIs
- useNavigation for imperative actions
- setOptions to change header title/buttons per screen
- Link component (web-like) when using deep linking later
