# Advanced Navigation (React Navigation)

Deep linking
```ts
import { NavigationContainer } from '@react-navigation/native';
import * as Linking from 'expo-linking'; // or custom in bare

const linking = {
  prefixes: [Linking.createURL('/'), 'myapp://'],
  config: {
    screens: {
      Home: 'home',
      Add: 'add',
      TransactionDetails: 'tx/:id',
    },
  },
};

export default function AppNav() {
  return (
    <NavigationContainer linking={linking} fallback={<Text>Loading...</Text>}>
      {/* navigators */}
    </NavigationContainer>
  );
}
```

Passing params and typing
```ts
type RootStackParamList = { Home: undefined; Add: undefined; TransactionDetails: { id: string } };
```

Persisting navigation state
```ts
import AsyncStorage from '@react-native-async-storage/async-storage';

const PERSISTENCE_KEY = 'NAVIGATION_STATE_V1';

function Root() {
  const [initialState, setInitialState] = useState();

  useEffect(() => {
    const restore = async () => {
      const json = await AsyncStorage.getItem(PERSISTENCE_KEY);
      setInitialState(json ? JSON.parse(json) : undefined);
    };
    restore();
  }, []);

  return (
    <NavigationContainer
      initialState={initialState}
      onStateChange={(state) => AsyncStorage.setItem(PERSISTENCE_KEY, JSON.stringify(state))}
    >
      {/* navigators */}
    </NavigationContainer>
  );
}
```

Tips
- Keep params serializable
- Prefer nested navigators for feature areas (Tabs -> Stacks)
- Use useNavigation, useRoute hooks; avoid prop drilling
