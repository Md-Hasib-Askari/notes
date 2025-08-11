# TypeScript in React Native

Setup
- npm i -D typescript @types/react @types/react-native
- Add tsconfig.json

tsconfig.json (suggested)
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["ES2020"],
    "jsx": "react-jsx",
    "moduleResolution": "node",
    "strict": true,
    "noEmit": true,
    "skipLibCheck": true,
    "baseUrl": ".",
    "paths": { "@/*": ["src/*"] }
  },
  "include": ["src", "App.tsx"]
}
```

Typing navigation (React Navigation)
```ts
type RootStackParamList = { Home: undefined; Details: { id: string } };

type DetailsRoute = RouteProp<RootStackParamList, 'Details'>;
function Details({ route }: { route: DetailsRoute }) { /* ... */ }
```

Typing Redux Toolkit
```ts
export const store = configureStore({ reducer: { /* ... */ } });
export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
export const useAppDispatch: () => AppDispatch = useDispatch;
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;
```

Props and components
```ts
type ButtonProps = { title: string; onPress: () => void; disabled?: boolean };
const PrimaryButton: React.FC<ButtonProps> = ({ title, onPress, disabled }) => { /* ... */ };
```

Tips
- Use discriminated unions for state machines
- Declare module aliases in Babel (module-resolver) to match tsconfig paths
- Add ESLint + typescript-eslint for consistency
