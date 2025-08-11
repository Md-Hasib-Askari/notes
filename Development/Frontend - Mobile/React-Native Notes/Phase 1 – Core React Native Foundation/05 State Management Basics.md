# State Management (Basics)

Local state with useState
```tsx
import { useState } from 'react';
import { View, Text, Button } from 'react-native';

export default function Counter() {
  const [count, setCount] = useState(0);
  return (
    <View>
      <Text>Count: {count}</Text>
      <Button title="+" onPress={() => setCount((c) => c + 1)} />
    </View>
  );
}
```

Side effects with useEffect
- Use for data fetching, subscriptions, timers.
```tsx
import { useEffect, useState } from 'react';

function Joke() {
  const [joke, setJoke] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    fetch('https://official-joke-api.appspot.com/jokes/random')
      .then((res) => res.json())
      .then((data) => {
        if (alive) setJoke(`${data.setup} ${data.punchline}`);
      })
      .catch(() => setError('Failed to load'));
    return () => {
      alive = false;
    };
  }, []);

  // ...render
}
```

Context API (lightweight global state)
```tsx
import { createContext, useContext, useMemo, useState } from 'react';

type Theme = 'light' | 'dark';

const ThemeContext = createContext<{ theme: Theme; toggle: () => void } | undefined>(undefined);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>('light');
  const value = useMemo(() => ({ theme, toggle: () => setTheme((t) => (t === 'light' ? 'dark' : 'light')) }), [theme]);
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error('useTheme must be used within ThemeProvider');
  return ctx;
}
```

When to scale up
- If state crosses many screens or needs persistence, consider Redux Toolkit or Zustand (covered in Phase 3).
