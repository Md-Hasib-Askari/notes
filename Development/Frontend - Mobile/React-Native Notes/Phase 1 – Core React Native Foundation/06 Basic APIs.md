# Basic APIs

Fetching data
```tsx
import { useEffect, useState } from 'react';

export function useFetch<T = unknown>(url: string) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    fetch(url, { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error(String(res.status));
        return res.json();
      })
      .then(setData)
      .catch((e) => {
        if (e.name !== 'AbortError') setError('Failed to fetch');
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [url]);

  return { data, loading, error };
}
```

Axios (optional)
```ts
// services/api.ts
import axios from 'axios';
export const api = axios.create({ baseURL: 'https://api.example.com' });
```

Permissions and device features

Location (Expo)
```tsx
import * as Location from 'expo-location';

export async function getCurrentCoords() {
  const { status } = await Location.requestForegroundPermissionsAsync();
  if (status !== 'granted') throw new Error('Location permission denied');
  const pos = await Location.getCurrentPositionAsync({});
  return { lat: pos.coords.latitude, lon: pos.coords.longitude };
}
```

Image Picker (react-native-image-picker)
```tsx
import { launchImageLibrary } from 'react-native-image-picker';

export function pickImage() {
  return launchImageLibrary({ mediaType: 'photo', selectionLimit: 1 });
}
```

Notes
- On Android, declare required permissions in AndroidManifest when using native modules.
- On Expo, prefer expo-location and expo-image-picker which handle permissions.
