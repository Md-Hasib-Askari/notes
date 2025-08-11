# Practice – Weather App

Goal: Build a simple weather app that
- Detects current location (permission)
- Fetches current weather from an API
- Uses navigation (Home → Details)

Tech choices
- Expo + React Navigation (stack)
- OpenWeatherMap (https://openweathermap.org/) free API

1) Bootstrap
- npx create-expo-app WeatherApp
- cd WeatherApp; npx expo start
- Install navigation:
  - npx expo install @react-navigation/native react-native-screens react-native-safe-area-context
  - npx expo install @react-navigation/native-stack

2) Screens
- HomeScreen: Shows current location weather (temp, condition, city)
- DetailsScreen: Shows more fields (humidity, wind, pressure)

3) Location + fetch
```tsx
// src/services/weather.ts
import * as Location from 'expo-location';

const API = 'https://api.openweathermap.org/data/2.5/weather';
const KEY = 'YOUR_API_KEY'; // For learning only; do not commit real keys.

export async function getLocalWeather() {
  const { status } = await Location.requestForegroundPermissionsAsync();
  if (status !== 'granted') throw new Error('Perms denied');
  const pos = await Location.getCurrentPositionAsync({});
  const url = `${API}?lat=${pos.coords.latitude}&lon=${pos.coords.longitude}&appid=${KEY}&units=metric`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Network error');
  return res.json();
}
```

4) HomeScreen example
```tsx
import { useEffect, useState } from 'react';
import { View, Text, ActivityIndicator, Button } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { getLocalWeather } from '../services/weather';

export default function HomeScreen() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const navigation = useNavigation();

  const load = async () => {
    try {
      setLoading(true);
      setError(null);
      const json = await getLocalWeather();
      setData(json);
    } catch (e) {
      setError('Failed to load weather');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  if (loading) return <ActivityIndicator style={{ marginTop: 40 }} />;
  if (error) return <Text>{error}</Text>;

  return (
    <View style={{ padding: 16 }}>
      <Text style={{ fontSize: 28, fontWeight: '700' }}>{Math.round(data.main.temp)}°C</Text>
      <Text>{data.weather?.[0]?.main}</Text>
      <Text>{data.name}</Text>
      <Button title="Details" onPress={() => navigation.navigate('Details' as never, { data } as never)} />
      <Button title="Reload" onPress={load} />
    </View>
  );
}
```

5) DetailsScreen example
```tsx
export default function DetailsScreen({ route }) {
  const { data } = route.params;
  return (
    <View style={{ padding: 16 }}>
      <Text>Humidity: {data.main.humidity}%</Text>
      <Text>Wind: {data.wind.speed} m/s</Text>
      <Text>Pressure: {data.main.pressure} hPa</Text>
    </View>
  );
}
```

6) Improvements (optional)
- Search by city name
- Store last result offline (AsyncStorage)
- Show icons and 5-day forecast (OpenWeatherMap One Call API)
- Extract UI into components and add basic theming

Notes
- Do not hardcode real API keys in repos. Use .env and a config layer in real projects.
- Test both on emulator and physical device if possible.
