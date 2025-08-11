# Device APIs (File system, Permissions, Background tasks)

File system
- Expo: expo-file-system
- Bare: react-native-fs

Example (Expo)
```ts
import * as FileSystem from 'expo-file-system';

const dir = FileSystem.documentDirectory + 'exports/';
await FileSystem.makeDirectoryAsync(dir, { intermediates: true });
await FileSystem.writeAsStringAsync(dir + 'report.json', JSON.stringify({ hello: 'world' }));
const info = await FileSystem.getInfoAsync(dir + 'report.json');
```

Permissions
- Use react-native-permissions for unified API (bare)
- Expo modules request internally (e.g., expo-camera, expo-location)

react-native-permissions example
```ts
import { request, PERMISSIONS, RESULTS } from 'react-native-permissions';

const res = await request(Platform.select({ ios: PERMISSIONS.IOS.CAMERA, android: PERMISSIONS.ANDROID.CAMERA })!);
if (res === RESULTS.GRANTED) {
  // proceed
}
```

Background tasks
- Expo: expo-task-manager + expo-background-fetch
- Bare Android: Headless JS; iOS: Background Fetch/Push (limited)

Expo example
```ts
import * as TaskManager from 'expo-task-manager';
import * as BackgroundFetch from 'expo-background-fetch';

const TASK = 'sync-task';

TaskManager.defineTask(TASK, async () => {
  try {
    // sync or cleanup
    return BackgroundFetch.BackgroundFetchResult.NewData;
  } catch {
    return BackgroundFetch.BackgroundFetchResult.Failed;
  }
});

await BackgroundFetch.registerTaskAsync(TASK, { minimumInterval: 15 * 60, stopOnTerminate: false, startOnBoot: true });
```

Caveats
- Background execution is constrained; keep work brief and network-aware
- Respect user battery/data; provide toggles for background features
