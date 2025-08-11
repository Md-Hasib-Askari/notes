# Push Notifications (Firebase Cloud Messaging)

Options
- Bare RN: @react-native-firebase/messaging
- Expo managed: expo-notifications (with Expo push service) or EAS + FCM/APNs

Permissions
- iOS: request user permission, add capabilities (Push Notifications, Background Modes)
- Android: POST_NOTIFICATIONS permission (Android 13+), channels for importance

Bare RN (FCM) basics
```ts
// Install @react-native-firebase/app and @react-native-firebase/messaging
import messaging from '@react-native-firebase/messaging';

// Ask permission and get FCM token
export async function initFCM() {
  const authStatus = await messaging().requestPermission();
  const enabled = authStatus === messaging.AuthorizationStatus.AUTHORIZED || authStatus === messaging.AuthorizationStatus.PROVISIONAL;
  if (!enabled) return null;
  const token = await messaging().getToken();
  return token; // send to your backend
}

// Foreground messages
messaging().onMessage(async (remoteMessage) => {
  // show in-app banner or local notification
});

// Background/quit state (index.js)
import messaging from '@react-native-firebase/messaging';
messaging().setBackgroundMessageHandler(async (remoteMessage) => {
  // handle background message
});
```

Expo managed
```ts
import * as Notifications from 'expo-notifications';
import * as Device from 'expo-device';

export async function registerForPush() {
  if (!Device.isDevice) return null;
  const { status: existingStatus } = await Notifications.getPermissionsAsync();
  let finalStatus = existingStatus;
  if (existingStatus !== 'granted') {
    const { status } = await Notifications.requestPermissionsAsync();
    finalStatus = status;
  }
  if (finalStatus !== 'granted') return null;
  const token = await Notifications.getExpoPushTokenAsync({ projectId: 'YOUR-PROJECT-ID' });
  if (Platform.OS === 'android') {
    await Notifications.setNotificationChannelAsync('default', { name: 'default', importance: Notifications.AndroidImportance.DEFAULT });
  }
  return token.data;
}
```

Notes
- Store device tokens securely; allow users to opt out
- Use topics or user-specific tokens on the backend
- Test on physical devices; emulators have limitations
