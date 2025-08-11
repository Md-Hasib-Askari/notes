# Core Components

Essentials you’ll use daily.

View and Text
```tsx
import { View, Text } from 'react-native';

export default function Hello() {
  return (
    <View style={{ padding: 16 }}>
      <Text style={{ fontSize: 18, fontWeight: '600' }}>Hello React Native</Text>
    </View>
  );
}
```

Image
```tsx
import { Image } from 'react-native';

<Image
  source={{ uri: 'https://picsum.photos/200' }}
  style={{ width: 120, height: 120, borderRadius: 8 }}
  resizeMode="cover"
/>
```

ScrollView vs FlatList
- ScrollView: Renders all children; good for small content.
- FlatList: Virtualized; efficient for long lists.

```tsx
import { FlatList, Text } from 'react-native';

const data = Array.from({ length: 100 }, (_, i) => ({ id: String(i), title: `Item ${i}` }));

<FlatList
  data={data}
  keyExtractor={(item) => item.id}
  renderItem={({ item }) => <Text style={{ padding: 12 }}>{item.title}</Text>}
  ItemSeparatorComponent={() => <Text style={{ opacity: 0.2 }}>—</Text>}
/>
```

Inputs and touchables
```tsx
import { useState } from 'react';
import { View, TextInput, Button, TouchableOpacity, Text } from 'react-native';

export default function Inputs() {
  const [name, setName] = useState('');

  return (
    <View style={{ gap: 12, padding: 16 }}>
      <TextInput
        placeholder="Your name"
        value={name}
        onChangeText={setName}
        style={{ borderWidth: 1, borderColor: '#ddd', borderRadius: 8, padding: 12 }}
      />
      <Button title="Submit" onPress={() => console.log(name)} />

      <TouchableOpacity onPress={() => console.log('Pressed')} style={{ padding: 12, backgroundColor: '#007aff', borderRadius: 8 }}>
        <Text style={{ color: 'white', textAlign: 'center' }}>Custom Button</Text>
      </TouchableOpacity>
    </View>
  );
}
```

Keyboard handling
- On iOS, TextInput may be covered by the keyboard. Use KeyboardAvoidingView.
```tsx
import { KeyboardAvoidingView, Platform } from 'react-native';

<KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={{ flex: 1 }}>
  {/* your form */}
</KeyboardAvoidingView>
```
