# Data Storage (AsyncStorage, SQLite, Realm, WatermelonDB)

Choosing storage
- AsyncStorage: small key-value; simple persistence
- SQLite: relational; good for lists, queries, joins
- Realm: object database; reactive; great DX
- WatermelonDB: scalable, sync-friendly; requires setup

AsyncStorage
```ts
import AsyncStorage from '@react-native-async-storage/async-storage';

export const storage = {
  async set<T>(key: string, value: T) {
    await AsyncStorage.setItem(key, JSON.stringify(value));
  },
  async get<T>(key: string): Promise<T | null> {
    const v = await AsyncStorage.getItem(key);
    return v ? JSON.parse(v) as T : null;
  },
  async remove(key: string) { await AsyncStorage.removeItem(key); },
};
```

SQLite (Expo)
```ts
import * as SQLite from 'expo-sqlite';
const db = SQLite.openDatabaseSync('app.db');

await db.execAsync(`CREATE TABLE IF NOT EXISTS tx (id TEXT PRIMARY KEY NOT NULL, amount REAL, category TEXT, note TEXT, createdAt INTEGER);`);
await db.runAsync('INSERT INTO tx (id, amount, category, createdAt) VALUES (?, ?, ?, ?)', [id, amount, category, Date.now()]);
const rows = await db.getAllAsync('SELECT * FROM tx ORDER BY createdAt DESC');
```

SQLite (Bare RN)
- Use react-native-sqlite-storage; API is callback-based; wrap with Promises.

Realm (cross-platform)
```ts
import Realm from 'realm';

class Tx extends Realm.Object<Tx> { _id!: string; amount!: number; category!: string; createdAt!: Date; static schema = {
  name: 'Tx', primaryKey: '_id', properties: { _id: 'string', amount: 'double', category: 'string', note: 'string?', createdAt: 'date' }
}; }

const realm = await Realm.open({ schema: [Tx] });
realm.write(() => { realm.create('Tx', { _id: id, amount, category, createdAt: new Date() }); });
const list = realm.objects<Tx>('Tx').sorted('createdAt', true);
```

WatermelonDB (overview)
- Define models and schema; sync via adapters
- Great for very large datasets and sync
- Consider only if you need complex sync or millions of rows

Patterns
- Repository layer: hide storage details behind functions (createTx, listTx)
- Migrations: write forward-only scripts when changing schema
- Indexing: add indices on frequently filtered fields (date, category)
