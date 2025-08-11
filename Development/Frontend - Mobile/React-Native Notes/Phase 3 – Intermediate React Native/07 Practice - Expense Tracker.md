# Practice – Expense Tracker App

Goal
- Build an expense tracker with offline storage, category filters, and optional reminders via push notifications.

Core features
- Add/edit/delete transactions (amount, category, note, date)
- Summary: total, by category, by period
- Filter/search
- Offline-first: data persists locally; optional export/import JSON

Tech stack
- State: Redux Toolkit (or Zustand if you prefer)
- Storage: SQLite (Expo) or Realm (bare)
- Navigation: Stack + Tabs; deep link to Add screen (myapp://add)
- Notifications: Expo Notifications or FCM for bill reminders

Data model
```ts
type Tx = { id: string; amount: number; category: string; note?: string; date: number };
type Category = { id: string; name: string; icon?: string };
```

Suggested screens
- Tab: Home (summary), Transactions (list), Settings
- Stack: Add/Edit, Details

Flow
- Add screen uses react-hook-form + Yup
- On submit: dispatch Redux action; persist to DB
- List pulls from DB (SQLite SELECT or Realm query) and subscribes to updates

Deep linking
- Config route "Add" → path "add"; test by opening myapp://add on device

Reminders (optional)
- Schedule local notifications for recurring bills

Stretch goals
- Import/export CSV or JSON to Filesystem
- Charts (Victory Native or react-native-svg + d3)
- Light/dark theme persisted
- Sync to backend later (Phase 5)
