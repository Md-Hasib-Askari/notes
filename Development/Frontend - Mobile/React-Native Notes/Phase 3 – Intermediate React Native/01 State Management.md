# State Management (Redux Toolkit or Zustand)

When to use global state
- Cross-screen data (auth user, settings, cached lists)
- Complex updates, side effects, or debugging needs
- Local state is fine for isolated UI concerns

Option A: Redux Toolkit (RTK)
- Pros: Opinionated, great devtools, middleware, thunks, ecosystem
- Cons: Slightly more setup than Zustand

Install
- npm i @reduxjs/toolkit react-redux

Create a slice
```tsx
// src/state/transactionsSlice.ts
import { createSlice, PayloadAction, createAsyncThunk } from '@reduxjs/toolkit';

export type Tx = { id: string; amount: number; note?: string; category: string; createdAt: number };

export const fetchRates = createAsyncThunk('rates/fetch', async () => {
  const res = await fetch('https://api.exchangerate.host/latest?base=USD');
  if (!res.ok) throw new Error('Network');
  return (await res.json()) as any;
});

const slice = createSlice({
  name: 'tx',
  initialState: { items: [] as Tx[], rates: null as any, loading: false, error: null as string | null },
  reducers: {
    add(state, action: PayloadAction<Tx>) { state.items.push(action.payload); },
    remove(state, action: PayloadAction<string>) { state.items = state.items.filter(t => t.id !== action.payload); },
    clear(state) { state.items = []; },
  },
  extraReducers(builder) {
    builder
      .addCase(fetchRates.pending, (s) => { s.loading = true; s.error = null; })
      .addCase(fetchRates.fulfilled, (s, a) => { s.loading = false; s.rates = a.payload; })
      .addCase(fetchRates.rejected, (s, a) => { s.loading = false; s.error = a.error.message ?? 'Error'; });
  }
});

export const { add, remove, clear } = slice.actions;
export default slice.reducer;
```

Configure store
```tsx
// src/state/store.ts
import { configureStore } from '@reduxjs/toolkit';
import tx from './transactionsSlice';

export const store = configureStore({ reducer: { tx } });
export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
```

Provider and hooks
```tsx
// App.tsx
import { Provider } from 'react-redux';
import { store } from './src/state/store';

export default function App() {
  return (
    <Provider store={store}>
      {/* your navigators */}
    </Provider>
  );
}
```

Usage in components
```tsx
import { useDispatch, useSelector } from 'react-redux';
import { add, remove } from '../state/transactionsSlice';
import type { RootState } from '../state/store';

const items = useSelector((s: RootState) => s.tx.items);
const dispatch = useDispatch();

dispatch(add({ id: '1', amount: 12.5, category: 'Food', createdAt: Date.now() }));
```

Middleware basics
- RTK includes thunk by default
- Custom middleware example:
```ts
const logger = (storeAPI: any) => (next: any) => (action: any) => {
  console.log('dispatch', action.type);
  return next(action);
};
```

Persistence (simple)
- For light persistence, save critical bits to AsyncStorage on change and hydrate on app start.
- For full solution, use redux-persist later when needed.

Option B: Zustand
- Minimal, hooks-based store; great for local-ish global state

Install
- npm i zustand

Create store
```ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

type Tx = { id: string; amount: number; category: string; createdAt: number };

type State = {
  items: Tx[];
  add: (t: Tx) => void;
  remove: (id: string) => void;
};

export const useTxStore = create<State>()(persist((set) => ({
  items: [],
  add: (t) => set((s) => ({ items: [...s.items, t] })),
  remove: (id) => set((s) => ({ items: s.items.filter(i => i.id !== id) })),
}), { name: 'tx-store' }));
```

Choosing
- Prefer RTK for larger teams, devtools, middleware, and async orchestration
- Prefer Zustand for simple apps or feature-scoped global state
