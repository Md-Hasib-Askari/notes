# Offline Sync

Core concepts
- Source of truth: local DB (SQLite/Realm/WatermelonDB)
- Sync loop: push local changes, then pull remote
- Versioning: updatedAt or revision numbers; tombstones for deletes
- Conflict resolution: last-write-wins, server-wins, client-wins, or CRDTs

Queueing local changes
```ts
type Change = { id: string; type: 'create'|'update'|'delete'; entity: 'post'|'like'|'comment'; payload: any; updatedAt: number };
```

Sync cycle (pseudo)
```ts
async function sync() {
  const pending = await getPendingChanges();
  for (const ch of pending) {
    try { await pushChange(ch); markAsSynced(ch.id); }
    catch (e) { if (isRetryable(e)) backoff(ch); else markAsFailed(ch.id); }
  }
  const since = await getLastPulledAt();
  const { records, serverTime } = await pullChanges({ since });
  await applyRemote(records); // upserts + deletes
  await setLastPulledAt(serverTime);
}
```

Background sync
- Expo: expo-background-fetch + expo-task-manager
- Bare: WorkManager (Android), BackgroundFetch (iOS)
- Trigger on app start/resume and periodically when online

Conflicts
- Detect by comparing updatedAt/version fields
- Strategy examples:
  - LWW (simple): keep latest timestamp
  - Merge: combine fields if non-overlapping
  - Server resolution: send both versions; server decides

Best practices
- Keep payloads small; paginate pull
- Encrypt sensitive data at rest if required
- Provide manual "Sync now" and show status/errors to users
