# WorkManager Background Tasks

Use WorkManager for deferrable, guaranteed background work.

CoroutineWorker example
```kotlin
class SyncWorker(appContext: Context, params: WorkerParameters) : CoroutineWorker(appContext, params) {
    override suspend fun doWork(): Result {
        val userId = inputData.getString("USER_ID") ?: return Result.failure()
        return try {
            // sync data to backend
            Result.success(workDataOf("count" to 42))
        } catch (e: Exception) {
            Result.retry()
        }
    }
}
```

Enqueue
```kotlin
val req = OneTimeWorkRequestBuilder<SyncWorker>()
    .setInputData(workDataOf("USER_ID" to "u1"))
    .setConstraints(Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build())
    .build()
WorkManager.getInstance(context).enqueueUniqueWork("sync", ExistingWorkPolicy.KEEP, req)
```

Periodic work
```kotlin
val periodic = PeriodicWorkRequestBuilder<SyncWorker>(15, TimeUnit.MINUTES).build()
WorkManager.getInstance(context).enqueueUniquePeriodicWork("sync-periodic", ExistingPeriodicWorkPolicy.UPDATE, periodic)
```

Observing
```kotlin
WorkManager.getInstance(context).getWorkInfoByIdLiveData(req.id).observe(this) { info ->
    if (info?.state?.isFinished == true) { val out = info.outputData.getInt("count", 0) }
}
```

Foreground service
- For long tasks, setForegroundAsync and show a notification

Notes
- Background limits differ across OEMs; keep tasks short and network-aware
