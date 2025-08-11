# Passing Data Between RN and Native

Supported types
- boolean, number (Double), String
- Arrays (WritableArray), Maps (WritableMap)
- null

Kotlin → JS (maps/arrays)
```kotlin
val map = Arguments.createMap().apply {
  putString("id", "n1")
  putDouble("amount", 12.5)
  putArray("tags", Arguments.fromList(listOf("food", "work")))
}
sendEvent("txAdded", map)
```

JS → Kotlin (params)
```kotlin
@ReactMethod
fun echo(value: String, promise: Promise) { promise.resolve(value) }
```

Callbacks and Promises
```kotlin
@ReactMethod
fun computeAsync(a: Double, b: Double, callback: Callback) {
  Thread { callback.invoke(a + b) }.start()
}

@ReactMethod
fun computePromise(a: Double, b: Double, promise: Promise) { promise.resolve(a + b) }
```

Events
```kotlin
private fun sendEvent(name: String, params: WritableMap?) {
  reactApplicationContext.getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java).emit(name, params)
}
```

Guidelines
- Keep payloads small; for large data, persist to file/DB and pass references
- Validate input from JS; defensive programming on native side
