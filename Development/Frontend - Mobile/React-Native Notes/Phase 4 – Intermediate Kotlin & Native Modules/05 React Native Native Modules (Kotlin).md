# React Native Native Modules (Kotlin)

Goal: Create a simple Kotlin native module for a bare React Native app.

Module class
```kotlin
class DeviceInfoModule(private val reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {
    override fun getName() = "DeviceInfo"

    @ReactMethod
    fun getBatteryLevel(promise: Promise) {
      try {
        val bm = reactContext.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val level = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
        promise.resolve(level) // number passed to JS
      } catch (e: Exception) {
        promise.reject("ERR_BATTERY", e)
      }
    }

    override fun getConstants(): MutableMap<String, Any> = mutableMapOf(
        "manufacturer" to Build.MANUFACTURER,
        "model" to Build.MODEL
    )

    private fun sendEvent(name: String, params: WritableMap?) {
      reactContext.getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java).emit(name, params)
    }
}
```

Package
```kotlin
class MyAppPackage : ReactPackage {
  override fun createNativeModules(reactContext: ReactApplicationContext) = listOf(DeviceInfoModule(reactContext))
  override fun createViewManagers(reactContext: ReactApplicationContext) = emptyList<ViewManager<*, *>>()
}
```

Register package (MainApplication.kt)
```kotlin
override fun getPackages(): MutableList<ReactPackage> = mutableListOf(
  MainReactPackage(),
  MyAppPackage()
)
```

Use from JS
```ts
import { NativeModules, NativeEventEmitter } from 'react-native';
const { DeviceInfo } = NativeModules as { DeviceInfo: { getBatteryLevel: () => Promise<number>, manufacturer: string, model: string } };

const level = await DeviceInfo.getBatteryLevel();
const ee = new NativeEventEmitter(DeviceInfo);
const sub = ee.addListener('battery', (e) => console.log(e));
```

Notes
- Autolinking picks up modules if added under android/ with proper Gradle setup
- For permissions (e.g., CAMERA), request on Android side or via JS before calling
- New Architecture (TurboModules/Fabric) improves perf and type-safety; migrate later
