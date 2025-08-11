# Android Activity Lifecycle

Why it matters
- Activities are your screens; lifecycle drives initialization, UI updates, and resource cleanup.

Order (typical)
- onCreate → onStart → onResume → [running]
- onPause → onStop → onDestroy (finish) or onRestart → onStart → onResume

Basic example
```kotlin
class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        Log.d("Lifecycle", "onCreate")
    }
    override fun onStart() { super.onStart(); Log.d("Lifecycle", "onStart") }
    override fun onResume() { super.onResume(); Log.d("Lifecycle", "onResume") }
    override fun onPause() { super.onPause(); Log.d("Lifecycle", "onPause") }
    override fun onStop() { super.onStop(); Log.d("Lifecycle", "onStop") }
    override fun onDestroy() { super.onDestroy(); Log.d("Lifecycle", "onDestroy") }
}
```

Config changes and state
- Rotations recreate your Activity by default.
- Use onSaveInstanceState to persist transient UI state.
```kotlin
override fun onSaveInstanceState(outState: Bundle) {
    super.onSaveInstanceState(outState)
    outState.putString("result", binding.tvResult.text.toString())
}

override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    // ...
    val restored = savedInstanceState?.getString("result")
    binding.tvResult.text = restored ?: ""
}
```

Long-lived state
- Use ViewModel to survive rotation and keep business state out of Activities.
```kotlin
class CounterViewModel : ViewModel() { val count = MutableLiveData(0) }
```

Tips
- Heavy work → ViewModel + Kotlin coroutines.
- Release sensors/camera in onPause/onStop; reacquire in onResume.
