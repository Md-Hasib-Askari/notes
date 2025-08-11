# Intents and Navigation

Explicit intents (navigate to a specific Activity)
```kotlin
// From MainActivity to DetailsActivity
val intent = Intent(this, DetailsActivity::class.java)
intent.putExtra("USER_ID", 42)
startActivity(intent)
```

Read extras
```kotlin
class DetailsActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_details)
        val userId = intent.getIntExtra("USER_ID", -1)
    }
}
```

Get a result back (Activity Result APIs)
```kotlin
class MainActivity : AppCompatActivity() {
    private val pickContact = registerForActivityResult(ActivityResultContracts.PickContact()) { uri ->
        // handle uri (may be null)
    }
    fun openPicker() { pickContact.launch(null) }
}
```

Custom Activity for result
```kotlin
val launcher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
    if (result.resultCode == Activity.RESULT_OK) {
        val sum = result.data?.getDoubleExtra("SUM", 0.0)
        // use sum
    }
}

val i = Intent(this, SumActivity::class.java)
launcher.launch(i)
```

Implicit intents (delegate to other apps)
```kotlin
// Open URL
startActivity(Intent(Intent.ACTION_VIEW, Uri.parse("https://developer.android.com")))

// Dial number
startActivity(Intent(Intent.ACTION_DIAL, Uri.parse("tel:123456789")))
```

Tips
- Declare exported Activities safely in AndroidManifest (android:exported).
- Validate and sanitize any data passed via intents.
