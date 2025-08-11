# Practice – Simple Calculator (Kotlin)

Goal
- Build a single-Activity calculator that adds, subtracts, multiplies, and divides two numbers.

Steps
1) Create a new Android Studio project (Empty Views Activity), language = Kotlin, min SDK 24+.
2) Enable ViewBinding in module build.gradle:
```gradle
android { buildFeatures { viewBinding true } }
```
3) Design UI in activity_main.xml with two EditText, four Buttons (+, −, ×, ÷), and a TextView for the result.
4) Implement logic in MainActivity using ViewBinding and safe parsing (toDoubleOrNull).
5) Handle rotation via onSaveInstanceState to preserve the result.

Sample layout (simplified)
```xml
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:padding="16dp">

    <EditText android:id="@+id/etA" android:layout_width="match_parent" android:layout_height="wrap_content" android:inputType="numberDecimal" android:hint="First number" />
    <EditText android:id="@+id/etB" android:layout_width="match_parent" android:layout_height="wrap_content" android:inputType="numberDecimal" android:hint="Second number" />

    <LinearLayout android:layout_width="match_parent" android:layout_height="wrap_content" android:orientation="horizontal">
        <Button android:id="@+id/btnAdd" android:layout_width="0dp" android:layout_height="wrap_content" android:layout_weight="1" android:text="+" />
        <Button android:id="@+id/btnSub" android:layout_width="0dp" android:layout_height="wrap_content" android:layout_weight="1" android:text="-" />
        <Button android:id="@+id/btnMul" android:layout_width="0dp" android:layout_height="wrap_content" android:layout_weight="1" android:text="×" />
        <Button android:id="@+id/btnDiv" android:layout_width="0dp" android:layout_height="wrap_content" android:layout_weight="1" android:text="÷" />
    </LinearLayout>

    <TextView android:id="@+id/tvResult" android:layout_width="wrap_content" android:layout_height="wrap_content" android:textSize="24sp" />
</LinearLayout>
```

MainActivity logic (core)
```kotlin
class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        fun parse(): Pair<Double?, Double?> =
            binding.etA.text.toString().toDoubleOrNull() to binding.etB.text.toString().toDoubleOrNull()

        fun show(x: Double?) { binding.tvResult.text = x?.toString() ?: "Invalid" }

        binding.btnAdd.setOnClickListener { val (a,b)=parse(); show(if(a!=null&&b!=null) a+b else null) }
        binding.btnSub.setOnClickListener { val (a,b)=parse(); show(if(a!=null&&b!=null) a-b else null) }
        binding.btnMul.setOnClickListener { val (a,b)=parse(); show(if(a!=null&&b!=null) a*b else null) }
        binding.btnDiv.setOnClickListener { val (a,b)=parse(); show(if(a!=null&&b!=null&&b!=0.0) a/b else null) }

        binding.tvResult.text = savedInstanceState?.getString("result") ?: ""
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putString("result", binding.tvResult.text.toString())
    }
}
```

Stretch goals
- Add a second Activity to view a history list (pass recent calculations via Intent extras or a simple in-memory singleton).
- Add input validation and error styles.
- Add unit tests for the pure math functions.
