# UI with XML Layouts

Layouts and resources
- Common layouts: ConstraintLayout, LinearLayout, FrameLayout.
- Use dp for sizes and sp for text.
- Keep strings in res/values/strings.xml.

Example: activity_main.xml (ConstraintLayout)
```xml
<?xml version="1.0" encoding="utf-8"?>
<androidx.constraintlayout.widget.ConstraintLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:padding="16dp">

    <EditText
        android:id="@+id/etA"
        android:layout_width="0dp"
        android:layout_height="wrap_content"
        android:inputType="numberDecimal"
        android:hint="First number"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintTop_toTopOf="parent"/>

    <EditText
        android:id="@+id/etB"
        android:layout_width="0dp"
        android:layout_height="wrap_content"
        android:inputType="numberDecimal"
        android:hint="Second number"
        app:layout_constraintTop_toBottomOf="@id/etA"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        android:layout_marginTop="12dp"/>

    <Button
        android:id="@+id/btnAdd"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Add"
        app:layout_constraintTop_toBottomOf="@id/etB"
        app:layout_constraintStart_toStartOf="parent"
        android:layout_marginTop="16dp"/>

    <TextView
        android:id="@+id/tvResult"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:textSize="24sp"
        app:layout_constraintTop_toBottomOf="@id/btnAdd"
        app:layout_constraintStart_toStartOf="parent"
        android:layout_marginTop="24dp"/>
</androidx.constraintlayout.widget.ConstraintLayout>
```

ViewBinding (recommended over findViewById)
- Enable in module build.gradle:
```gradle
android {
    buildFeatures { viewBinding true }
}
```

Use in Activity
```kotlin
private lateinit var binding: ActivityMainBinding

override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    binding = ActivityMainBinding.inflate(layoutInflater)
    setContentView(binding.root)

    binding.btnAdd.setOnClickListener {
        val a = binding.etA.text.toString().toDoubleOrNull()
        val b = binding.etB.text.toString().toDoubleOrNull()
        binding.tvResult.text = if (a != null && b != null) "${a + b}" else "Invalid" 
    }
}
```

Notes
- Use Material Components theme for modern UI.
- Keep layout simple; extract styles for reuse.
