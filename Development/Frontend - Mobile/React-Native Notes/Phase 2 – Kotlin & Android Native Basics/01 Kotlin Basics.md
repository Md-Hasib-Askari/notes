# Kotlin Basics

Variables and types
```kotlin
val pi: Double = 3.1415   // immutable
var counter = 0           // mutable; type inferred as Int
val name = "Kotlin ${pi}" // string templates
```

Functions
```kotlin
fun greet(who: String = "world"): String = "Hello, $who"

fun sum(a: Int, b: Int): Int {
    return a + b
}

// Higher-order + lambda
fun operate(a: Int, b: Int, op: (Int, Int) -> Int): Int = op(a, b)
val result = operate(3, 4) { x, y -> x + y }

// Extension function
fun String.shout(): String = this.uppercase()
```

Control flow
```kotlin
val x = 7
val desc = when {
    x < 0 -> "negative"
    x in 0..9 -> "single-digit"
    else -> "multi-digit"
}

for (i in 0 until 3) println(i)
```

Collections
```kotlin
val nums = listOf(1, 2, 3)         // immutable
val bag = mutableListOf(1, 2, 3)   // mutable
val doubled = nums.map { it * 2 }.filter { it > 3 }

val capitals = mapOf("BD" to "Dhaka", "JP" to "Tokyo")
```

Classes and inheritance
```kotlin
open class Animal(val name: String) {
    open fun speak() = "$name makes a noise"
}

class Dog(name: String): Animal(name) {
    override fun speak() = "$name barks"
}

interface Clickable { fun click() }
class Button: Clickable { override fun click() { println("clicked") } }

// Data class = value object with equals/hashCode/toString
data class User(val id: Int, val username: String)

// Singleton
object Settings { var dark: Boolean = false }

// Companion object (static-like)
class MathUtil { companion object { fun sq(x: Int) = x * x } }
```

Null safety
```kotlin
var maybeText: String? = null           // nullable type
val length = maybeText?.length ?: 0     // safe call + Elvis
val forced = maybeText!!.length         // not recommended; throws if null

// let for scoped null-check
maybeText?.let { println(it.length) }
```

Useful stdlib
```kotlin
val answer = run { val a = 20; val b = 22; a + b }
val cfg = buildMap {
    put("env", "dev")
    put("retry", 3)
}
```

Testing yourself
- Rewrite small JS utilities in Kotlin (map/filter/reduce, string ops).
- Model a small domain with data classes and inheritance.
