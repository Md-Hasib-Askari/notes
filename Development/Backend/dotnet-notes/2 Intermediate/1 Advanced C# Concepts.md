# Advanced C# Concepts (Intermediate Notes)

## 1. Delegates & Events

* **Delegates**: Type-safe references to methods.

  ```csharp
  delegate void Greet(string name);
  Greet greet = (n) => Console.WriteLine($"Hello {n}");
  greet("Hasib");
  ```
* **Events**: Built on delegates, used for publisher-subscriber pattern.

  ```csharp
  class Alarm
  {
      public event Action OnRing;
      public void Ring() => OnRing?.Invoke();
  }
  ```

---

## 2. Lambda Expressions

* Short, inline functions using `=>` syntax.

  ```csharp
  Func<int, int, int> add = (a, b) => a + b;
  Console.WriteLine(add(5, 3));
  ```
* Often used with LINQ, delegates, and events.

---

## 3. LINQ (Language Integrated Query)

* Query collections in a declarative way.

  ```csharp
  int[] numbers = {1, 2, 3, 4, 5};
  var evens = numbers.Where(n => n % 2 == 0);

  foreach (var e in evens)
      Console.WriteLine(e);
  ```
* Supports filtering (`Where`), projection (`Select`), ordering (`OrderBy`), grouping (`GroupBy`), aggregation (`Sum`, `Average`).

---

## 4. Asynchronous Programming

* **Tasks & async/await**:

  ```csharp
  async Task<int> GetDataAsync()
  {
      await Task.Delay(1000);
      return 42;
  }
  ```

  * Prevents blocking the main thread.
* **Threads**:

  * Lower-level parallel execution using `Thread` class.
  * Modern apps prefer `Task` and `async/await` for simplicity.

---

## 5. Attributes & Reflection

* **Attributes**: Metadata applied to code elements.

  ```csharp
  [Obsolete("Use NewMethod instead")]
  void OldMethod() { }
  ```
* **Reflection**: Inspect and manipulate code at runtime.

  ```csharp
  var type = typeof(string);
  foreach (var method in type.GetMethods())
      Console.WriteLine(method.Name);
  ```
* Useful for frameworks, dependency injection, testing libraries.
