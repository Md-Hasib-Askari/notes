# Basic Programming Concepts (Beginner Notes)

## 1. Variables, Data Types, Operators

* **Variables**: Containers for storing data values. Declared with a name and type in C#.

  ```csharp
  int age = 25;
  string name = "Hasib";
  ```
* **Data Types**:

  * **Value types**: int, float, double, bool, char.
  * **Reference types**: string, arrays, objects.
  * **Nullable types**: Allow null assignment, e.g., `int? x = null;`
* **Operators**:

  * **Arithmetic**: +, -, \*, /, %
  * **Comparison**: ==, !=, <, >, <=, >=
  * **Logical**: &&, ||, !
  * **Assignment**: =, +=, -=, \*=, /=

---

## 2. Control Flow (if/else, switch, loops)

* **If/Else**:

  ```csharp
  if (age > 18)
      Console.WriteLine("Adult");
  else
      Console.WriteLine("Minor");
  ```
* **Switch**:

  ```csharp
  switch(day)
  {
      case 1: Console.WriteLine("Monday"); break;
      case 2: Console.WriteLine("Tuesday"); break;
      default: Console.WriteLine("Other day"); break;
  }
  ```
* **Loops**:

  * **For**: Execute block fixed number of times.
  * **While**: Repeat until condition is false.
  * **Do-While**: Execute at least once, then check condition.
  * **Foreach**: Iterate through collections.

---

## 3. Functions & Parameters

* **Functions (Methods)**: Blocks of reusable code.

  ```csharp
  int Add(int a, int b)
  {
      return a + b;
  }
  ```
* **Parameters**:

  * **By Value**: Default, copies data.
  * **By Reference**: `ref` or `out` keyword.
  * **Optional Parameters**: Provide default values.

---

## 4. Arrays, Lists, Strings

* **Arrays**: Fixed-size collection.

  ```csharp
  int[] numbers = {1, 2, 3};
  ```
* **Lists**: Dynamic collection.

  ```csharp
  List<int> numbers = new List<int>() {1, 2, 3};
  numbers.Add(4);
  ```
* **Strings**: Immutable sequence of characters.

  ```csharp
  string message = "Hello";
  Console.WriteLine(message.Length);
  ```

  * Useful methods: `Substring()`, `ToUpper()`, `ToLower()`, `Contains()`.
