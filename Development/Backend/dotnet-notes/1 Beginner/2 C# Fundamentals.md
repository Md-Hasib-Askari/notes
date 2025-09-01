# C# Fundamentals (Beginner Notes)

## 1. Object-Oriented Programming (OOP)

* **Classes & Objects**: Blueprint (`class`) and its instances (`object`).

  ```csharp
  class Car
  {
      public string Model;
      public void Drive() => Console.WriteLine($"Driving {Model}");
  }

  Car myCar = new Car { Model = "Toyota" };
  myCar.Drive();
  ```
* **Inheritance**: One class derives from another.

  ```csharp
  class ElectricCar : Car
  {
      public int Battery;
  }
  ```
* **Polymorphism**: Ability to override or overload behavior.

  * **Method Overloading**: Same name, different parameters.
  * **Method Overriding**: Redefine in child class using `override`.
* **Encapsulation**: Restrict access to data with access modifiers (`private`, `public`, `protected`) and properties.

---

## 2. Exception Handling

* **Try-Catch**: Handle runtime errors.

  ```csharp
  try
  {
      int x = 10 / 0;
  }
  catch (DivideByZeroException ex)
  {
      Console.WriteLine(ex.Message);
  }
  finally
  {
      Console.WriteLine("Always runs");
  }
  ```
* Create custom exceptions with class inheritance from `Exception`.

---

## 3. Generics

* Allow defining classes, methods, or interfaces with type parameters.

  ```csharp
  class Box<T>
  {
      public T Value;
  }

  Box<int> intBox = new Box<int> { Value = 100 };
  Box<string> strBox = new Box<string> { Value = "Hello" };
  ```
* Benefits: Type safety, reusability, performance.

---

## 4. Collections & LINQ Basics

* **Collections**: More flexible than arrays.

  * `List<T>`: Dynamic resizing.
  * `Dictionary<TKey, TValue>`: Key-value pairs.
  * `HashSet<T>`: Unique items.
* **LINQ (Language Integrated Query)**:

  ```csharp
  List<int> numbers = new List<int> {1, 2, 3, 4, 5};
  var evens = numbers.Where(n => n % 2 == 0).ToList();

  foreach (var e in evens)
      Console.WriteLine(e);
  ```
* LINQ provides SQL-like operations on collections: `Where`, `Select`, `OrderBy`, `GroupBy`, etc.
