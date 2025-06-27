# Python Fundamentals

## Python Basics

### Syntax and Structure
- Indentation-based structure
- Comments and docstrings
- Variable naming conventions
- PEP 8 style guide

### Data Types
- **Numeric**: int, float, complex
- **Sequence**: str, list, tuple
- **Mapping**: dict
- **Set**: set, frozenset
- **Boolean**: bool
- **None**: NoneType

### Variables and Assignment
```python
# Variable assignment
name = "Alice"
age = 30
height = 5.6
is_student = True

# Multiple assignment
x, y, z = 1, 2, 3
a = b = c = 0
```

## Control Structures

### Conditional Statements
```python
# if-elif-else
if score >= 90:
    grade = 'A'
elif score >= 80:
    grade = 'B'
elif score >= 70:
    grade = 'C'
else:
    grade = 'F'

# Ternary operator
result = "Pass" if score >= 60 else "Fail"
```

### Loops
```python
# for loop
for i in range(10):
    print(i)

# while loop
count = 0
while count < 5:
    print(count)
    count += 1

# Loop with else
for i in range(5):
    if i == 3:
        break
else:
    print("Loop completed without break")
```

### List Comprehensions
```python
# Basic list comprehension
squares = [x**2 for x in range(10)]

# With condition
even_squares = [x**2 for x in range(10) if x % 2 == 0]

# Nested comprehension
matrix = [[i*j for j in range(3)] for i in range(3)]

# Dictionary comprehension
word_lengths = {word: len(word) for word in ['apple', 'banana', 'cherry']}
```

## Functions

### Function Definition
```python
def greet(name, greeting="Hello"):
    """
    Greet a person with a custom message.
    
    Args:
        name (str): Person's name
        greeting (str): Greeting message (default: "Hello")
    
    Returns:
        str: Formatted greeting
    """
    return f"{greeting}, {name}!"

# Function call
message = greet("Alice")
custom_message = greet("Bob", "Hi")
```

### Advanced Function Features
```python
# *args and **kwargs
def flexible_function(*args, **kwargs):
    print(f"Args: {args}")
    print(f"Kwargs: {kwargs}")

# Lambda functions
square = lambda x: x**2
numbers = [1, 2, 3, 4, 5]
squared = list(map(square, numbers))

# Higher-order functions
def apply_operation(numbers, operation):
    return [operation(x) for x in numbers]

result = apply_operation([1, 2, 3], lambda x: x * 2)
```

## Object-Oriented Programming

### Classes and Objects
```python
class Student:
    # Class variable
    school = "Data Science Academy"
    
    def __init__(self, name, age, major):
        # Instance variables
        self.name = name
        self.age = age
        self.major = major
        self.grades = []
    
    def add_grade(self, grade):
        """Add a grade to the student's record."""
        self.grades.append(grade)
    
    def get_average(self):
        """Calculate average grade."""
        if not self.grades:
            return 0
        return sum(self.grades) / len(self.grades)
    
    def __str__(self):
        return f"{self.name}, {self.age} years old, majoring in {self.major}"
    
    def __repr__(self):
        return f"Student('{self.name}', {self.age}, '{self.major}')"

# Create objects
alice = Student("Alice", 20, "Computer Science")
alice.add_grade(85)
alice.add_grade(92)
print(f"Average grade: {alice.get_average()}")
```

### Inheritance
```python
class GraduateStudent(Student):
    def __init__(self, name, age, major, research_area):
        super().__init__(name, age, major)
        self.research_area = research_area
        self.publications = []
    
    def add_publication(self, title):
        """Add a publication to the student's record."""
        self.publications.append(title)
    
    def __str__(self):
        return f"{super().__str__()}, Research: {self.research_area}"

# Multiple inheritance
class TeachingAssistant(GraduateStudent):
    def __init__(self, name, age, major, research_area, course):
        super().__init__(name, age, major, research_area)
        self.course = course
    
    def grade_assignment(self, assignment, grade):
        """Grade a student assignment."""
        return f"Graded {assignment}: {grade}"
```

## Error Handling

### Exception Handling
```python
def safe_divide(a, b):
    """Safely divide two numbers."""
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("Error: Cannot divide by zero")
        return None
    except TypeError:
        print("Error: Invalid input types")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
    finally:
        print("Division operation completed")

# Custom exceptions
class InvalidGradeError(Exception):
    """Raised when an invalid grade is provided."""
    pass

def validate_grade(grade):
    if not 0 <= grade <= 100:
        raise InvalidGradeError(f"Grade {grade} is not between 0 and 100")
    return grade
```

## File I/O and Data Handling

### File Operations
```python
# Reading files
def read_data(filename):
    """Read data from a file."""
    try:
        with open(filename, 'r') as file:
            data = file.read()
        return data
    except FileNotFoundError:
        print(f"File {filename} not found")
        return None

# Writing files
def write_data(filename, data):
    """Write data to a file."""
    with open(filename, 'w') as file:
        file.write(data)

# CSV handling (basic)
def read_csv_simple(filename):
    """Simple CSV reader."""
    data = []
    with open(filename, 'r') as file:
        for line in file:
            row = line.strip().split(',')
            data.append(row)
    return data
```

## Modules and Packages

### Creating Modules
```python
# math_utils.py
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# Using modules
import math_utils
from math_utils import add, multiply

result1 = math_utils.add(5, 3)
result2 = multiply(4, 6)
```

### Package Structure
```
my_package/
    __init__.py
    module1.py
    module2.py
    subpackage/
        __init__.py
        submodule.py
```

## Useful Built-in Functions

### Essential Functions
```python
# Type conversion
int_val = int("42")
float_val = float("3.14")
str_val = str(123)

# Sequence operations
numbers = [1, 5, 3, 9, 2]
total = sum(numbers)
maximum = max(numbers)
minimum = min(numbers)
length = len(numbers)

# Functional programming
def is_even(x):
    return x % 2 == 0

evens = list(filter(is_even, numbers))
doubled = list(map(lambda x: x * 2, numbers))

# Zip and enumerate
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35]
combined = list(zip(names, ages))

for index, name in enumerate(names):
    print(f"{index}: {name}")
```

## Best Practices

### Code Style
- Follow PEP 8 conventions
- Use meaningful variable names
- Write clear docstrings
- Keep functions small and focused

### Performance Tips
- Use list comprehensions for simple operations
- Avoid global variables
- Use appropriate data structures
- Profile code when necessary

### Debugging
```python
# Print debugging
def debug_function(x):
    print(f"Input: {x}")
    result = x * 2
    print(f"Result: {result}")
    return result

# Using assert for testing
def validate_input(x):
    assert isinstance(x, (int, float)), "Input must be a number"
    assert x >= 0, "Input must be non-negative"
    return x
```

## Learning Exercises

### Basic Exercises
1. Create a calculator class with basic operations
2. Implement a simple banking system with accounts
3. Build a text analyzer (word count, character frequency)
4. Create a basic inventory management system

### Intermediate Exercises
1. Implement data structures (stack, queue, linked list)
2. Create a file organizer script
3. Build a simple web scraper
4. Design a basic game (tic-tac-toe, guessing game)

## Common Patterns for ML

### Data Processing Pattern
```python
def process_data(data, transformations):
    """Apply a series of transformations to data."""
    result = data
    for transform in transformations:
        result = transform(result)
    return result

# Usage
transformations = [
    lambda x: x.lower(),
    lambda x: x.strip(),
    lambda x: x.replace(' ', '_')
]

processed = process_data("  Hello World  ", transformations)
```

### Iterator Pattern
```python
class DataBatch:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
    
    def __iter__(self):
        self.index = 0
        return self
    
    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        
        batch = self.data[self.index:self.index + self.batch_size]
        self.index += self.batch_size
        return batch

# Usage
data = list(range(100))
batch_iterator = DataBatch(data, 10)

for batch in batch_iterator:
    print(f"Processing batch: {batch[:3]}...")  # Show first 3 items
```

## Learning Objectives
- [ ] Master Python syntax and data types
- [ ] Understand control structures and functions
- [ ] Apply object-oriented programming concepts
- [ ] Handle errors and exceptions properly
- [ ] Work with files and modules
- [ ] Write clean, readable code

## Next Steps
- Learn NumPy for numerical computing
- Master Pandas for data manipulation
- Explore advanced Python features
- Practice with real datasets
