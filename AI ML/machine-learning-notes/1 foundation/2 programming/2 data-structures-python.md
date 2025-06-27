# Data Structures in Python

## Overview
Python data structures are fundamental building blocks for storing, organizing, and manipulating data efficiently. Understanding these structures is crucial for machine learning applications where data handling and algorithm implementation are key.

## Built-in Data Structures

### 1. Lists - Mutable Sequences
**Characteristics**: Ordered, mutable, allow duplicates
```python
# Creating and manipulating lists
numbers = [1, 2, 3, 4, 5]
mixed_list = [1, "hello", 3.14, True]

# Common operations
numbers.append(6)           # Add to end
numbers.insert(0, 0)        # Insert at index
numbers.remove(3)           # Remove first occurrence
popped = numbers.pop()      # Remove and return last item

# List comprehensions (powerful for data processing)
squares = [x**2 for x in range(10)]
filtered = [x for x in numbers if x > 2]

# Slicing
subset = numbers[1:4]       # Elements 1 to 3
reversed_list = numbers[::-1]
```

**ML Applications**:
- Feature vectors
- Training data batches
- Model predictions storage

### 2. Tuples - Immutable Sequences
**Characteristics**: Ordered, immutable, allow duplicates
```python
# Creating tuples
coordinates = (10, 20)
rgb_color = (255, 128, 0)
single_item = (42,)  # Note the comma for single item

# Unpacking
x, y = coordinates
r, g, b = rgb_color

# Named tuples for better readability
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p1 = Point(10, 20)
print(p1.x, p1.y)
```

**ML Applications**:
- Immutable configurations
- Data point coordinates
- Model architecture parameters

### 3. Dictionaries - Key-Value Mappings
**Characteristics**: Unordered (Python 3.7+ maintains insertion order), mutable, unique keys
```python
# Creating dictionaries
student = {
    'name': 'Alice',
    'age': 25,
    'grades': [85, 92, 78]
}

# Accessing and modifying
print(student['name'])
student['major'] = 'Computer Science'
student.setdefault('gpa', 0.0)

# Dictionary methods
keys = student.keys()
values = student.values()
items = student.items()

# Dictionary comprehensions
squares_dict = {x: x**2 for x in range(5)}
filtered_dict = {k: v for k, v in student.items() if isinstance(v, str)}

# Default dictionaries
from collections import defaultdict
word_count = defaultdict(int)
for word in ['apple', 'banana', 'apple']:
    word_count[word] += 1
```

**ML Applications**:
- Feature mappings
- Model hyperparameters
- Data preprocessing configurations
- Vocabulary mappings in NLP

### 4. Sets - Unique Collections
**Characteristics**: Unordered, mutable, unique elements only
```python
# Creating sets
unique_numbers = {1, 2, 3, 4, 5}
empty_set = set()  # Not {} which creates empty dict

# Set operations
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

union = set1 | set2         # {1, 2, 3, 4, 5, 6}
intersection = set1 & set2   # {3, 4}
difference = set1 - set2     # {1, 2}
symmetric_diff = set1 ^ set2 # {1, 2, 5, 6}

# Adding and removing
set1.add(7)
set1.discard(1)  # Won't raise error if not found
set1.remove(2)   # Will raise error if not found
```

**ML Applications**:
- Feature selection (unique features)
- Data deduplication
- Vocabulary creation
- Label sets in classification

## Advanced Data Structures

### 1. Collections Module

#### Counter - Counting Elements
```python
from collections import Counter

# Count elements in a sequence
text = "hello world"
char_count = Counter(text)
print(char_count)  # Counter({'l': 3, 'o': 2, 'h': 1, ...})

# Most common elements
word_list = ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']
word_counter = Counter(word_list)
print(word_counter.most_common(2))  # [('apple', 3), ('banana', 2)]
```

#### Deque - Double-ended Queue
```python
from collections import deque

# Efficient operations at both ends
queue = deque([1, 2, 3])
queue.appendleft(0)    # Add to left
queue.append(4)        # Add to right
left_item = queue.popleft()   # Remove from left
right_item = queue.pop()      # Remove from right

# Fixed-size deque (circular buffer)
buffer = deque(maxlen=3)
for i in range(5):
    buffer.append(i)
print(buffer)  # deque([2, 3, 4], maxlen=3)
```

#### OrderedDict - Ordered Dictionary
```python
from collections import OrderedDict

# Maintains insertion order (less needed in Python 3.7+)
ordered = OrderedDict()
ordered['first'] = 1
ordered['second'] = 2
ordered['third'] = 3

# Move to end
ordered.move_to_end('first')
print(list(ordered.keys()))  # ['second', 'third', 'first']
```

### 2. Custom Classes for ML

#### Data Point Class
```python
class DataPoint:
    def __init__(self, features, label=None):
        self.features = features
        self.label = label
        self.predictions = []
    
    def add_prediction(self, model_name, prediction):
        self.predictions.append((model_name, prediction))
    
    def __repr__(self):
        return f"DataPoint(features={self.features}, label={self.label})"

# Usage
point = DataPoint([1.5, 2.3, 0.8], label='positive')
point.add_prediction('model_1', 'positive')
```

#### Simple Dataset Class
```python
class Dataset:
    def __init__(self):
        self.data = []
        self.labels = []
    
    def add_sample(self, features, label):
        self.data.append(features)
        self.labels.append(label)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def batch_iterator(self, batch_size):
        for i in range(0, len(self.data), batch_size):
            yield (self.data[i:i+batch_size], 
                   self.labels[i:i+batch_size])

# Usage
dataset = Dataset()
dataset.add_sample([1, 2, 3], 'A')
dataset.add_sample([4, 5, 6], 'B')

for batch_data, batch_labels in dataset.batch_iterator(2):
    print(f"Batch: {batch_data}, Labels: {batch_labels}")
```

### 3. Linked Lists Implementation

```python
class ListNode:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        new_node = ListNode(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def prepend(self, data):
        new_node = ListNode(data)
        new_node.next = self.head
        self.head = new_node
    
    def delete(self, data):
        if not self.head:
            return
        
        if self.head.data == data:
            self.head = self.head.next
            return
        
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                return
            current = current.next
    
    def display(self):
        elements = []
        current = self.head
        while current:
            elements.append(current.data)
            current = current.next
        return elements

# Usage
ll = LinkedList()
ll.append(1)
ll.append(2)
ll.prepend(0)
print(ll.display())  # [0, 1, 2]
```

### 4. Stacks and Queues

#### Stack Implementation
```python
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("Stack is empty")
    
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("Stack is empty")
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

# Usage in algorithm implementation
def evaluate_postfix(expression):
    stack = Stack()
    operators = {'+', '-', '*', '/'}
    
    for token in expression.split():
        if token not in operators:
            stack.push(float(token))
        else:
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.push(a + b)
            elif token == '-':
                stack.push(a - b)
            elif token == '*':
                stack.push(a * b)
            elif token == '/':
                stack.push(a / b)
    
    return stack.pop()
```

#### Queue Implementation
```python
class Queue:
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):
        self.items.append(item)
    
    def dequeue(self):
        if not self.is_empty():
            return self.items.popleft()
        raise IndexError("Queue is empty")
    
    def front(self):
        if not self.is_empty():
            return self.items[0]
        raise IndexError("Queue is empty")
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

# Usage in breadth-first search
def bfs_graph(graph, start):
    visited = set()
    queue = Queue()
    queue.enqueue(start)
    visited.add(start)
    result = []
    
    while not queue.is_empty():
        vertex = queue.dequeue()
        result.append(vertex)
        
        for neighbor in graph.get(vertex, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.enqueue(neighbor)
    
    return result
```

## Memory Optimization Techniques

### 1. Generators for Large Datasets
```python
def data_generator(filename):
    """Generator for reading large files line by line"""
    with open(filename, 'r') as file:
        for line in file:
            yield process_line(line)

def process_line(line):
    # Process each line
    return line.strip().split(',')

# Usage - memory efficient for large files
for data_point in data_generator('large_dataset.csv'):
    # Process one point at a time
    pass
```

### 2. Array Module for Numeric Data
```python
import array

# More memory efficient than lists for numeric data
numbers_list = [1, 2, 3, 4, 5]  # Each int is a Python object
numbers_array = array.array('i', [1, 2, 3, 4, 5])  # Compact storage

print(f"List size: {numbers_list.__sizeof__()}")
print(f"Array size: {numbers_array.__sizeof__()}")
```

### 3. Slots for Classes
```python
class OptimizedDataPoint:
    __slots__ = ['features', 'label', 'prediction']
    
    def __init__(self, features, label):
        self.features = features
        self.label = label
        self.prediction = None

# Uses less memory than regular classes
```

## Performance Comparison

### Time Complexity Summary
| Operation | List | Tuple | Dict | Set |
|-----------|------|-------|------|-----|
| Access | O(1) | O(1) | O(1) | - |
| Search | O(n) | O(n) | O(1) | O(1) |
| Insert | O(n) | - | O(1) | O(1) |
| Delete | O(n) | - | O(1) | O(1) |

### Choosing the Right Structure
```python
# Performance testing example
import time

def performance_test():
    # Test list vs set for membership testing
    data_list = list(range(10000))
    data_set = set(range(10000))
    
    target = 9999
    
    # List search
    start = time.time()
    result = target in data_list
    list_time = time.time() - start
    
    # Set search
    start = time.time()
    result = target in data_set
    set_time = time.time() - start
    
    print(f"List search: {list_time:.6f}s")
    print(f"Set search: {set_time:.6f}s")
    print(f"Set is {list_time/set_time:.1f}x faster")

performance_test()
```

## Applications in Machine Learning

### 1. Data Storage and Manipulation
```python
# Storing training data
training_data = {
    'features': [[1, 2], [3, 4], [5, 6]],
    'labels': ['A', 'B', 'A'],
    'metadata': {'source': 'experiment_1', 'date': '2024-01-01'}
}

# Feature engineering with sets
def extract_unique_values(data, column):
    return set(row[column] for row in data)

unique_categories = extract_unique_values(training_data['features'], 0)
```

### 2. Algorithm Implementation
```python
# K-Nearest Neighbors using heaps
import heapq

def knn_predict(train_data, train_labels, test_point, k=3):
    distances = []
    
    for i, train_point in enumerate(train_data):
        dist = sum((a - b)**2 for a, b in zip(test_point, train_point))**0.5
        heapq.heappush(distances, (dist, train_labels[i]))
    
    # Get k nearest neighbors
    neighbors = [heapq.heappop(distances)[1] for _ in range(k)]
    
    # Return most common label
    return Counter(neighbors).most_common(1)[0][0]
```

### 3. Caching and Memoization
```python
from functools import lru_cache

class ModelCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
    
    def get_prediction(self, features_tuple):
        if features_tuple in self.cache:
            return self.cache[features_tuple]
        return None
    
    def store_prediction(self, features_tuple, prediction):
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[features_tuple] = prediction

# Usage
cache = ModelCache()
features = (1.5, 2.3, 0.8)
prediction = cache.get_prediction(features)
if prediction is None:
    prediction = expensive_model_prediction(features)
    cache.store_prediction(features, prediction)
```

## Best Practices

### 1. Choose Appropriate Data Structure
- **Lists**: When order matters and you need mutability
- **Tuples**: For immutable sequences and function returns
- **Dictionaries**: For key-value mappings and fast lookups
- **Sets**: For unique elements and set operations

### 2. Memory Considerations
- Use generators for large datasets
- Consider array module for numeric data
- Use __slots__ for memory-critical classes

### 3. Performance Optimization
- Use built-in functions when possible
- Prefer dict/set lookups over list searches
- Use list comprehensions over loops
- Consider collections module for specialized needs

## Learning Objectives
- [ ] Master Python built-in data structures
- [ ] Implement custom data structures
- [ ] Optimize data access patterns
- [ ] Choose appropriate structures for ML tasks
- [ ] Understand time/space complexity
- [ ] Apply memory optimization techniques
- [ ] Use collections module effectively
- [ ] Implement algorithms using data structures

## Quick Reference

**Lists**: `[1, 2, 3]`, append(), insert(), remove(), pop()
**Tuples**: `(1, 2, 3)`, immutable, tuple unpacking
**Dicts**: `{'key': 'value'}`, get(), setdefault(), keys(), values()
**Sets**: `{1, 2, 3}`, add(), discard(), union |, intersection &
**Collections**: Counter, deque, defaultdict, OrderedDict

## Next Steps
- Practice implementing algorithms with different data structures
- Learn about NumPy arrays for numerical computing
- Explore pandas DataFrames for data analysis
- Study advanced data structures (trees, graphs) for specific ML algorithms