## 📘 Topic 1.2: **Calculus for Deep Learning**

### 🔑 Key Concepts

1. **Derivatives**

   * Measures how a function changes as its input changes.
   * Notation:

     ![image](https://github.com/user-attachments/assets/89c34922-868f-439b-b8be-526153baa939)


2. **Chain Rule**

   * Used when functions are composed:

     ![image](https://github.com/user-attachments/assets/fded2a53-64d9-4347-8965-626893aac268)

   * Critical in **backpropagation**.

3. **Partial Derivatives**

   * Derivatives w\.r.t. one variable while keeping others constant.
   * Notation:

     ![image](https://github.com/user-attachments/assets/a97e3652-52e9-49a8-934e-eb5cd3f5056c)

4. **Gradient**

   * Vector of partial derivatives for all input variables.
   * Points in direction of steepest ascent.

     * Used in **gradient descent** to minimize loss.

5. **Jacobian**

   * Matrix of all partial derivatives of vector-valued functions.
   * Important in multivariable networks and transformations.

6. **Hessian**

   * Square matrix of second-order partial derivatives.
   * Used in advanced optimization (Newton’s Method).

---

### 🧠 Intuition for Deep Learning

* **Training** = Adjust weights to minimize loss.
* Derivatives & gradients = Tell how much to tweak weights.
* Chain rule = Allows calculating how changes flow across many layers (backpropagation).

---

### 🧪 Exercises

#### ✅ Conceptual

1. What is the role of the chain rule in backpropagation?
2. How does the gradient help minimize a loss function?
3. What is the difference between partial derivatives and full derivatives?

#### ✅ Math Practice

1. Find the derivative of ![image](https://github.com/user-attachments/assets/6d6278d7-699f-4b83-8937-bc2510d5a418)

2. Use chain rule to find:
   If ![image](https://github.com/user-attachments/assets/33ab6247-361b-4c07-97dd-562e7a543f9c)

#### ✅ Coding (Symbolic & Numerical Derivatives)

```python
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, sin

# Symbolic differentiation
x = symbols('x')
f = x**2 + 3*x + 5
df_dx = diff(f, x)
print("Symbolic derivative of f(x):", df_dx)

# Chain rule example
f2 = sin(2*x**2)
df2_dx = diff(f2, x)
print("df/dx for sin(2x^2):", df2_dx)

# Numerical Gradient
def f(x):
    return x**2 + 3*x + 5

def numerical_gradient(f, x, eps=1e-6):
    return (f(x + eps) - f(x - eps)) / (2 * eps)

x_val = 2.0
grad = numerical_gradient(f, x_val)
print(f"Numerical gradient at x = {x_val}: {grad}")
```
