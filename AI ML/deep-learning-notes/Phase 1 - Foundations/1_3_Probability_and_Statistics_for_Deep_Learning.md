## 📘 Topic 1.3: **Probability and Statistics for Deep Learning**

### 🔑 Key Concepts

---

### 📊 1. **Probability Basics**

* **Random Variable**: A variable whose value is the outcome of a random phenomenon.
* **Probability Distribution**: Describes how probabilities are distributed over values.

  * Discrete: Bernoulli, Binomial
  * Continuous: Gaussian (Normal), Uniform

---

### 📈 2. **Key Distributions**

#### ✅ **Bernoulli Distribution**

* Outcome: 0 or 1  
![image](https://github.com/user-attachments/assets/f8461585-0eb0-492b-9a4e-87881438326b)


#### ✅ **Binomial Distribution**

* Number of successes in $n$ independent trials  
![image](https://github.com/user-attachments/assets/7d821bac-3d1c-4bce-a930-f656f5bea81f)


#### ✅ **Normal (Gaussian) Distribution**

* Continuous, bell-shaped  
![image](https://github.com/user-attachments/assets/e5020e1c-09f7-4053-bc62-84a16584daee)  
![image](https://github.com/user-attachments/assets/3e21de55-482a-47bf-8359-99fba98ecce9)


---

### 📐 3. **Descriptive Statistics**

* **Mean**: Average
* **Median**: Middle value
* **Mode**: Most frequent value
* **Variance**: Spread of the data
  ![image](https://github.com/user-attachments/assets/47f55bba-9a5a-483f-befa-a7c32c2e4f54)

* **Standard Deviation**: Square root of variance

---

### 📌 4. **Bayes’ Theorem**

<p align="center">
  <img src="https://github.com/user-attachments/assets/beb5f753-7a70-4da2-8f1b-535e8bd42f95" alt="Bayes’ Theorem"/>
</p>

* Used in **generative models**, **Naive Bayes**, **Bayesian neural nets**

---

### 🧠 Intuition for Deep Learning

* **Uncertainty**: Models predict probabilities (not certainties)
* **Distributions**: Used in loss functions (cross-entropy, KL divergence)
* **Bayesian Thinking**: Used in probabilistic models and variational inference

---

### 🧪 Exercises

#### ✅ Conceptual

1. What's the difference between a PDF and PMF?
2. Why is the Gaussian distribution important in deep learning?
3. Explain how Bayes' Theorem could be used in classification.

---

#### ✅ Coding (Python + NumPy/Matplotlib)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, bernoulli, binom

# Gaussian distribution plot
x = np.linspace(-5, 5, 100)
mu, sigma = 0, 1
pdf = norm.pdf(x, mu, sigma)
plt.plot(x, pdf)
plt.title('Standard Normal Distribution')
plt.show()

# Sampling from distributions
ber_samples = bernoulli.rvs(p=0.3, size=1000)
binom_samples = binom.rvs(n=10, p=0.5, size=1000)
normal_samples = np.random.normal(loc=0, scale=1, size=1000)

print("Bernoulli samples mean:", np.mean(ber_samples))
print("Binomial samples std dev:", np.std(binom_samples))
print("Normal samples variance:", np.var(normal_samples))
```
