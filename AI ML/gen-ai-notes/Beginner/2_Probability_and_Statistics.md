

## 🟢 **2. Math & Programming Basics – Part 2: Probability & Statistics**

### 🔍 **Key Concepts**

* **Random Variables** (Discrete vs. Continuous)
* **Probability Distributions** (Uniform, Normal, Bernoulli)
* **Mean, Median, Mode, Variance, Standard Deviation**
* **Bayes’ Theorem**:

  $$
  P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
  $$
* **Expectation & Entropy**

---

### ✍️ **Quick Notes**

* **Mean**: Average value
* **Variance**: Spread of the data
* **Standard Deviation (σ)**: Square root of variance
* **Normal Distribution**: Bell curve; essential in generative models and noise modeling
* **Bayes’ Theorem**: Crucial in probabilistic generative models like VAEs

---

### 🧠 **Why This Matters in Generative AI**

* Generative models often **sample** from distributions (like normal).
* Variational Autoencoders and Diffusion models rely heavily on **probabilistic reasoning** and **Bayesian inference**.
* **Entropy** quantifies uncertainty, important in GANs and other models during training.

---

### 🧪 **Practice Exercises**

1. Simulate 1000 samples from a normal distribution in Python, then plot the histogram.
2. Compute the **mean**, **variance**, and **standard deviation** for a list of values manually and with NumPy.
3. Apply Bayes' Theorem:

   > Suppose 1% of emails are spam. A spam filter flags 99% of spam and 2% of non-spam. What’s the probability an email is actually spam given it was flagged?

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate normal distribution
samples = np.random.normal(loc=0, scale=1, size=1000)
plt.hist(samples, bins=30, density=True)
plt.title("Normal Distribution")
plt.show()

# Mean, variance, std
print("Mean:", np.mean(samples))
print("Variance:", np.var(samples))
print("Std Dev:", np.std(samples))
```
