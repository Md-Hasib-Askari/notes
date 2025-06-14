

## 🎯 4.2: **Hyperparameter Optimization**

### 🧠 Goal

Systematically search for the best set of hyperparameters that improve model **accuracy**, **speed**, or **generalization**.

---

## 🧩 1. What Are Hyperparameters?

These are **not learned** during training, but must be set before training.

| Category           | Examples                                           |
| ------------------ | -------------------------------------------------- |
| **Model**          | Layers, neurons, activation functions              |
| **Training**       | Learning rate, batch size, epochs                  |
| **Optimization**   | Optimizer type (Adam, SGD), momentum, weight decay |
| **Regularization** | Dropout rate, L2 penalty                           |

---

## 🔍 2. Techniques for Optimization

### 🔸 **Grid Search**

* Try all combinations in a parameter grid
* **Very slow**, but exhaustive

```python
from sklearn.model_selection import GridSearchCV
```

### 🔸 **Random Search**

* Sample hyperparameters randomly
* **Faster** than grid search, surprisingly effective

```python
from sklearn.model_selection import RandomizedSearchCV
```

### 🔸 **Bayesian Optimization**

* Uses past evaluation results to choose the next promising configuration
* Smarter and faster than grid/random

#### Tools:

* `Optuna`
* `Hyperopt`
* `BayesOpt`

### 🔸 **Evolutionary Algorithms**

* Inspired by natural selection
* Used in **Neuroevolution**, AutoML

### 🔸 **Population-Based Training (PBT)**

* Explore and exploit hyperparameter combinations during training
* Used in **DeepMind’s AlphaStar**

---

## 🔧 3. Libraries for Automation

| Library                     | Features                                          |
| --------------------------- | ------------------------------------------------- |
| **Optuna**                  | Lightweight, fast, supports pruning               |
| **Ray Tune**                | Scalable, distributed, integrates with PyTorch/TF |
| **Keras Tuner**             | Hyperparameter search for TensorFlow/Keras        |
| **Weights & Biases Sweeps** | Cloud-based tuning + visualization                |

---

## ⚙️ 4. Real-World Workflow Example

```python
import optuna

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    model = build_model(lr, dropout)
    accuracy = train_and_eval(model)
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

---

## 🧪 Exercises

### ✅ Theory

* Why is random search often better than grid search?
* What is pruning in hyperparameter tuning?

### ✅ Practical

* Tune a CNN’s learning rate and dropout using Optuna
* Run Ray Tune to scale tuning across multiple CPUs/GPUs
* Visualize the best trials with `optuna.visualization`

