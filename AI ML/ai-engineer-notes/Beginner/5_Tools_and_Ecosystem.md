

## 🧰 5. Tools & Ecosystem – Notes

### 📌 Overview:

Mastering the right tools helps streamline the model development, debugging, visualization, and experimentation workflow in AI engineering.

---

### 🧪 5.1 Jupyter Notebooks

#### ✅ Features:

* Interactive coding with immediate output
* Markdown support for documentation
* Great for prototyping and presentations

#### ✅ Common Commands:

```python
!pip install numpy
%timeit my_function()
```

---

### ☁️ 5.2 Google Colab

#### ✅ Benefits:

* Free cloud GPU/TPU access
* Jupyter-like interface
* Collaboration with Google Drive

#### ✅ Use Cases:

* Running heavy models
* Demos and notebooks for teams
* Training deep learning models

---

### 🧱 5.3 Scikit-learn

#### ✅ Purpose:

* A high-level ML library in Python
* Best for classical ML models and preprocessing

#### ✅ Key Modules:

* `sklearn.model_selection` – Train/test split, cross-validation
* `sklearn.preprocessing` – Scaling, encoding
* `sklearn.linear_model`, `tree`, `ensemble` – Algorithms
* `sklearn.metrics` – Model evaluation

#### ✅ Example:

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier().fit(X_train, y_train)
```

---

### 🔁 5.4 MLflow (Experiment Tracking)

#### ✅ Why Use It?

* Track experiments, parameters, and metrics
* Save and version trained models

#### ✅ Components:

* Tracking server
* Models registry
* Projects
* Model deployment (with REST API)

---

### 📚 Other Helpful Tools (at this stage):

| Tool                           | Use Case             |
| ------------------------------ | -------------------- |
| `Matplotlib`, `Seaborn`        | Data visualization   |
| `Pandas Profiling`, `Sweetviz` | Automated EDA        |
| `Joblib`, `Pickle`             | Model serialization  |
| `DVC`                          | Dataset versioning   |
| `Hydra`                        | ML config management |

