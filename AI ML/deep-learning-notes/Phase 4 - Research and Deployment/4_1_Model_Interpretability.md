

## 🧪 4.1: **Model Interpretability**

### 🎯 Goal

Make deep learning models more understandable and **transparent** — especially important in sensitive domains (healthcare, finance, etc.).

---

## 🔍 Why Interpretability Matters

| Purpose            | Use Case Example                               |
| ------------------ | ---------------------------------------------- |
| **Debugging**      | Identify why the model fails on edge cases     |
| **Trust**          | Show end-users how decisions were made         |
| **Compliance**     | Required in regulated industries (GDPR, HIPAA) |
| **Bias Detection** | Detect unethical or unwanted behavior          |

---

## 🛠️ 1. Feature Importance Methods

### 🔹 **SHAP (SHapley Additive exPlanations)**

* Based on game theory (Shapley values)
* Measures each feature’s contribution to predictions
* **Works with any model**
* Tools: `shap` Python package

**Example Plot:**

* SHAP summary plot showing global feature influence
* SHAP force plot showing local explanation for one prediction

### 🔹 **LIME (Local Interpretable Model-Agnostic Explanations)**

* Perturbs input data locally to understand the effect of each feature
* Trains a simple, interpretable model (like linear regression) locally

**Use Case:** Explain why a classifier predicted "spam" for a specific email.

---

## 🧠 2. Model-Specific Interpretability

| Model Type         | Techniques                           |
| ------------------ | ------------------------------------ |
| **CNNs**           | Grad-CAM, Guided Backpropagation     |
| **Tabular Models** | Feature importance, SHAP, LIME       |
| **Transformers**   | Attention visualization              |
| **RNNs**           | Hidden state analysis, saliency maps |

---

## 🧪 3. Exercises

### ✅ Theory

* What’s the difference between SHAP and LIME?
* When would Grad-CAM be more useful than SHAP?

### ✅ Practice

* Use `shap.Explainer` on a trained XGBoost or PyTorch model
* Visualize Grad-CAM on a CNN trained on CIFAR-10
* Apply LIME on a sentiment analysis classifier

