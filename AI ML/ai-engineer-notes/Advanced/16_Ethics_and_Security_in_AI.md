
## 🛡️ 16. Ethics & Security in AI – Notes

### 📌 Overview:

As AI systems influence critical decisions in society, **ethics and security** are not optional. Responsible AI design ensures fairness, transparency, and protection from harm or misuse.

---

### ⚖️ 16.1 Bias and Fairness in ML

#### ✅ Common Bias Types:

* **Historical Bias**: Data reflects historical inequalities.
* **Sampling Bias**: Training data doesn’t represent the full population.
* **Measurement Bias**: Incorrect or inconsistent data labeling.

#### ✅ Mitigation Techniques:

* **Preprocessing**: Reweight samples, remove biased features.
* **In-Processing**: Modify training (e.g., adversarial debiasing).
* **Postprocessing**: Calibrate outputs (e.g., equalized odds).

#### ✅ Tools:

* **IBM AI Fairness 360**
* **Fairlearn** (Microsoft)

---

### 🔍 16.2 Model Explainability (SHAP, LIME)

#### ✅ Why Explainability Matters:

* Build trust
* Debug models
* Ensure fairness and compliance

#### ✅ Techniques:

* **SHAP** (SHapley Additive exPlanations): Global + local importance
* **LIME** (Local Interpretable Model-agnostic Explanations): Local approximations

#### ✅ Tools:

* `shap`, `lime`, `captum` (for PyTorch)

---

### ⚔️ 16.3 Adversarial Attacks & Defenses

#### ✅ Common Attack Types:

* **FGSM (Fast Gradient Sign Method)**: Slight perturbations fool model
* **PGD (Projected Gradient Descent)**: Iterative attack
* **Data Poisoning**: Inject malicious samples during training

#### ✅ Defenses:

* **Adversarial Training**
* **Gradient Masking**
* **Input Sanitization**

#### ✅ Libraries:

* `foolbox`, `cleverhans`, `robustness`

---

### 🔐 16.4 Data Privacy (Differential Privacy, Federated Learning)

#### ✅ Differential Privacy:

* Injects noise to data or gradients to **protect individual data points**
* Ensures outputs are statistically similar whether a record is in the dataset or not

#### ✅ Federated Learning:

* Trains models across distributed devices **without centralizing data**
* Used by Google for Gboard, health data, etc.

#### ✅ Tools:

* **PySyft** (OpenMined)
* **TensorFlow Federated**

---

### ✅ Summary Table:

| Concern            | Key Technique                           | Tools               |
| ------------------ | --------------------------------------- | ------------------- |
| Bias & Fairness    | Pre/post-processing, adversarial debias | Fairlearn, AIF360   |
| Explainability     | SHAP, LIME                              | shap, lime, captum  |
| Adversarial Safety | FGSM, PGD defense, robust training      | foolbox, robustness |
| Privacy            | DP, Federated Learning                  | PySyft, TFF         |

