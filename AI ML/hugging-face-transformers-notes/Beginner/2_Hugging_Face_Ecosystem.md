

## 🔹 **2. Hugging Face Ecosystem Overview**

### 📘 Notes

Hugging Face offers a modular ecosystem designed to support **end-to-end NLP workflows**.

---

### 🧱 Core Libraries

| Library           | Purpose                                            |
| ----------------- | -------------------------------------------------- |
| `transformers`    | Pretrained models & tokenizers                     |
| `datasets`        | NLP datasets for training/evaluation               |
| `tokenizers`      | Fast tokenization (Rust backend)                   |
| `evaluate`        | Easy metric computation (accuracy, F1, etc.)       |
| `accelerate`      | Lightweight training across CPUs, GPUs, TPUs       |
| `huggingface_hub` | Access to models, datasets, and Spaces from HF Hub |

---

### 🛠️ Key Services

| Tool                 | Description                                        |
| -------------------- | -------------------------------------------------- |
| **Hugging Face Hub** | Online platform to share/download models, datasets |
| **Spaces**           | Host apps using Gradio or Streamlit                |
| **Model Cards**      | Metadata for models (usage, training, limitations) |
| **Inference API**    | Pay-as-you-go hosted API for running models        |

---

### 🔧 Transformers Library Overview

```bash
pip install transformers
```

* Auto Classes:

  * `AutoModel`, `AutoTokenizer`, `AutoModelForSequenceClassification`, etc.
* Pipelines for quick usage:

  ```python
  from transformers import pipeline
  classifier = pipeline("sentiment-analysis")
  print(classifier("I love Hugging Face!"))
  ```

---

### 💡 When to Use What

| Task                   | Tool                    |
| ---------------------- | ----------------------- |
| Quickly test a model   | `pipeline`              |
| Custom model training  | `Trainer`, `accelerate` |
| Custom dataset         | `datasets`              |
| Evaluation             | `evaluate`              |
| Hosting demo           | `Spaces`                |
| Upload models/datasets | `huggingface_hub`       |

---

### 🧠 Exercises

**Task 1**: List which Hugging Face tools you’d use for the following:

* Load a sentiment classification model.
* Train a BERT model on your own data.
* Host an interactive NLP app online.
* Track F1-score while evaluating.

**Task 2**: Go to [https://huggingface.co/models](https://huggingface.co/models)

* Choose any model you like.
* Note its architecture (e.g., BERT, T5), size, and what it’s used for.

**Task 3**: Install the required libraries:

```bash
pip install transformers datasets tokenizers accelerate evaluate
```
