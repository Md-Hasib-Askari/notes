

## 🔴 17. **Using Hugging Face Hub**

> The Hugging Face Hub is the central platform to **upload**, **version**, **share**, and **discover** models, datasets, and spaces.

---

### ✅ 1. **Login to Hugging Face CLI**

```bash
pip install huggingface_hub
huggingface-cli login
```

> You’ll be prompted to paste your token from: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

---

### ✅ 2. **Upload a Model**

You can upload any Transformers-based model using:

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("path/to/your/model")
model.push_to_hub("your-username/your-model-name")
```

Also upload tokenizer:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("path/to/your/tokenizer")
tokenizer.push_to_hub("your-username/your-model-name")
```

---

### ✅ 3. **Upload a Dataset**

```python
from datasets import load_dataset, DatasetDict

dataset = load_dataset("path/to/your/local/dataset/script.py")
dataset.push_to_hub("your-username/your-dataset-name")
```

Or upload directly via the CLI:

```bash
huggingface-cli repo create your-dataset-name --type dataset
git clone https://huggingface.co/datasets/your-username/your-dataset-name
cd your-dataset-name
# add files + commit
git add . && git commit -m "initial commit" && git push
```

---

### ✅ 4. **Model Versioning**

* Each push creates a **snapshot**.
* You can create **tags** for releases using the Hub UI or Git.

---

### ✅ 5. **Model Cards**

Create a `README.md` alongside your model with:

```markdown
---
tags:
- text-classification
- bert
license: apache-2.0
---
# My Custom BERT Model

This model is fine-tuned on...
```

Use `model.push_to_hub(..., commit_message="..."` to auto-commit it.

---

### ✅ 6. **Private vs Public Models**

You can choose visibility when creating a repo:

```bash
huggingface-cli repo create your-model-name --private
```

---

### ✅ 7. **Download from Hub**

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("your-username/your-model-name")
```

---

### 🧠 Exercises

**Task 1:** Create a Hugging Face repo and push a trained model + tokenizer.
**Task 2:** Create a dataset repo and version it after a preprocessing change.
**Task 3:** Write a model card for your fine-tuned model.

