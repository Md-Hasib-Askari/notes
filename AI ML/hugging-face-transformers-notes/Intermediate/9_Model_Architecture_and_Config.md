

## 🟡 **9. Model Architecture and Config**

### 📘 Notes

Hugging Face Transformers provides high-level classes that wrap pretrained models for different tasks. Each model has:

* A **base architecture** (e.g., BERT, GPT-2, T5)
* A **task-specific head** (e.g., classification, QA, generation)
* A **config object** controlling architecture details

---

### ✅ 1. **Predefined Model Classes**

Each model family has task-specific variants:

| Model | Class                           | Use                       |
| ----- | ------------------------------- | ------------------------- |
| BERT  | `BertForSequenceClassification` | Classification            |
| GPT-2 | `GPT2LMHeadModel`               | Text generation           |
| T5    | `T5ForConditionalGeneration`    | Translation/Summarization |
| BERT  | `BertForQuestionAnswering`      | QA tasks                  |

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
```

---

### ✅ 2. **Accessing Model Configuration**

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("bert-base-uncased")
print(config.hidden_size)  # 768
print(config.num_attention_heads)  # 12
```

---

### ✅ 3. **Modifying Config and Creating a New Model**

You can tweak config to define a fresh model:

```python
config = AutoConfig.from_pretrained("bert-base-uncased", num_labels=3)
new_model = AutoModelForSequenceClassification.from_config(config)
```

This doesn’t load pretrained weights — just structure.

---

### ✅ 4. **Inspecting Model Internals**

```python
print(model.config)             # Full config
print(model.bert.encoder)       # Internal encoder
print(model.classifier)         # Classification head
```

---

### ✅ 5. **Freezing Layers**

To fine-tune only the classification head:

```python
for param in model.bert.parameters():
    param.requires_grad = False
```

---

### 🧠 Exercises

**Task 1**: Load a `BertForSequenceClassification` model with `num_labels=3`, and inspect:

* `config.hidden_size`
* `model.classifier`

**Task 2**: Create a custom `AutoConfig` with a modified number of attention heads and initialize a new model from it.

**Task 3**: Freeze all layers of the model except the classifier head.

