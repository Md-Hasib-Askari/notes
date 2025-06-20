

## 🔴 13. **Accelerate & Distributed Training**

### 📘 Notes

[`Accelerate`](https://huggingface.co/docs/accelerate/) is Hugging Face’s library that simplifies **distributed training**, **multi-GPU**, and **mixed precision** setups.

---

### ✅ 1. **Why Use `accelerate`?**

* Train on **multiple GPUs**, **TPUs**, or **multi-node** environments easily.
* Use **mixed precision (fp16/bf16)** to speed up training and reduce memory.
* Keep your code **device-agnostic** and simple.

---

### ✅ 2. **Quick Setup**

```bash
pip install accelerate
accelerate config  # interactive setup
```

This generates a `default_config.yaml` for your environment.

---

### ✅ 3. **Minimal Accelerate Training Loop**

```python
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch

accelerator = Accelerator()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load and tokenize dataset
dataset = load_dataset("imdb", split="train[:2000]")
dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length"), batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask"])

# DataLoader
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model & optimizer
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Prepare with Accelerator
model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

model.train()
for batch in loader:
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

---

### ✅ 4. **Mixed Precision Training**

When running `accelerate config`, choose:

```
- Use FP16: yes
- Compute dtype: float16
```

You can also enable it in code:

```python
accelerator = Accelerator(mixed_precision="fp16")
```

---

### 🧠 Exercises

**Task 1:** Use `accelerate` to fine-tune BERT on the IMDb dataset with fp16 precision.

**Task 2:** Modify a basic PyTorch training loop and integrate `accelerate` for multi-GPU training (if available).

**Task 3:** Compare memory usage and speed between normal and mixed precision training.

