

## 🟡 **8. Data Preprocessing**

### 📘 Notes

Preprocessing transforms raw datasets into **model-ready tensors** using:

* Tokenizers
* Padding & truncation
* Collators (for batching)

---

### ✅ 1. **Tokenizing a Dataset**

```python
from transformers import AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = load_dataset("imdb", split="train[:5000]")

def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
```

> ✅ `batched=True` processes items in chunks for speed.

---

### ✅ 2. **Padding and Truncation Options**

```python
tokenizer(example["text"], padding="max_length", truncation=True, max_length=256)
```

* `padding="max_length"` ensures all sequences are the same length.
* `truncation=True` avoids exceeding model max length.

---

### ✅ 3. **Remove Unnecessary Columns**

After tokenizing, remove original text fields:

```python
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format("torch")  # or "tensorflow"
```

---

### ✅ 4. **Data Collators**

Used to batch together variable-length sequences dynamically.

```python
from transformers import DataCollatorWithPadding

collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
```

Usage with PyTorch DataLoader:

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(tokenized_dataset, batch_size=8, collate_fn=collator)
```

---

### 🧠 Exercises

**Task 1**: Load and tokenize the `"ag_news"` dataset using `bert-base-uncased`. Apply:

* truncation
* padding
* `max_length=128`

**Task 2**: Remove unnecessary columns (`text`, `label`) and set the format to PyTorch.

**Task 3**: Set up a `DataLoader` using `DataCollatorWithPadding`. Print the shape of the first batch.

