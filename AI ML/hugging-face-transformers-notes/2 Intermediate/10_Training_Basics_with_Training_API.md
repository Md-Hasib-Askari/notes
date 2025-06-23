

## 🟡 **10. Training Basics with the Trainer API**

### 📘 Notes

The Hugging Face `Trainer` API simplifies training, evaluation, and saving models with minimal boilerplate. Ideal for fine-tuning transformers on custom datasets.

---

### ✅ 1. **Key Components**

* `TrainingArguments`: Settings like batch size, logging, epochs
* `Trainer`: Manages the training loop, evaluation, checkpoints
* `compute_metrics`: Optional function to evaluate performance

---

### ✅ 2. **Basic Setup**

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score

# Load data
dataset = load_dataset("imdb", split="train[:5000]").train_test_split(test_size=0.2)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

tokenized_ds = dataset.map(tokenize_fn, batched=True).remove_columns(["text"])
tokenized_ds.set_format("torch")

# Load model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
```

---

### ✅ 3. **Define TrainingArguments**

```python
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
)
```

---

### ✅ 4. **Add Evaluation Metrics**

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}
```

---

### ✅ 5. **Create and Run Trainer**

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    compute_metrics=compute_metrics,
)

trainer.train()
```

---

### ✅ 6. **Saving and Loading**

```python
trainer.save_model("./my-model")
# Load later:
# model = AutoModelForSequenceClassification.from_pretrained("./my-model")
```

---

### 🧠 Exercises

**Task 1**: Fine-tune `bert-base-uncased` on the `"ag_news"` dataset for 3 epochs with a batch size of 16.

**Task 2**: Add accuracy and log it after every epoch.

**Task 3**: Save your model to disk and try loading it again into a new script.

