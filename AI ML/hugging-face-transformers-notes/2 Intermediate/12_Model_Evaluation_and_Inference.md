
## 🟡 **12. Model Evaluation and Inference**

### 📘 Notes

After training or fine-tuning, you’ll want to:

* **Evaluate** your model’s performance using proper metrics
* **Run inference** on new/unseen data
* **Save and load** models for reuse or deployment

---

### ✅ 1. **Common Evaluation Metrics**

| Task           | Metric                          |
| -------------- | ------------------------------- |
| Classification | Accuracy, Precision, Recall, F1 |
| Summarization  | ROUGE                           |
| Translation    | BLEU                            |
| Generation     | Perplexity, custom human evals  |

**Example (Classification - F1 Score):**

```python
from sklearn.metrics import f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {"f1": f1_score(labels, preds, average="weighted")}
```

---

### ✅ 2. **Inference with Fine-Tuned Model**

#### Text Classification (e.g., BERT)

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="./bert_agnews")
classifier("Breaking news: stock market hits record high.")
```

#### Text Generation (e.g., GPT-2)

```python
generator = pipeline("text-generation", model="./gpt2_finetuned")
generator("Once upon a time", max_length=50, num_return_sequences=1)
```

#### Summarization (e.g., T5)

```python
summarizer = pipeline("summarization", model="./t5_sum")
summarizer("The long article goes here...")
```

---

### ✅ 3. **Batch Inference with DataLoader**

```python
from torch.utils.data import DataLoader
import torch

model.eval()
dataloader = DataLoader(tokenized_ds["test"], batch_size=8)

all_preds = []
for batch in dataloader:
    with torch.no_grad():
        outputs = model(**{k: v.to(model.device) for k, v in batch.items()})
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
```

---

### ✅ 4. **Saving and Reloading Models**

```python
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")

# Reload later
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained("./my_model")
tokenizer = AutoTokenizer.from_pretrained("./my_model")
```

---

### 🧠 Exercises

**Task 1**: Load a fine-tuned model and run inference on 10 custom examples.

**Task 2**: Evaluate your model on validation data using F1, Accuracy, and Confusion Matrix.

**Task 3**: Save your model and test reloading it in a separate script for inference.

