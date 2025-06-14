

## 🟡 **11. Fine-Tuning**

### 📘 Notes

Fine-tuning is where you adapt a **pretrained model** to your **custom task and data**. With Hugging Face, you can fine-tune models like:

* **BERT** → Text classification
* **T5** → Summarization or translation
* **GPT-2** → Text generation

---

### ✅ 1. **Fine-Tuning BERT for Classification**

```python
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

dataset = load_dataset("ag_news").train_test_split(test_size=0.2)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

tokenized_ds = dataset.map(tokenize_fn, batched=True).remove_columns(["text"])
tokenized_ds.set_format("torch")

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)

training_args = TrainingArguments(output_dir="./bert_agnews", evaluation_strategy="epoch", per_device_train_batch_size=16, num_train_epochs=3)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_ds["train"], eval_dataset=tokenized_ds["test"])
trainer.train()
```

---

### ✅ 2. **Fine-Tuning T5 for Summarization**

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

def preprocess(example):
    inputs = "summarize: " + example["article"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(example["highlights"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:5000]").map(preprocess, batched=True)
dataset.set_format("torch")

training_args = TrainingArguments(output_dir="./t5_sum", per_device_train_batch_size=4, num_train_epochs=2)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
```

---

### ✅ 3. **Fine-Tuning GPT-2 for Text Generation**

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Required for padding

model = GPT2LMHeadModel.from_pretrained("gpt2")

def preprocess(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:2000]").map(preprocess, batched=True)
dataset.set_format("torch")

training_args = TrainingArguments(output_dir="./gpt2_finetuned", per_device_train_batch_size=4, num_train_epochs=3)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
```

---

### 🧠 Exercises

**Task 1**: Fine-tune `bert-base-uncased` on `"yelp_review_full"` dataset with 5 sentiment classes.

**Task 2**: Fine-tune `t5-small` for summarizing Bangla news (if available, or English alternatives).

**Task 3**: Fine-tune `gpt2` on your own text corpus (e.g., customer service replies or social media comments).

