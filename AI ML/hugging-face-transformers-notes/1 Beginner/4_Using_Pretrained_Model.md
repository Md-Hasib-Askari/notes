
## 🔹 **4. Using Pretrained Models**

### 📘 Notes

Hugging Face provides thousands of **pretrained models** for different tasks. You can load and use them in **1-2 lines of code** using `AutoModel` classes or `pipeline()`.

---

### ✅ 1. **Loading Models and Tokenizers**

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

---

### ✅ 2. **Using Pipelines (Quick & Easy)**

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love using Hugging Face Transformers!")
print(result)
```

---

### 🧠 Popular Tasks & Pipelines

| Task               | Pipeline Name          | Example                                     |
| ------------------ | ---------------------- | ------------------------------------------- |
| Sentiment Analysis | `"sentiment-analysis"` | `pipeline("sentiment-analysis")`            |
| Text Generation    | `"text-generation"`    | `pipeline("text-generation", model="gpt2")` |
| Question Answering | `"question-answering"` | `pipeline("question-answering")`            |
| Summarization      | `"summarization"`      | `pipeline("summarization")`                 |
| Translation        | `"translation"`        | `pipeline("translation_en_to_fr")`          |
| NER                | `"ner"`                | `pipeline("ner", grouped_entities=True)`    |

---

### ⚙️ 3. **Custom Inputs**

```python
qa = pipeline("question-answering")
qa({
    'question': 'What is Transformers?',
    'context': 'Transformers are models introduced in the Attention is All You Need paper.'
})
```

```python
generator = pipeline("text-generation", model="gpt2")
generator("The future of AI is", max_length=30, num_return_sequences=1)
```

---

### 💡 Behind the Scenes

When you call `from_pretrained()`, it:

1. Downloads the model weights from Hugging Face Hub.
2. Caches them locally.
3. Loads the architecture automatically.

You can also pass `cache_dir` or `revision` to load specific versions.

---

### 🧠 Exercises

**Task 1**: Use a pipeline to try:

* Sentiment Analysis
* Question Answering
* Text Generation

Post the results from at least one.

**Task 2**: Use `AutoTokenizer` and `AutoModelForSequenceClassification` to:

* Load the `"distilbert-base-uncased-finetuned-sst-2-english"` model
* Tokenize a sentence manually
* Pass it to the model and print raw logits

```python
import torch

inputs = tokenizer("Transformers are awesome!", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
print(outputs.logits)
```

