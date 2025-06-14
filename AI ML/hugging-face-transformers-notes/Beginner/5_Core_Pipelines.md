

## 🔹 **5. Core Pipelines (Detailed Use Cases)**

### 📘 Notes

The `pipeline()` API abstracts away model/tokenizer loading and makes **inference easy**.

---

### ✅ 1. **Sentiment Analysis**

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("I feel great today!")
print(result)
```

**Output:**

```json
[{'label': 'POSITIVE', 'score': 0.9998}]
```

---

### ✅ 2. **Named Entity Recognition (NER)**

```python
ner = pipeline("ner", grouped_entities=True)
ner("Hugging Face Inc. is based in New York and was founded by Julien.")
```

**Grouped output** merges consecutive tokens into entities.

---

### ✅ 3. **Question Answering**

```python
qa = pipeline("question-answering")
qa({
    "question": "Where is Hugging Face based?",
    "context": "Hugging Face Inc. is based in New York."
})
```

---

### ✅ 4. **Summarization**

```python
summarizer = pipeline("summarization")
summarizer("Transformers are state-of-the-art models for NLP. They use self-attention and are very efficient for large-scale data.", max_length=30, min_length=5)
```

---

### ✅ 5. **Text Generation**

```python
generator = pipeline("text-generation", model="gpt2")
generator("In the future, AI will", max_length=30, num_return_sequences=1)
```

---

### ✅ 6. **Translation**

```python
translator = pipeline("translation_en_to_fr")
translator("Transformers are very powerful in NLP.")
```

> Translation pipelines follow the pattern: `"translation_<src>_to_<tgt>"`

---

### ✅ 7. **Zero-Shot Classification**

No fine-tuning needed. Just describe your labels.

```python
classifier = pipeline("zero-shot-classification")
classifier("This is a business email.", candidate_labels=["business", "sports", "politics"])
```

---

### 🧠 Exercises

**Task 1**: Run and observe output for each of the following pipelines:

* `sentiment-analysis`
* `ner`
* `question-answering`
* `summarization`
* `text-generation`
* `translation_en_to_fr`
* `zero-shot-classification`

**Task 2**: Write a single script that asks for user input and dynamically picks a pipeline based on a CLI arg like:

```bash
python app.py --task sentiment
```

**Task 3 (Optional)**: Create a small command-line app with `argparse` to test different pipeline tasks interactively.

