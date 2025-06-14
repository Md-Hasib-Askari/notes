
## 📘 3.3: **Natural Language Processing (NLP)**

### 🎯 Goal

Learn how deep learning models understand, represent, and generate human language using embeddings and Transformer-based models.

---

## 🧠 Core Concepts

### 1. **Why is NLP hard?**

* Language is **contextual** (e.g., “bank” can mean riverbank or finance).
* Meaning depends on **order**, **syntax**, **semantics**, and **ambiguity**.
* Traditional methods (like Bag-of-Words or TF-IDF) can't capture meaning or order.

---

## 🔤 2. **Word Embeddings**

Word embeddings convert discrete tokens (words) into continuous vector space.

### 🔹 Common Techniques:

| Method       | Description                                               |
| ------------ | --------------------------------------------------------- |
| **Word2Vec** | Predict surrounding words (Skip-Gram/CBOW)                |
| **GloVe**    | Global word co-occurrence matrix factorization            |
| **FastText** | Embeddings for subwords (handles out-of-vocabulary words) |

**Example:**

```plaintext
king - man + woman ≈ queen
```

---

## 🔥 3. **Transformers in NLP**

> Revolutionized NLP by allowing parallel computation and capturing global context through **self-attention**.

### 🔹 Key Concepts:

* **Tokenization**: Subword units (BPE, WordPiece).
* **Positional Encoding**: Adds position info to tokens.
* **Encoder-Decoder** structure: Used in models like T5, BART.

---

## 🧠 4. **Popular Models**

| Model          | Task                       | Notes                       |
| -------------- | -------------------------- | --------------------------- |
| **BERT**       | Text classification, QA    | Bidirectional encoder       |
| **GPT**        | Text generation, chatbots  | Unidirectional decoder      |
| **T5**         | Translation, summarization | Text-to-text format         |
| **DistilBERT** | Lightweight BERT           | Faster, good for deployment |

---

## 🧪 Applications of NLP

* **Text Classification**: Spam detection, sentiment analysis
* **Named Entity Recognition (NER)**: Extracting people, locations, etc.
* **Question Answering**: e.g., SQuAD
* **Translation**: Multilingual NLP
* **Text Generation**: Chatbots, story writing

---

## 🧰 Practical Example (Transformers + Hugging Face)

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print(classifier("I love deep learning!"))
```

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

inputs = tokenizer("Once upon a time", return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_length=50)
print(tokenizer.decode(outputs[0]))
```

---

## 🧪 Exercises

### ✅ Conceptual

1. Compare Word2Vec and GloVe.
2. Why is BERT bidirectional and GPT unidirectional?

### ✅ Practical

* Train Word2Vec on a custom dataset using `gensim`.
* Fine-tune BERT on a sentiment classification dataset.
* Generate creative text using GPT-2.

