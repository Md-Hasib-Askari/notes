

## 🗣️ 9. Natural Language Processing (NLP) – Notes

### 📌 Overview:

NLP enables machines to **read, understand, generate, and respond to human language** using a mix of linguistics and deep learning.

---

### 🔤 9.1 Text Preprocessing

#### ✅ Core Steps:

* **Tokenization**: Split text into words or subwords
* **Lowercasing**
* **Stopword Removal**: Remove common but useless words (e.g., "the", "is")
* **Stemming**/**Lemmatization**: Reduce words to base/root forms
* **Padding**: Make sequences equal in length

#### ✅ Libraries:

* `nltk`, `spaCy`, `re`, `transformers` tokenizers

---

### 📚 9.2 Word Embeddings

#### ✅ Idea:

Map words into dense vector space to capture semantic meaning.

#### ✅ Techniques:

| Type       | Example         | Notes                         |
| ---------- | --------------- | ----------------------------- |
| Static     | Word2Vec, GloVe | Same vector for a word always |
| Contextual | BERT, GPT       | Varies with context           |

```python
from gensim.models import Word2Vec
model = Word2Vec(sentences, vector_size=100)
```

---

### 🧠 9.3 RNN, LSTM, GRU

#### ✅ Recurrent Neural Networks (RNNs):

* Process sequences step-by-step
* Capture time-dependent patterns

#### ✅ LSTM / GRU:

* Handle long-term dependencies
* Avoid vanishing gradients

```python
nn.LSTM(input_size=100, hidden_size=64)
```

---

### 🤗 9.4 Transformers

#### ✅ Key Innovations:

* **Self-Attention**: Compare each word with all others
* **Positional Encoding**: Adds order info to input tokens
* **Parallelization**: No sequential processing needed

#### ✅ Popular Architectures:

| Model   | Use Case               |
| ------- | ---------------------- |
| BERT    | Encoding/understanding |
| GPT     | Text generation        |
| T5      | Text-to-text tasks     |
| RoBERTa | Robust variant of BERT |

```python
from transformers import pipeline
qa = pipeline("question-answering")
```

---

### 🧠 9.5 NLP Tasks

| Task                           | Description                        | Example Model   |
| ------------------------------ | ---------------------------------- | --------------- |
| Text Classification            | Spam detection, sentiment analysis | BERT, RoBERTa   |
| Named Entity Recognition (NER) | Tag entities like names, orgs      | spaCy, BERT     |
| Machine Translation            | Translate between languages        | MarianMT, T5    |
| Question Answering             | Answer from context                | BERT, DeBERTa   |
| Text Summarization             | Shorten content meaningfully       | T5, Pegasus     |
| Chatbots/Dialogue              | Human-like conversation            | GPT, BlenderBot |

