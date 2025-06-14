

## 🧠 12. Advanced NLP & LLMs – Notes

### 📌 Overview:

Advanced NLP focuses on **state-of-the-art models like Transformers and LLMs**, which power applications such as ChatGPT, Google Translate, and summarization tools.

---

### ⚙️ 12.1 Transformers In Depth

#### ✅ Core Concepts:

* **Self-Attention**: Computes attention weights across all tokens.
* **Multi-Head Attention**: Captures information from multiple subspaces.
* **Positional Encoding**: Adds position context to tokens.
* **Encoder-Decoder Architecture** (e.g., T5, BART): Used for translation, summarization.

#### ✅ Resources:

* Papers: [Attention is All You Need](https://arxiv.org/abs/1706.03762)
* Libraries: `transformers`, `xformers`, `bertviz`

---

### 🧠 12.2 BERT & GPT Family

| Model     | Type            | Use Cases                  |
| --------- | --------------- | -------------------------- |
| BERT      | Encoder         | Classification, NER        |
| RoBERTa   | BERT++          | Same, with better training |
| GPT       | Decoder         | Text generation, chatbots  |
| GPT-2/3/4 | Decoder         | Summarization, dialogue    |
| T5        | Encoder-decoder | Text-to-text               |

#### ✅ HuggingFace Usage Example:

```python
from transformers import pipeline
qa = pipeline("question-answering", model="distilbert-base-cased")
qa(question="What is AI?", context="AI is the field of making machines intelligent.")
```

---

### 🛠️ 12.3 Fine-Tuning vs Prompting

#### ✅ Fine-Tuning:

* Retrain the model on task-specific data
* Requires compute + labeled data
* Better for long-term tasks or domain shifts

#### ✅ Prompting:

* Provide context + instructions (e.g., "Summarize:")
* Zero-shot / few-shot learning
* Use APIs like OpenAI's `chat/completions`

---

### 📚 12.4 RAG (Retrieval-Augmented Generation)

#### ✅ Workflow:

1. **Retrieve**: Search relevant documents using embeddings (via FAISS, Weaviate, etc.)
2. **Augment**: Inject them into prompt/context
3. **Generate**: Use LLM to answer based on retrieved info

#### ✅ Use Cases:

* Chatbots with custom knowledge
* Search over PDFs, websites, internal docs

---

### 🔗 12.5 LangChain & LlamaIndex

#### ✅ LangChain:

* Framework for building LLM-powered applications
* Components: Chains, Agents, Tools, Memory

```python
from langchain.chains import LLMChain
```

#### ✅ LlamaIndex:

* Connect LLMs to external data sources
* Index PDFs, SQL, Notion, etc.

```python
from llama_index import SimpleDirectoryReader
```

