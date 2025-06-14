

## 🔴 19. **Integrations**

> Hugging Face models can be integrated into **broader pipelines** like LangChain, Haystack, and **vector databases** for building production-grade AI systems.

---

### ✅ 1. **Integration with LangChain**

LangChain lets you build apps with LLMs, chaining together prompts, memory, tools, and retrieval.

#### Example: Using HF Model in LangChain

```python
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

hf_pipeline = pipeline("text-generation", model="gpt2")
llm = HuggingFacePipeline(pipeline=hf_pipeline)

llm("Translate to French: I love AI.")
```

Use `HuggingFaceHub` to connect to HF Hub directly:

```python
from langchain.llms import HuggingFaceHub

llm = HuggingFaceHub(repo_id="gpt2", model_kwargs={"temperature":0.5, "max_new_tokens":100})
```

---

### ✅ 2. **Using Transformers with Haystack**

Haystack helps with retrieval-based QA (RAG), semantic search, and indexing.

#### Example:

```python
from haystack.nodes import TransformersReader, FARMReader

reader = TransformersReader(model_name_or_path="deepset/roberta-base-squad2")
```

You can use Hugging Face models as the retriever or reader in a RAG pipeline.

---

### ✅ 3. **Integrating Vector Databases (e.g., FAISS)**

Use `SentenceTransformers` or HF embedding models to generate vectors and store/query them in FAISS.

#### Example:

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = ["How are you?", "What's your name?", "Tell me a joke"]
vectors = model.encode(texts)

# Indexing
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(np.array(vectors))

# Query
query = model.encode(["Who are you?"])
D, I = index.search(np.array(query), k=1)
print("Most similar:", texts[I[0][0]])
```

Use FAISS with Transformers' embeddings from models like `sentence-transformers`, `BAAI/bge-base-en`, or `thenlper/gte-base`.

---

### ✅ 4. **Combining All (RAG)**

* Use Hugging Face model to embed/query documents.
* Store embeddings in FAISS / Weaviate / Qdrant.
* Use LangChain or Haystack to manage flow:

  * `Retriever → Prompt Template → LLM`

---

### 🧠 Exercises

**Task 1:** Use LangChain with a Hugging Face LLM (e.g., `gpt2`, `tiiuae/falcon`).
**Task 2:** Build a FAISS-based search index using `sentence-transformers`.
**Task 3:** Integrate LangChain with a FAISS retriever and Hugging Face model to complete a QA task.

