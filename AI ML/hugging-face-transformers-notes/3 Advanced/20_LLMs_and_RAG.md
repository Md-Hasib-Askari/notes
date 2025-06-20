
## 🔴 20. **LLMs and Retrieval-Augmented Generation (RAG)**

> This is where Hugging Face truly scales — using **open-source LLMs** like LLaMA or Falcon with **retrieval-based augmentation** for domain-specific, accurate, and efficient generation.

---

### ✅ 1. **Using Large Language Models**

Open-source LLMs like:

* `meta-llama/Llama-2-7b-chat-hf`
* `tiiuae/falcon-7b`
* `mistralai/Mistral-7B-Instruct-v0.2`
* `google/gemma-7b`

Load with:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
```

Generate:

```python
inputs = tokenizer("What is RAG?", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

---

### ✅ 2. **Building a RAG Pipeline**

**RAG = Retriever + Generator**

* **Retriever:** Gets relevant docs (from FAISS/Qdrant/Weaviate)
* **Generator:** Uses them as context to generate answers

With `transformers`:

```python
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
retriever = RagRetriever.from_pretrained("facebook/rag-token-base", index_name="exact", use_dummy_dataset=True)
model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-base", retriever=retriever)
```

---

### ✅ 3. **LangChain + RAG**

```python
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings=embedding_model)

retriever = vectorstore.as_retriever()
llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct")

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
qa.run("What is the use of LoRA?")
```

---

### ✅ 4. **Optimizations**

* Use **vectorstore caching** (e.g., chroma, FAISS, Qdrant) to avoid re-embedding.
* Use **streaming generation** with `generate(..., streamer=TextStreamer)`.
* Optimize models with:

  * LoRA + PEFT
  * Quantization
  * DeepSpeed or Accelerate

---

### 🧠 Exercises

**Task 1:** Fine-tune or run inference with `Llama`, `Falcon`, or `Mistral`.
**Task 2:** Build a LangChain-based RAG pipeline with FAISS and HF model.
**Task 3:** Optimize a RAG system with quantized models and caching.

