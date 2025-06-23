
## 🔹 **1. Introduction to Transformers**

### 📘 Notes

#### What Are Transformers?

* A **transformer** is a deep learning architecture introduced in the paper ["Attention Is All You Need" (2017)](https://arxiv.org/abs/1706.03762).
* It uses **self-attention** to process input data in parallel, making it faster and more scalable than RNNs/LSTMs.

#### Key Concepts:

* **Self-Attention**: Each word in a sentence attends to every other word to understand the context.
* **Encoder-Decoder Architecture**:

  * **Encoder**: Processes input (e.g., sentence in English)
  * **Decoder**: Generates output (e.g., translated sentence in French)
* **Transformer-based Models**: BERT, GPT, T5, RoBERTa, DistilBERT, etc.

#### Why Transformers Matter:

* State-of-the-art in NLP tasks: translation, summarization, classification, etc.
* Foundation for large language models (LLMs) like GPT-4, LLaMA, Claude, etc.

#### Differences from RNNs:

| Feature           | RNN / LSTM    | Transformer    |
| ----------------- | ------------- | -------------- |
| Processing        | Sequential    | Parallel       |
| Long Dependencies | Weak          | Strong         |
| Training Speed    | Slow          | Fast           |
| Context Window    | Fixed/limited | Dynamic/global |

---

### 🧠 Exercise

**Task 1**: In your own words, write a 5–7 sentence summary of how transformers differ from RNNs and why they are widely used in NLP today.

**Task 2**: Read and summarize this paper:

> ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) — Read the abstract and introduction. Summarize the problem and the proposed solution.

**Task 3**: Watch this explainer video:

> [YouTube: The Illustrated Transformer by Jay Alammar](https://www.youtube.com/watch?v=4Bdc55j80l8) — Summarize the role of self-attention in less than 100 words.

