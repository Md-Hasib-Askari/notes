# hugging-face-transformers-notes


## 🟢 Beginner Level: Foundations

### 1. **Introduction to Transformers**

* What are transformers?
* Attention mechanism basics
* Difference between RNNs and Transformers

### 2. **Hugging Face Ecosystem Overview**

* `transformers` library
* `datasets` library
* `tokenizers` library
* `hub`, `spaces`, `accelerate` overview

### 3. **Setting Up Environment**

* Installing `transformers`, `datasets`, and `tokenizers`
* Using GPU (Colab, Kaggle, or local setup)

### 4. **Using Pretrained Models**

* Loading models with `from_pretrained()`
* Loading tokenizers
* Text classification with pipelines

### 5. **Core Pipelines**

* Sentiment analysis
* Text generation
* Question answering
* Named Entity Recognition (NER)
* Translation and summarization

### 6. **Tokenizer Basics**

* Tokenization concepts
* WordPiece, BPE, SentencePiece
* Using tokenizer.encode/decode

---

## 🟡 Intermediate Level: Customization & Training

### 7. **Hugging Face Datasets**

* Loading datasets with `load_dataset()`
* Exploring dataset splits and formats
* Creating custom datasets

### 8. **Data Preprocessing**

* Tokenizing datasets
* Padding and truncation
* Data collators and batching

### 9. **Model Architecture and Config**

* Model classes (e.g., `BertForSequenceClassification`, `GPT2LMHeadModel`)
* Configurations with `AutoConfig`
* Changing model parameters

### 10. **Training Basics with Trainer API**

* `Trainer` and `TrainingArguments`
* Evaluating with metrics
* Logging and saving models

### 11. **Fine-Tuning**

* Fine-tuning BERT for classification
* Fine-tuning T5 for summarization
* GPT-2 for custom text generation

### 12. **Model Evaluation and Inference**

* Accuracy, F1, ROUGE, BLEU, etc.
* Batch inference with DataLoader
* Saving/loading fine-tuned models

---

## 🔴 Advanced Level: Custom Models & Deployment

### 13. **Accelerate & Distributed Training**

* Using `accelerate` for multi-GPU/TPU
* Mixed precision (fp16)
* Custom training loops with `accelerate`

### 14. **Tokenizers Library (Advanced)**

* Train your own tokenizer from scratch
* Handling special tokens, vocab limits

### 15. **Custom Model Architectures**

* Creating a new model with `PreTrainedModel`
* Modifying heads (e.g., regression instead of classification)

### 16. **Low-Rank Adaptation (LoRA) & PEFT**

* Parameter-efficient fine-tuning
* Using `peft` with LoRA adapters

### 17. **Using Hugging Face Hub**

* Uploading models and datasets
* Versioning and sharing
* Using model cards and metadata

### 18. **Deployment**

* Deploying to Hugging Face Spaces (Gradio/Streamlit)
* Using `transformers` with FastAPI
* Optimizing models with ONNX, TorchScript, or quantization

### 19. **Integrations**

* Integration with LangChain
* Using models in pipelines like Haystack
* Plugging into vector databases (e.g., FAISS + Transformers)

### 20. **LLMs and Retrieval-Augmented Generation (RAG)**

* Using models like LLaMA, Falcon, Mistral
* RAG with `rag-token` or custom pipelines
* Caching and optimization for inference

