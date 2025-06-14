
## 🔴 14. **Tokenizers Library (Advanced)**

> Build custom tokenizers from scratch using the `tokenizers` library (the low-level core behind 🤗 Transformers' tokenizer system).

---

### 📘 Why Use Custom Tokenizers?

* Improve performance and efficiency on domain-specific corpora
* Reduce vocabulary size
* Handle custom tokens for specific tasks (e.g., code, legal, medical)

---

### ✅ 1. **Install and Import**

```bash
pip install tokenizers
```

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
```

---

### ✅ 2. **Training a BPE Tokenizer**

```python
# Initialize tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Pre-tokenizer (splits on whitespace)
tokenizer.pre_tokenizer = Whitespace()

# Trainer
trainer = BpeTrainer(vocab_size=5000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

# Train on files
files = ["./data/corpus.txt"]
tokenizer.train(files, trainer)

# Save tokenizer
tokenizer.save("custom-tokenizer.json")
```

---

### ✅ 3. **Load and Use Your Tokenizer**

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("custom-tokenizer.json")

encoded = tokenizer.encode("Hello world!")
print(encoded.tokens)
print(encoded.ids)
```

---

### ✅ 4. **Using with 🤗 Transformers**

```python
from transformers import PreTrainedTokenizerFast

hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file="custom-tokenizer.json")
hf_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

text = "Tokenize this with my custom tokenizer."
tokens = hf_tokenizer(text)
```

---

### 🛠 Optional: Train WordPiece or Unigram

* `WordPiece` is used by BERT
* `Unigram` is used by T5, XLNet

```python
from tokenizers.models import WordPiece
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
```

---

### 🧠 Exercises

**Task 1**: Collect a corpus of at least 1000 lines of text and train a custom BPE tokenizer.

**Task 2**: Save your tokenizer and load it with `PreTrainedTokenizerFast`.

**Task 3**: Compare vocabulary sizes of your tokenizer vs. `bert-base-uncased`.

