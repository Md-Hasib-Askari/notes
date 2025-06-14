

## 🔹 **6. Tokenization and Preprocessing**

### 📘 Notes

Transformers don’t understand raw text. They require **tokenization** — the process of converting text to numeric tensors (input IDs, attention masks).

Hugging Face offers two types of tokenizers:

| Type     | Description                                                 |
| -------- | ----------------------------------------------------------- |
| **Fast** | Backed by Rust, very efficient (`use_fast=True` by default) |
| **Slow** | Pure Python, easier to inspect/debug                        |

---

### ✅ 1. **Basic Tokenization**

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

tokens = tokenizer("Transformers are powerful!", return_tensors="pt")
print(tokens)
```

**Output Keys:**

* `input_ids`: numerical IDs of tokens
* `attention_mask`: 1 where there's a token, 0 for padding
* `token_type_ids`: used for sentence-pair tasks (e.g., QA)

---

### ✅ 2. **Inspect Tokens**

```python
print(tokenizer.tokenize("Transformers are powerful!"))
print(tokenizer.convert_tokens_to_ids(["transformers", "are", "powerful"]))
```

---

### ✅ 3. **Padding and Truncation**

```python
tokens = tokenizer(["short sentence", "This is a longer sentence that may be truncated"],
                   padding=True, truncation=True, return_tensors="pt")
```

---

### ✅ 4. **Decoding**

```python
ids = tokens["input_ids"][0]
print(tokenizer.decode(ids))  # Converts back to string
```

---

### ✅ 5. **Batch Tokenization**

```python
sentences = ["This is great.", "I love tokenization."]
batch = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
```

---

### ⚠️ Common Mistakes

* ❌ Using raw text directly with `model()` — always tokenize first.
* ❌ Ignoring padding when batching — may crash your model.
* ❌ Not setting `return_tensors="pt"` — model expects PyTorch/TF tensors.

---

### 🧠 Exercises

**Task 1**: Tokenize a single sentence and print:

* `input_ids`
* `attention_mask`
* tokenized words (with `tokenizer.tokenize()`)

**Task 2**: Try padding and truncating a batch of 3 sentences with very different lengths. Print the shape of the output tensors.

**Task 3**: Take some `input_ids` and decode them back to a string.

