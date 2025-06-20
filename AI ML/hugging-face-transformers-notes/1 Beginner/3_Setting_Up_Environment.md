
## 🔹 **3. Setting Up Environment**

### 📘 Notes

Before using Hugging Face Transformers effectively, you need a proper environment with **GPU support** for faster training/inference.

---

### 🖥️ 1. **Choose Your Environment**

#### ✅ Recommended Options:

| Platform                      | Pros           | Example                                                                |
| ----------------------------- | -------------- | ---------------------------------------------------------------------- |
| **Google Colab**              | Free GPU       | [https://colab.research.google.com](https://colab.research.google.com) |
| **Kaggle Notebooks**          | Free GPU       | [https://kaggle.com/code](https://kaggle.com/code)                     |
| **Local Machine**             | Full control   | Requires CUDA + GPU                                                    |
| **Paperspace / AWS / Lambda** | Paid, scalable | Good for production                                                    |

---

### ⚙️ 2. **Installing Required Libraries**

Use this one-liner to install everything:

```bash
pip install transformers datasets tokenizers accelerate evaluate huggingface_hub
```

---

### ⚡ 3. **Check for GPU Support**

In Python:

```python
import torch
print(torch.cuda.is_available())  # True if GPU is available
```

In Colab:

```python
!nvidia-smi
```

If you're using GPU, models will run much faster. Transformers can still work on CPU, just slower.

---

### 🧪 4. **Basic Test Pipeline**

Try this to check your installation:

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print(classifier("Transformers are amazing!"))
```

Output:

```python
[{'label': 'POSITIVE', 'score': 0.9998}]
```

---

### 🧠 Exercises

**Task 1**: Set up a Colab notebook or local Python environment and install all required Hugging Face libraries.

**Task 2**: Run the `torch.cuda.is_available()` check and post the result to confirm whether your system has GPU access.

**Task 3**: Run a `pipeline()` for any of the following:

* `"text-classification"`
* `"question-answering"`
* `"summarization"`

Log the output and make sure everything works.

