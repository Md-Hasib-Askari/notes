

## 🟡 **7. Hugging Face Datasets**

### 📘 Notes

The 🤗 `datasets` library is tightly integrated with `transformers`. It supports:

* **Loading public datasets** (e.g., IMDb, SQuAD, GLUE)
* **Creating custom datasets**
* **Efficient preprocessing & streaming**

---

### ✅ 1. **Installing Datasets**

```bash
pip install datasets
```

---

### ✅ 2. **Loading a Dataset**

```python
from datasets import load_dataset

dataset = load_dataset("imdb")
print(dataset)
```

**Output:**

```bash
DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 25000
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 25000
    })
    unsupervised: Dataset({...})
})
```

---

### ✅ 3. **Inspecting Data**

```python
print(dataset["train"][0])
print(dataset["train"].features)
```

---

### ✅ 4. **Creating a Custom Dataset**

```python
from datasets import Dataset

data = {
    "text": ["I love Transformers!", "Hugging Face is cool."],
    "label": [1, 1]
}
custom_ds = Dataset.from_dict(data)
print(custom_ds)
```

---

### ✅ 5. **Dataset Splits**

Use `train_test_split()` to manually split:

```python
split_ds = custom_ds.train_test_split(test_size=0.2)
print(split_ds["train"][0])
```

---

### ✅ 6. **Streaming Large Datasets**

```python
streamed = load_dataset("ag_news", split="train", streaming=True)
for sample in streamed:
    print(sample)
    break
```

---

### 🧠 Exercises

**Task 1**: Load the `"ag_news"` dataset. Print:

* Number of training samples
* First 3 samples with text and label

**Task 2**: Create a dataset with at least 5 custom sentences and binary labels. Use `Dataset.from_dict()` and split it.

**Task 3**: Try streaming any large dataset (`cc_news`, `mc4`, etc.) and print 1 sample.

