
## 🔴 15. **Custom Model Architectures**

> Create your own transformer-based models or modify existing ones using `PreTrainedModel`.

---

### 📘 Why Customize Models?

* To **extend** existing models (e.g., add a new classification or regression head)
* To **build new models** from scratch using custom layers
* To **adapt architectures** for new tasks

---

### ✅ 1. **Extending a Pretrained Model**

Example: Add a **regression head** on top of BERT.

```python
from transformers import BertModel, BertConfig, PreTrainedModel
import torch.nn as nn

class BertForRegression(PreTrainedModel):
    config_class = BertConfig

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.regressor = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.regressor(pooled_output)
```

---

### ✅ 2. **Using Your Custom Model**

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("Example sentence", return_tensors="pt")

model = BertForRegression.from_pretrained("bert-base-uncased")
output = model(**inputs)
print(output)
```

---

### ✅ 3. **Training from Scratch**

If you're **not using a pretrained model**, use a config object:

```python
config = BertConfig(
    vocab_size=30522,
    hidden_size=256,
    num_attention_heads=4,
    num_hidden_layers=4,
    intermediate_size=512,
)
model = BertForRegression(config)
```

---

### ✅ 4. **Custom Heads for Other Tasks**

* **Multi-label classification**: Use `nn.Sigmoid` instead of `nn.Softmax`
* **Token classification**: Output per token from `last_hidden_state`
* **Question answering**: Predict start and end positions with two output layers

---

### 🧠 Exercises

**Task 1:** Create a custom BERT model with a regression head to predict a continuous score (e.g., IMDb rating from a review).

**Task 2:** Modify `GPT2LMHeadModel` to add an additional linear layer after logits.

**Task 3:** Build a transformer model from scratch using `BertConfig`, train it on dummy data.
