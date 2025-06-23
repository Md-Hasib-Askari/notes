

## 🔴 18. **Deployment**

> Once your model is trained and pushed to the Hub, it’s time to **serve it for real users** — via UI apps or APIs — and optimize for performance.

---

### ✅ 1. **Deploy on Hugging Face Spaces (Gradio / Streamlit)**

Spaces allow easy public-facing interfaces for your models.

#### Gradio Example (`app.py`):

```python
import gradio as gr
from transformers import pipeline

pipe = pipeline("text-generation", model="gpt2")

def generate_text(prompt):
    return pipe(prompt, max_new_tokens=50)[0]["generated_text"]

gr.Interface(fn=generate_text, inputs="text", outputs="text").launch()
```

Then push it to a Space:

```bash
git lfs install
git clone https://huggingface.co/spaces/your-username/your-space-name
cd your-space-name
# copy app.py and requirements.txt
git add . && git commit -m "initial commit" && git push
```

---

### ✅ 2. **Deploy with FastAPI**

Use `transformers` in your FastAPI backend to serve your model.

#### Example:

```python
from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()
pipe = pipeline("sentiment-analysis")

@app.get("/predict")
def predict(text: str):
    return pipe(text)
```

Run with:

```bash
uvicorn main:app --reload
```

---

### ✅ 3. **Optimizing Models**

To speed up inference or reduce resource use:

#### a. **Quantization (e.g., 8-bit)**

```bash
pip install bitsandbytes accelerate
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained("gpt2", quantization_config=bnb_config, device_map="auto")
```

#### b. **ONNX Conversion**

```bash
pip install optimum[onnxruntime]
```

```python
from optimum.onnxruntime import ORTModelForSequenceClassification

model = ORTModelForSequenceClassification.from_pretrained("bert-base-uncased", export=True)
```

#### c. **TorchScript / Tracing**

```python
traced = torch.jit.trace(model, example_inputs)
torch.jit.save(traced, "traced_model.pt")
```

---

### ✅ 4. **Batch Inference**

Use `torch.utils.data.DataLoader` for fast batch inference:

```python
from torch.utils.data import DataLoader
loader = DataLoader(tokenized_data, batch_size=16)

for batch in loader:
    outputs = model(**batch)
```

---

### 🧠 Exercises

**Task 1:** Deploy a model with Gradio on HF Spaces.
**Task 2:** Build a FastAPI backend to serve a model via REST.
**Task 3:** Convert a Transformer model to ONNX or 8-bit and compare performance.

