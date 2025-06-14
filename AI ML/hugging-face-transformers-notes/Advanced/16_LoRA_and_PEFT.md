
## 🔴 16. **Low-Rank Adaptation (LoRA) & PEFT**

> Learn how to fine-tune large models efficiently using *Parameter-Efficient Fine-Tuning* (PEFT) techniques like LoRA.

---

### 📘 Why PEFT?

Traditional fine-tuning of LLMs is:

* Expensive (millions/billions of parameters)
* Memory intensive
* Often unnecessary for many downstream tasks

**PEFT** allows you to:

* Fine-tune only a *small subset* of parameters
* Retain most of the base model frozen
* Reduce training cost drastically

---

### ✅ 1. **Install PEFT Library**

```bash
pip install peft
```

---

### ✅ 2. **Basic LoRA Workflow with PEFT**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")

# LoRA configuration
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Apply PEFT
model = get_peft_model(model, config)
model.print_trainable_parameters()
```

---

### ✅ 3. **Train the LoRA-Injected Model**

Use the Hugging Face `Trainer` or your own loop:

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=10,
    output_dir="./lora-checkpoints",
    save_total_limit=2,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
)

trainer.train()
```

---

### ✅ 4. **Inference with LoRA-Injected Model**

```python
model.eval()
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

### ✅ 5. **Saving & Loading**

```python
model.save_pretrained("./my-lora-model")
# Load again later:
# model = PeftModel.from_pretrained(base_model, "./my-lora-model")
```

---

### 🧠 Exercises

**Task 1:** Apply LoRA to fine-tune `gpt2` for text generation using a small custom dataset.
**Task 2:** Experiment with different `r` and `alpha` values in `LoraConfig` and compare output.
**Task 3:** Evaluate memory and speed improvements over full fine-tuning.

