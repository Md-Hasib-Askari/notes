## ✍️ **3. Prompt Engineering**

---

### 🔹 **3.1 What is Prompt Engineering?**

**Prompt Engineering** is the practice of crafting inputs (prompts) to large language models (LLMs) in a way that **maximizes the quality, relevance, and usefulness** of the output.

#### 🧠 Why It Matters:

* LLMs don’t “understand” intent — they rely on how you frame it.
* A slight change in wording can significantly alter results.
* It’s often more efficient than fine-tuning.

> **Prompting ≠ Programming**, but it’s just as powerful.

---

### 🔹 **3.2 Writing Good Prompts**

Here are essential principles and techniques:

---

#### ✅ **Be Specific in What You Want**

**Bad Prompt:**
"Write something about AI."

**Better Prompt:**
"Write a 150-word introduction to the concept of AI agents used in personal assistants."

✔️ Define task
✔️ Set format
✔️ Mention topic

---

#### ✅ **Provide Additional Context**

* Context improves relevance and accuracy.
* Include prior messages, objectives, or constraints.

```text
You're an AI assistant helping a user with scheduling. The user wants to book meetings only on weekdays between 10am–4pm.
Suggest 3 time slots for a 30-minute meeting next week.
```

---

#### ✅ **Use Relevant Technical Terms**

LLMs understand terminology from many domains. Use it to your advantage.

**Instead of:**
"Tell me how a model picks stuff."

**Use:**
"Explain how an LLM uses attention mechanisms to select the next token."

---

#### ✅ **Use Examples in Your Prompt**

**Few-shot prompting** shows the LLM what kind of output you expect.

```text
Translate English to Bengali:
English: How are you?
Bengali: তুমি কেমন আছো?

English: What is your name?
Bengali:
```

→ The model will now follow this structure.

---

#### ✅ **Iterate and Test Your Prompts**

Prompting is experimental. Tweak and evaluate:

* Try changing structure, phrasing, or constraints.
* A/B test outputs.
* Ask the model to explain its own thinking.

---

#### ✅ **Specify Length, Format, and Style**

**Structure your output!**

```text
Summarize this article in bullet points (max 5), using simple language.
```

Or:

```text
Output format:
- Title:
- Summary:
- Keywords:
```

Helps ensure consistency and parse-ability for downstream tools.

---

### 🧪 Prompt Engineering Tips for Agents

* **Use system prompts** to define behavior (e.g., "You are a helpful research assistant").
* **Guide reasoning**: use CoT ("Let's think step by step").
* **Control verbosity**: “Answer in under 100 words.”
