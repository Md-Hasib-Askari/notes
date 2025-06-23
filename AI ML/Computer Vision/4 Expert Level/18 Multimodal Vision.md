## 🧪 Topic 18: **Multimodal Vision**

### 🎯 Goal:

Learn how to connect **images + text** so models can describe, understand, and reason about visual content using natural language.

---

## 🧠 What Is Multimodal Vision?

Multimodal models process **more than one type of data** (e.g., vision + language) and **align** them in a shared space.

> 🖼️ image ↔ 🧠 joint embedding space ↔ 📝 text

---

## 📦 1. **CLIP (Contrastive Language–Image Pre-training)** — by OpenAI

### 🔹 What It Does:

* Learns to **match images with their correct text descriptions**.
* Zero-shot classifier — you just provide **class names**, no training!

### 🧪 Example:

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

image = Image.open("cat.jpg")
texts = ["a photo of a dog", "a photo of a cat"]

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
pred = torch.argmax(logits_per_image)
print(texts[pred])  # Should print "a photo of a cat"
```

---

## 📦 2. **BLIP (Bootstrapped Language Image Pretraining)**

### 🔹 What It Does:

* **Generates captions** from images
* Can be used for **image-text retrieval** and **VQA (visual question answering)**

```python
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

image = Image.open("cat.jpg")
inputs = processor(images=image, return_tensors="pt")
output = model.generate(**inputs)
caption = processor.decode(output[0])
```

---

## 📦 3. **Flamingo (DeepMind)** — Multimodal Few-shot Learner

* Combines vision + text inputs with a **few-shot setup**
* Can **converse** about an image
* Requires large-scale infrastructure (used in Gemini, GPT-4V)

---

## 🔄 Tasks Multimodal Vision Can Do:

| Task                                | Example                                |
| ----------------------------------- | -------------------------------------- |
| **Zero-shot classification**        | CLIP predicts “cat” without retraining |
| **Image captioning**                | BLIP: “A black cat on a bed”           |
| **Visual question answering (VQA)** | “What color is the car?”               |
| **Visual search**                   | Search by image or description         |
| **Grounded language**               | Highlight “the red cup” in an image    |

---

## 🧪 Mini Project Ideas:

* Build a **zero-shot classifier** using CLIP and your own image classes.
* Use BLIP to **auto-caption** your photo collection.
* Integrate CLIP/BLIP into a **VQA chatbot**.
* Combine **SAM + CLIP** to create “segment anything you describe” apps.

---

## 📚 Summary:

| Model               | Use                               |
| ------------------- | --------------------------------- |
| **CLIP**            | Aligning vision + text, zero-shot |
| **BLIP**            | Image captioning, VQA             |
| **Flamingo**        | Multimodal few-shot tasks         |
| **SAM + CLIP/BLIP** | Visual grounding & segmentation   |

Multimodal vision is powering **AI agents**, **assistants**, **search engines**, and even **robotics**.
