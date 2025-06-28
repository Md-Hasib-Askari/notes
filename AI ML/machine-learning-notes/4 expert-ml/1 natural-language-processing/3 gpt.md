# GPT (Generative Pre-trained Transformer)

## Overview
GPT is a family of autoregressive language models that use transformer architecture for text generation. GPT models are trained to predict the next word in a sequence, making them powerful for generative tasks.

## Key Features
- **Autoregressive**: Generates text one token at a time, left-to-right
- **Decoder-only**: Uses only the decoder part of the transformer
- **Causal Attention**: Can only attend to previous tokens (masked self-attention)
- **Unsupervised Pre-training**: Trained on large text corpora without labels

## Model Evolution
### GPT-1 (2018)
- **Parameters**: 117M
- **Training Data**: BookCorpus dataset
- **Innovation**: Demonstrated unsupervised pre-training + supervised fine-tuning

### GPT-2 (2019)
- **Parameters**: 124M to 1.5B
- **Training Data**: WebText (40GB of text)
- **Innovation**: Zero-shot task performance without fine-tuning

### GPT-3 (2020)
- **Parameters**: 175B
- **Training Data**: Common Crawl, WebText2, Books1, Books2, Wikipedia
- **Innovation**: Few-shot learning with in-context examples

### GPT-4 (2023)
- **Parameters**: Estimated 1-100T (multimodal)
- **Innovation**: Multimodal capabilities (text + images)

## Architecture
```
Input Tokens → Token Embeddings + Positional Embeddings →
Transformer Decoder Layers (Masked Self-Attention + Feed-Forward) →
Language Modeling Head → Next Token Probabilities
```

## Training Process
### Pre-training
```python
# Objective: Maximize likelihood of next token
Loss = -Σ log P(token_i | token_1, ..., token_{i-1})
```

### Fine-tuning (Optional)
```python
# Task-specific fine-tuning with supervised data
# Example: Text classification, summarization
```

## Implementation Example
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained GPT-2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Add padding token
tokenizer.pad_token = tokenizer.eos_token

# Generate text
prompt = "The future of artificial intelligence is"
inputs = tokenizer.encode(prompt, return_tensors='pt')

# Generate with sampling
with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_length=100,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

## Generation Strategies
### Greedy Decoding
```python
# Always select the most probable next token
outputs = model.generate(inputs, max_length=50)
```

### Beam Search
```python
# Keep track of multiple hypotheses
outputs = model.generate(inputs, num_beams=5, max_length=50)
```

### Sampling
```python
# Sample from probability distribution
outputs = model.generate(
    inputs, 
    do_sample=True, 
    temperature=0.8,
    top_k=50,
    top_p=0.9
)
```

## Prompt Engineering
### Few-shot Learning
```python
prompt = """
Translate English to French:
English: Hello
French: Bonjour

English: Thank you
French: Merci

English: Good morning
French:"""
```

### Chain-of-Thought
```python
prompt = """
Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. 
Each can has 3 tennis balls. How many tennis balls does he have now?

Let me think step by step:
1. Roger starts with 5 tennis balls
2. He buys 2 cans, each with 3 balls
3. New balls = 2 × 3 = 6 balls
4. Total = 5 + 6 = 11 balls

Answer: 11 tennis balls
"""
```

## Applications
- **Text Generation**: Creative writing, content creation
- **Code Generation**: Programming assistance (GitHub Copilot)
- **Conversational AI**: ChatGPT, virtual assistants
- **Text Completion**: Auto-complete, email drafting
- **Translation**: Language-to-language conversion
- **Summarization**: Document and article summarization

## Advantages
- **Versatile**: Single model for multiple tasks
- **Few-shot Learning**: Learns from examples in context
- **Human-like Text**: Generates coherent, contextual text
- **No Task-specific Training**: Works zero-shot on many tasks

## Disadvantages
- **Computational Cost**: Large models require significant resources
- **Hallucination**: May generate false or nonsensical information
- **Bias**: Reflects biases present in training data
- **Context Length**: Limited by maximum sequence length
- **No Knowledge Cutoff**: Training data has a cutoff date

## Best Practices
1. **Prompt Engineering**: Craft clear, specific prompts
2. **Temperature Control**: Adjust for creativity vs coherence
3. **Context Management**: Keep important information in context
4. **Iterative Refinement**: Improve prompts based on outputs
5. **Safety Considerations**: Monitor for harmful or biased content

## Fine-tuning Techniques
### Instruction Tuning
```python
# Train on instruction-following examples
"Instruction: Summarize the following text
Input: [long text]
Output: [summary]"
```

### Reinforcement Learning from Human Feedback (RLHF)
```python
# Use human feedback to improve model behavior
# 1. Collect human preferences
# 2. Train reward model
# 3. Optimize policy with PPO
```

## Evaluation Metrics
- **Perplexity**: Measures how well model predicts text
- **BLEU Score**: For translation and generation tasks
- **Human Evaluation**: Coherence, relevance, fluency
- **Task-specific Metrics**: Accuracy, F1-score for downstream tasks

## Resources
- **Original Papers**: GPT, GPT-2, GPT-3, GPT-4 research papers
- **OpenAI API**: Access to GPT-3.5 and GPT-4 models
- **Hugging Face**: Open-source GPT models and tools
- **LangChain**: Framework for building LLM applications
- **Prompt Engineering Guide**: Best practices for prompting
