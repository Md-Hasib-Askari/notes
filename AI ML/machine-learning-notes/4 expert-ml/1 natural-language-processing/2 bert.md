# BERT (Bidirectional Encoder Representations from Transformers)

## Overview
BERT is a pre-trained transformer model that uses bidirectional training to understand context from both left and right directions. It revolutionized NLP by achieving state-of-the-art results on many tasks.

## Key Features
- **Bidirectional Context**: Reads text in both directions simultaneously
- **Pre-training + Fine-tuning**: Two-stage training approach
- **Masked Language Modeling**: Predicts masked words in sentences
- **Next Sentence Prediction**: Understands relationships between sentences

## Architecture
- **Encoder-only**: Uses only the encoder part of the transformer
- **12/24 Layers**: BERT-Base (12 layers) or BERT-Large (24 layers)
- **Multi-Head Attention**: 12/16 attention heads
- **Hidden Size**: 768 (Base) or 1024 (Large) dimensions

## Pre-training Tasks
### Masked Language Modeling (MLM)
```python
# Example: 15% of tokens are masked
Input: "The cat [MASK] on the mat"
Target: "The cat sat on the mat"
```

### Next Sentence Prediction (NSP)
```python
# Predicts if sentence B follows sentence A
Sentence A: "The man went to the store."
Sentence B: "He bought milk." (IsNext = True)
```

## Implementation Example
```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize text
text = "This movie is amazing!"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

# Fine-tune for classification
outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
```

## Fine-tuning for Tasks
- **Text Classification**: Sentiment analysis, spam detection
- **Named Entity Recognition**: Identify entities in text
- **Question Answering**: Reading comprehension tasks
- **Text Similarity**: Semantic similarity between texts

## Variants
- **RoBERTa**: Robustly optimized BERT (removes NSP, longer training)
- **DistilBERT**: Smaller, faster version (40% fewer parameters)
- **ALBERT**: A Lite BERT (parameter sharing, factorized embeddings)
- **DeBERTa**: Decoupled attention (improved performance)

## Advantages
- **Strong Performance**: Excellent results on many NLP benchmarks
- **Transfer Learning**: Pre-trained representations work well across tasks
- **Bidirectional Context**: Better understanding than unidirectional models

## Disadvantages
- **Computational Cost**: Large model requires significant resources
- **Fixed Input Length**: Limited to 512 tokens
- **No Text Generation**: Encoder-only, not suitable for generative tasks

## Best Practices
1. **Start with pre-trained models** and fine-tune for your task
2. **Use appropriate tokenization** and handle special tokens correctly
3. **Experiment with learning rates** (typically 2e-5 to 5e-5)
4. **Use gradient clipping** to prevent exploding gradients
5. **Monitor validation performance** to avoid overfitting

## Applications
- **Search Engines**: Improve search relevance
- **Chatbots**: Better intent understanding
- **Document Analysis**: Content classification and extraction
- **Sentiment Analysis**: Social media monitoring
- **Content Moderation**: Automated text filtering

## Resources
- **Original Paper**: "BERT: Pre-training of Deep Bidirectional Transformers"
- **Hugging Face**: Pre-trained models and tutorials
- **Google Research**: Official BERT repository
- **GLUE Benchmark**: Evaluation tasks for language understanding
