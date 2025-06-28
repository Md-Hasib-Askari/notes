# Transfer Learning

## Overview
Transfer learning leverages pre-trained models to solve new tasks with limited data by transferring knowledge from source domains to target domains.

## Core Concepts

### What is Transfer Learning?
- **Definition**: Using a model trained on one task as the starting point for a related task
- **Motivation**: Reduces training time, data requirements, and computational resources
- **Key Idea**: Lower layers learn general features, higher layers learn task-specific features

### Types of Transfer Learning

#### 1. Feature Extraction
```python
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load pre-trained model without top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
base_model.trainable = False

# Add custom classifier
model = tf.keras.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])
```

#### 2. Fine-tuning
```python
# After initial training with frozen base
base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all layers before fine_tune_at
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Use lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5/10),  # Lower LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

#### 3. Domain Adaptation
```python
# Gradual unfreezing strategy
def gradual_unfreezing(model, base_model, epochs_per_stage=5):
    stages = [
        (0, 0),      # All frozen
        (-50, -1),   # Last 50 layers
        (-100, -1),  # Last 100 layers
        (0, -1)      # All layers
    ]
    
    for start, end in stages:
        # Unfreeze layers
        for layer in base_model.layers[start:end]:
            layer.trainable = True
            
        # Train for specified epochs
        model.fit(train_data, epochs=epochs_per_stage)
```

## Popular Pre-trained Models

### Computer Vision
```python
# Image Classification
from tensorflow.keras.applications import (
    VGG16, VGG19,
    ResNet50, ResNet101,
    InceptionV3,
    MobileNetV2,
    EfficientNetB0
)

# Object Detection
# YOLO, R-CNN, SSD (using libraries like Detectron2)

# Semantic Segmentation
# U-Net, DeepLab (using segmentation_models library)
```

### Natural Language Processing
```python
# Using Hugging Face Transformers
from transformers import (
    AutoTokenizer, AutoModel,
    BertForSequenceClassification,
    GPT2LMHeadModel,
    T5ForConditionalGeneration
)

# BERT for classification
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', 
    num_labels=num_classes
)
```

## Implementation Strategies

### 1. Layer Freezing Strategy
```python
def selective_freeze(model, freeze_layers):
    """Freeze specific layers by name or index"""
    for i, layer in enumerate(model.layers):
        if i in freeze_layers or layer.name in freeze_layers:
            layer.trainable = False
        else:
            layer.trainable = True

# Example usage
selective_freeze(model, freeze_layers=['conv1', 'conv2', 0, 1, 2])
```

### 2. Learning Rate Scheduling
```python
def create_lr_schedule(base_lr, fine_tune_lr):
    """Different learning rates for different parts"""
    def lr_schedule(layer_name):
        if 'base_model' in layer_name:
            return fine_tune_lr  # Lower LR for pre-trained layers
        else:
            return base_lr       # Higher LR for new layers
    return lr_schedule

# Multi-optimizer approach
optimizer_new = tf.keras.optimizers.Adam(1e-3)
optimizer_pretrained = tf.keras.optimizers.Adam(1e-5)
```

### 3. Progressive Resizing
```python
def progressive_training(model, sizes=[224, 299, 380]):
    """Train with progressively larger image sizes"""
    for size in sizes:
        # Resize data
        train_data = preprocess_data(train_images, target_size=(size, size))
        
        # Adjust input shape if needed
        model.layers[0] = tf.keras.layers.Input(shape=(size, size, 3))
        
        # Train
        model.fit(train_data, epochs=10)
```

## Best Practices

### 1. Data Preprocessing
```python
# Match pre-trained model's preprocessing
from tensorflow.keras.applications.imagenet_utils import preprocess_input

def preprocess_for_pretrained(images, model_name='vgg16'):
    """Apply model-specific preprocessing"""
    if model_name in ['vgg16', 'vgg19', 'resnet']:
        return preprocess_input(images, mode='caffe')
    elif model_name in ['inception', 'xception']:
        return preprocess_input(images, mode='tf')
    else:
        return images / 255.0  # Generic normalization
```

### 2. Model Selection
```python
def select_backbone(task_type, data_size, computational_budget):
    """Choose appropriate pre-trained model"""
    recommendations = {
        'small_data': {
            'high_budget': 'EfficientNetB7',
            'medium_budget': 'ResNet50',
            'low_budget': 'MobileNetV2'
        },
        'medium_data': {
            'high_budget': 'EfficientNetB5',
            'medium_budget': 'InceptionV3',
            'low_budget': 'MobileNetV2'
        },
        'large_data': {
            'high_budget': 'Custom_Architecture',
            'medium_budget': 'ResNet101',
            'low_budget': 'ResNet50'
        }
    }
    return recommendations[data_size][computational_budget]
```

### 3. Evaluation and Monitoring
```python
def transfer_learning_pipeline(base_model, train_data, val_data):
    """Complete transfer learning pipeline"""
    
    # Phase 1: Feature extraction
    base_model.trainable = False
    model = build_classifier(base_model)
    
    # Train classifier
    history1 = model.fit(
        train_data, 
        validation_data=val_data,
        epochs=10,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3),
            tf.keras.callbacks.ReduceLROnPlateau()
        ]
    )
    
    # Phase 2: Fine-tuning
    base_model.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5))
    
    history2 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=10,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5)
        ]
    )
    
    return model, (history1, history2)
```

## Domain-Specific Applications

### Medical Imaging
```python
# RadImageNet for medical images
def medical_transfer_learning():
    base_model = tf.keras.applications.ResNet50(
        weights=None,  # Load RadImageNet weights separately
        include_top=False
    )
    # Load domain-specific pre-trained weights
    base_model.load_weights('radimagenet_resnet50_weights.h5')
```

### Satellite Imagery
```python
# Use models pre-trained on satellite data
def satellite_transfer_learning():
    # Models like BiDS or SatMAE
    from satellite_models import BiDS
    base_model = BiDS(weights='imagenet_satellite')
```

### Time Series
```python
# Transfer learning for time series
def time_series_transfer():
    # Pre-train on large time series dataset
    # Fine-tune on specific domain (finance, IoT, etc.)
    pass
```

## Common Pitfalls and Solutions

### 1. Feature Mismatch
```python
# Problem: Different input/output dimensions
# Solution: Adaptive layers
def adapt_model_architecture(pretrained_model, new_input_shape, new_output_size):
    # Modify input layer
    if pretrained_model.input_shape != new_input_shape:
        # Add resizing layer or modify input
        pass
    
    # Modify output layer
    pretrained_model.layers[-1] = Dense(new_output_size)
```

### 2. Overfitting
```python
# Solutions for overfitting in transfer learning
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3),
    tf.keras.callbacks.ModelCheckpoint(save_best_only=True)
]

# Add regularization
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.BatchNormalization())
```

### 3. Learning Rate Issues
```python
# Different learning rates for different parts
def multi_lr_training(model, base_layers, new_layers):
    # Lower LR for pre-trained layers
    base_optimizer = tf.keras.optimizers.Adam(1e-5)
    # Higher LR for new layers  
    new_optimizer = tf.keras.optimizers.Adam(1e-3)
```

## Evaluation Metrics

### Performance Comparison
```python
def compare_transfer_methods(data):
    """Compare different transfer learning approaches"""
    methods = {
        'from_scratch': train_from_scratch,
        'feature_extraction': feature_extraction,
        'fine_tuning': fine_tuning,
        'gradual_unfreezing': gradual_unfreezing
    }
    
    results = {}
    for name, method in methods.items():
        model = method(data)
        results[name] = evaluate_model(model, data)
    
    return results
```

### Domain Similarity Analysis
```python
def measure_domain_similarity(source_features, target_features):
    """Measure similarity between source and target domains"""
    from scipy.spatial.distance import cosine
    
    # Feature-based similarity
    similarity = 1 - cosine(source_features.mean(axis=0), 
                           target_features.mean(axis=0))
    
    # Distribution-based similarity (KL divergence, etc.)
    return similarity
```

## Advanced Techniques

### Meta-Learning for Transfer
```python
# Model-Agnostic Meta-Learning (MAML)
def meta_learning_transfer():
    """Learn to quickly adapt to new tasks"""
    # Implementation of MAML or similar meta-learning approaches
    pass
```

### Multi-Task Learning
```python
def multi_task_transfer(shared_layers, task_specific_heads):
    """Share representations across multiple related tasks"""
    shared_model = build_shared_model(shared_layers)
    
    task_models = {}
    for task_name, head_config in task_specific_heads.items():
        task_model = tf.keras.Sequential([
            shared_model,
            build_task_head(head_config)
        ])
        task_models[task_name] = task_model
    
    return task_models
```

## Key Takeaways

1. **Start with feature extraction**, then fine-tune if needed
2. **Use lower learning rates** for pre-trained layers
3. **Match preprocessing** to the original pre-trained model
4. **Consider domain similarity** when selecting pre-trained models
5. **Monitor for overfitting** especially with small datasets
6. **Gradual unfreezing** often works better than fine-tuning all layers at once
7. **Different learning rates** for different parts of the network
8. **Domain-specific pre-trained models** often outperform ImageNet models

## Resources
- **Hugging Face Model Hub**: Pre-trained models for NLP
- **TensorFlow Hub**: Pre-trained models for various domains
- **Papers With Code**: Latest transfer learning research
- **Model Zoos**: Domain-specific pre-trained models
