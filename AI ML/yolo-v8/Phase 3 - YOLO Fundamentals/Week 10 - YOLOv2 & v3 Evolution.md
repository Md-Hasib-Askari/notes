# Week 10: YOLOv2 & v3 Evolution

## YOLOv2 Improvements

### Anchor Boxes Introduction
**Motivation**: Address YOLOv1's limitation with multiple objects and varying aspect ratios

```python
# Anchor box concept
anchor_boxes = [
    (1.3221, 1.73145),    # Tall and narrow
    (3.19275, 4.00944),   # Large
    (5.05587, 8.09892),   # Very large
    (9.47112, 4.84053),   # Wide
    (11.2364, 10.0071)    # Very wide and tall
]

# Each grid cell now predicts using anchor boxes
output_shape = (grid_size, grid_size, num_anchors * (5 + num_classes))
```

**Key Changes**:
- Removed fully connected layers
- Each cell predicts multiple bounding boxes using anchor boxes
- Anchor boxes learned through k-means clustering on training data
- Improved recall from 81% to 88%

### Multi-Scale Training
**Dynamic Input Sizes**: Train with different input resolutions
```python
# Multi-scale training implementation
input_sizes = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]

def multiscale_training():
    for epoch in range(num_epochs):
        if epoch % 10 == 0:  # Change size every 10 batches
            new_size = random.choice(input_sizes)
            model.input_size = new_size
```

**Benefits**:
- Better performance across different image scales
- More robust to varying input sizes
- Trade-off between speed and accuracy

### Batch Normalization Impact
**Implementation**: Added batch normalization to all convolutional layers
```python
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
```

**Results**:
- 2% mAP improvement
- Faster convergence
- Better generalization
- Eliminated need for dropout

### Darknet-19 Architecture
**Backbone Evolution**: From custom CNN to Darknet-19
```
Darknet-19 Architecture:
├── 19 convolutional layers
├── 5 max pooling layers
├── Global average pooling
└── 1000-class classification head

Key Features:
├── 3×3 and 1×1 convolutions
├── Batch normalization after each conv
├── ReLU activations
└── No fully connected layers except final
```

**Advantages**:
- Fewer parameters than VGG-16
- Better feature extraction
- More efficient computation

## YOLOv3 Advancements

### Multi-Scale Detection
**Three Detection Scales**: Detect objects at different scales simultaneously
```python
# YOLOv3 multi-scale detection
detection_scales = [
    (13, 13),   # Large objects
    (26, 26),   # Medium objects  
    (52, 52)    # Small objects
]

# Anchor boxes for each scale
scale_anchors = [
    [(116,90), (156,198), (373,326)],      # Scale 1 (13×13)
    [(30,61), (62,45), (59,119)],          # Scale 2 (26×26)
    [(10,13), (16,30), (33,23)]            # Scale 3 (52×52)
]
```

**Feature Pyramid Network (FPN) Concept**:
- Detect at multiple feature map resolutions
- Combine high-level semantic features with low-level detail
- Better performance on small objects

### Feature Pyramid Networks
**Upsampling and Concatenation**:
```python
class YOLOv3Neck(nn.Module):
    def forward(self, features):
        # Features from different backbone stages
        f1, f2, f3 = features  # 52×52, 26×26, 13×13
        
        # Detection at largest scale (13×13)
        out1 = self.detect_large(f3)
        
        # Upsample and concatenate for medium scale
        up1 = self.upsample(f3)
        concat1 = torch.cat([up1, f2], dim=1)
        out2 = self.detect_medium(concat1)
        
        # Upsample and concatenate for small scale  
        up2 = self.upsample(concat1)
        concat2 = torch.cat([up2, f1], dim=1)
        out3 = self.detect_small(concat2)
        
        return out1, out2, out3
```

### Residual Connections
**Darknet-53 Backbone**: Incorporated residual connections
```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBNReLU(channels, channels//2, 1)
        self.conv2 = ConvBNReLU(channels//2, channels, 3, padding=1)
    
    def forward(self, x):
        residual = x
        out = self.conv2(self.conv1(x))
        return out + residual  # Skip connection
```

**Benefits**:
- Deeper networks (53 layers vs 19)
- Better gradient flow
- Improved feature learning
- Higher accuracy

### Darknet-53 Architecture
```
Darknet-53 Structure:
├── Initial Conv: 32 filters, 3×3
├── Residual Block 1: 64 filters, ×1
├── Residual Block 2: 128 filters, ×2  
├── Residual Block 3: 256 filters, ×8
├── Residual Block 4: 512 filters, ×8
├── Residual Block 5: 1024 filters, ×4
└── Feature extraction at multiple scales

Total: 53 convolutional layers
```

## Comparative Analysis

### Performance Improvements Across Versions
```
Performance Metrics (COCO dataset):
┌─────────┬────────┬─────────┬────────────┐
│ Version │   mAP  │   FPS   │ Parameters │
├─────────┼────────┼─────────┼────────────┤
│ YOLOv1  │  21.6  │   45    │     -      │
│ YOLOv2  │  21.6  │   40    │    67M     │
│ YOLOv3  │  31.0  │   20    │    65M     │
└─────────┴────────┴─────────┴────────────┘
```

### Computational Cost Analysis
**YOLOv2 vs YOLOv3**:
```python
# Computational complexity comparison
yolov2_flops = calculate_flops(darknet19, input_size=416)  # ~8.5 BFLOPs
yolov3_flops = calculate_flops(darknet53, input_size=416)  # ~65.9 BFLOPs

# Memory usage
yolov2_memory = estimate_memory(yolov2_model)  # ~256 MB
yolov3_memory = estimate_memory(yolov3_model)  # ~512 MB
```

### Use Case Suitability
**YOLOv2 Best For**:
- Real-time applications with limited compute
- Embedded systems
- Applications where speed > accuracy

**YOLOv3 Best For**:
- Higher accuracy requirements
- Small object detection
- Multi-class detection scenarios
- Sufficient computational resources

## Implementation Details

### Loss Function Evolution
**YOLOv2 Loss**:
```python
def yolov2_loss(predictions, targets, anchors):
    # Object/No-object loss
    obj_loss = binary_crossentropy(pred_conf, target_conf)
    
    # Coordinate loss (with anchor box scaling)
    coord_loss = mse_loss(pred_boxes, target_boxes) * coord_scale
    
    # Classification loss
    class_loss = binary_crossentropy(pred_classes, target_classes)
    
    return obj_loss + coord_loss + class_loss
```

**YOLOv3 Loss**:
```python
def yolov3_loss(predictions, targets):
    total_loss = 0
    
    # Loop through each detection scale
    for scale_pred, scale_target in zip(predictions, targets):
        # Binary cross-entropy for objectness
        obj_loss = bce_loss(scale_pred[..., 4], scale_target[..., 4])
        
        # Binary cross-entropy for classification (multi-label)
        class_loss = bce_loss(scale_pred[..., 5:], scale_target[..., 5:])
        
        # Coordinate regression loss
        coord_loss = mse_loss(scale_pred[..., :4], scale_target[..., :4])
        
        total_loss += obj_loss + class_loss + coord_loss
    
    return total_loss
```

### Data Augmentation
**YOLOv2/v3 Augmentations**:
```python
augmentations = [
    RandomHorizontalFlip(p=0.5),
    RandomScale(scale_range=(0.8, 1.2)),
    RandomTranslate(translate_range=0.1),
    RandomRotate(angle_range=(-5, 5)),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    Mosaic(p=0.5),  # Introduced in later versions
]
```

## Training Strategies

### Transfer Learning
```python
# Pre-training on ImageNet classification
darknet53_classifier = Darknet53(num_classes=1000)
train_classifier(darknet53_classifier, imagenet_data)

# Transfer to detection
yolov3_detector = YOLOv3(backbone=darknet53_classifier.backbone)
train_detector(yolov3_detector, detection_data)
```

### Learning Rate Scheduling
```python
# YOLOv3 training schedule
initial_lr = 0.001
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Warmup strategy
warmup_epochs = 5
for epoch in range(warmup_epochs):
    lr = initial_lr * (epoch + 1) / warmup_epochs
    set_learning_rate(optimizer, lr)
```

## Key Innovations Summary

### YOLOv2 Contributions
1. **Anchor Boxes**: Better handling of object shapes
2. **Multi-Scale Training**: Robust to different input sizes
3. **Batch Normalization**: Improved training stability
4. **Direct Location Prediction**: Better bounding box regression

### YOLOv3 Contributions
1. **Multi-Scale Detection**: Better small object detection
2. **Feature Pyramid Networks**: Rich feature representation
3. **Residual Connections**: Deeper, more powerful networks
4. **Binary Classification**: Multi-label capability

## Practical Considerations

### Model Selection Criteria
```python
def choose_yolo_version(requirements):
    if requirements.speed > requirements.accuracy:
        return "YOLOv2"
    elif requirements.small_objects:
        return "YOLOv3"
    elif requirements.embedded_device:
        return "YOLOv2-tiny"
    else:
        return "YOLOv3"
```

### Deployment Considerations
- **YOLOv2**: Better for edge devices, mobile applications
- **YOLOv3**: Better for server-side applications, high-accuracy needs
- **Memory vs Accuracy**: Choose based on resource constraints
