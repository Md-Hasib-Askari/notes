# Week 11: Modern YOLO Versions

## YOLOv4 & v5 Features

### CSPDarknet53 Backbone
**Cross Stage Partial Networks (CSP)**:
```python
class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super().__init__()
        mid_channels = out_channels // 2
        
        # Split input into two paths
        self.split_conv = nn.Conv2d(in_channels, mid_channels, 1)
        self.blocks_conv = nn.Conv2d(in_channels, mid_channels, 1)
        
        # Residual blocks on one path
        self.blocks = nn.Sequential(*[
            ResidualBlock(mid_channels) for _ in range(num_blocks)
        ])
        
        # Concatenate and final conv
        self.concat_conv = nn.Conv2d(out_channels, out_channels, 1)
    
    def forward(self, x):
        # Split computation
        split_out = self.split_conv(x)
        blocks_out = self.blocks(self.blocks_conv(x))
        
        # Concatenate and process
        concat = torch.cat([split_out, blocks_out], dim=1)
        return self.concat_conv(concat)
```

**Benefits**:
- Reduced computational complexity
- Better gradient flow
- Improved inference speed
- Maintained accuracy

### SPP and PANet Components
**Spatial Pyramid Pooling (SPP)**:
```python
class SPPBlock(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super().__init__()
        self.pools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=size, stride=1, padding=size//2)
            for size in pool_sizes
        ])
    
    def forward(self, x):
        outputs = [x]  # Original feature map
        for pool in self.pools:
            outputs.append(pool(x))
        return torch.cat(outputs, dim=1)
```

**Path Aggregation Network (PANet)**:
```python
class PANet(nn.Module):
    def __init__(self):
        super().__init__()
        # Bottom-up path (FPN)
        self.fpn_conv1 = ConvBNReLU(512, 256, 1)
        self.fpn_conv2 = ConvBNReLU(1024, 512, 1)
        
        # Top-down path
        self.pan_conv1 = ConvBNReLU(256, 256, 3, stride=2, padding=1)
        self.pan_conv2 = ConvBNReLU(512, 512, 3, stride=2, padding=1)
    
    def forward(self, features):
        c3, c4, c5 = features  # Multi-scale features
        
        # FPN path (top-down)
        p5 = self.fpn_conv2(c5)
        p4 = self.fpn_conv1(c4) + F.interpolate(p5, scale_factor=2)
        p3 = c3 + F.interpolate(p4, scale_factor=2)
        
        # PAN path (bottom-up)
        n3 = p3
        n4 = p4 + self.pan_conv1(n3)
        n5 = p5 + self.pan_conv2(n4)
        
        return n3, n4, n5
```

### Advanced Data Augmentation
**Mosaic Augmentation**:
```python
def mosaic_augmentation(images, labels, prob=0.5):
    if random.random() < prob:
        # Select 4 images
        indices = random.sample(range(len(images)), 4)
        
        # Create mosaic
        h, w = images[0].shape[:2]
        mosaic = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        
        # Place images in quadrants
        mosaic[:h, :w] = images[indices[0]]      # Top-left
        mosaic[:h, w:] = images[indices[1]]      # Top-right  
        mosaic[h:, :w] = images[indices[2]]      # Bottom-left
        mosaic[h:, w:] = images[indices[3]]      # Bottom-right
        
        # Adjust labels accordingly
        adjusted_labels = adjust_mosaic_labels(labels, indices)
        return mosaic, adjusted_labels
    
    return images, labels
```

**Mixup Augmentation**:
```python
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam
```

### Training Optimizations
**Advanced Learning Rate Scheduling**:
```python
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
```

## YOLOv6, v7, v8 Innovations

### Efficiency Improvements
**YOLOv6 Optimizations**:
```python
# Efficient backbone design
class EfficientRep(nn.Module):
    def __init__(self, in_channels, out_channels, n=1):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, 3, padding=1)
        self.conv2 = ConvBNReLU(out_channels, out_channels, 3, padding=1)
        self.blocks = nn.Sequential(*[
            RepBlock(out_channels) for _ in range(n)
        ])
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.blocks(x)

# Re-parameterization technique
class RepBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv3x3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv1x1 = nn.Conv2d(channels, channels, 1)
        self.identity = nn.BatchNorm2d(channels) if channels else None
    
    def forward(self, x):
        if self.training:
            return self.conv3x3(x) + self.conv1x1(x) + (self.identity(x) if self.identity else 0)
        else:
            # Fused convolution during inference
            return self.fused_conv(x)
```

**YOLOv7 Extended Efficient Layer Aggregation (ELAN)**:
```python
class ELAN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 4
        
        self.conv1 = ConvBNReLU(in_channels, mid_channels, 1)
        self.conv2 = ConvBNReLU(in_channels, mid_channels, 1)
        self.conv3 = ConvBNReLU(mid_channels, mid_channels, 3, padding=1)
        self.conv4 = ConvBNReLU(mid_channels, mid_channels, 3, padding=1)
        self.conv5 = ConvBNReLU(mid_channels*4, out_channels, 1)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        
        concat = torch.cat([x1, x2, x3, x4], dim=1)
        return self.conv5(concat)
```

### Deployment Optimizations
**TensorRT Integration**:
```python
import tensorrt as trt

def build_tensorrt_engine(onnx_path, engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX model
    with open(onnx_path, 'rb') as model:
        parser.parse(model.read())
    
    # Build engine
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    config.set_flag(trt.BuilderFlag.FP16)  # FP16 precision
    
    engine = builder.build_engine(network, config)
    
    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
```

### User-Friendly Interfaces
**YOLOv8 Unified API**:
```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')  # nano, small, medium, large, extra-large

# Training
model.train(
    data='coco128.yaml',
    epochs=100,
    imgsz=640,
    batch_size=16,
    device='gpu'
)

# Validation
metrics = model.val()

# Prediction
results = model.predict('image.jpg')

# Export
model.export(format='onnx')  # ONNX, TensorRT, CoreML, etc.
```

## Practical Implementation

### Using Ultralytics YOLOv8
**Installation and Setup**:
```bash
pip install ultralytics

# CLI usage
yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640
yolo detect predict model=yolov8n.pt source='image.jpg'
yolo detect val model=yolov8n.pt data=coco128.yaml
```

### Command-Line Training
**Basic Training Command**:
```bash
# Train from scratch
yolo detect train data=custom_dataset.yaml model=yolov8n.yaml epochs=300

# Transfer learning
yolo detect train data=custom_dataset.yaml model=yolov8n.pt epochs=100

# Resume training
yolo detect train resume model=last.pt

# Multi-GPU training
yolo detect train data=dataset.yaml model=yolov8n.pt device=0,1,2,3
```

**Advanced Training Options**:
```bash
yolo detect train \
    data=dataset.yaml \
    model=yolov8n.pt \
    epochs=300 \
    patience=50 \
    batch=16 \
    imgsz=640 \
    save=True \
    save_period=10 \
    cache=True \
    device=0 \
    workers=8 \
    project=my_project \
    name=experiment_1 \
    exist_ok=True \
    pretrained=True \
    optimizer=SGD \
    verbose=True \
    seed=0 \
    deterministic=True \
    single_cls=False \
    rect=False \
    cos_lr=False \
    close_mosaic=10 \
    resume=False \
    amp=True \
    fraction=1.0 \
    profile=False \
    overlap_mask=True \
    mask_ratio=4 \
    dropout=0.0 \
    val=True
```

### Python API Usage
**Complete Training Pipeline**:
```python
from ultralytics import YOLO
import yaml

# Custom dataset configuration
dataset_config = {
    'path': '/path/to/dataset',
    'train': 'images/train',
    'val': 'images/val', 
    'test': 'images/test',
    'nc': 80,  # number of classes
    'names': ['person', 'bicycle', 'car', ...]  # class names
}

# Save dataset config
with open('custom_dataset.yaml', 'w') as f:
    yaml.dump(dataset_config, f)

# Initialize model
model = YOLO('yolov8n.pt')

# Custom training parameters
train_results = model.train(
    data='custom_dataset.yaml',
    epochs=300,
    imgsz=640,
    batch_size=16,
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    pose=12.0,
    kobj=2.0,
    label_smoothing=0.0,
    nbs=64,
    overlap_mask=True,
    mask_ratio=4,
    dropout=0.0,
    val=True,
    save=True,
    save_period=-1,
    cache=False,
    device=None,
    workers=8,
    project=None,
    name=None,
    exist_ok=False,
    pretrained=True,
    optimizer='SGD',
    verbose=True,
    seed=0,
    deterministic=True,
    single_cls=False,
    rect=False,
    cos_lr=False,
    close_mosaic=10,
    resume=False,
    amp=True,
    fraction=1.0,
    profile=False
)

# Validation
val_results = model.val()

# Inference
results = model('test_image.jpg')
for r in results:
    # Plot results
    im_array = r.plot()
    
    # Get bounding boxes
    boxes = r.boxes.xywh.cpu()
    
    # Get confidence scores
    conf = r.boxes.conf.cpu()
    
    # Get class predictions
    cls = r.boxes.cls.cpu()

# Export model
model.export(format='onnx', dynamic=True, simplify=True)
```

## Performance Comparison

### Model Variants Comparison
```python
models_comparison = {
    'YOLOv8n': {'params': '3.2M', 'flops': '8.7B', 'mAP50-95': '37.3', 'speed': '0.99ms'},
    'YOLOv8s': {'params': '11.2M', 'flops': '28.6B', 'mAP50-95': '44.9', 'speed': '1.20ms'},
    'YOLOv8m': {'params': '25.9M', 'flops': '78.9B', 'mAP50-95': '50.2', 'speed': '1.83ms'},
    'YOLOv8l': {'params': '43.7M', 'flops': '165.2B', 'mAP50-95': '52.9', 'speed': '2.39ms'},
    'YOLOv8x': {'params': '68.2M', 'flops': '257.8B', 'mAP50-95': '53.9', 'speed': '3.53ms'},
}
```

### Version Evolution Timeline
```
2020: YOLOv4 - CSPDarknet53, SPP, PANet
2020: YOLOv5 - PyTorch implementation, user-friendly
2022: YOLOv6 - Efficient architecture, industrial applications  
2022: YOLOv7 - ELAN, compound scaling
2023: YOLOv8 - Unified framework, improved architecture
```

## Key Innovations Summary

### Architecture Improvements
1. **CSP Networks**: Better gradient flow and efficiency
2. **Advanced Necks**: SPP, PANet for better feature fusion
3. **Efficient Blocks**: RepBlocks, ELAN for speed-accuracy balance
4. **Multi-Scale Design**: Better detection across object sizes

### Training Enhancements
1. **Advanced Augmentations**: Mosaic, Mixup, CutMix
2. **Better Schedulers**: Cosine annealing, warmup strategies
3. **Loss Improvements**: Focal loss, IoU-based losses
4. **Optimization**: Better convergence and stability

### Deployment Features
1. **Export Formats**: ONNX, TensorRT, CoreML, TFLite
2. **Quantization**: FP16, INT8 support
3. **Hardware Optimization**: GPU, TPU, mobile optimization
4. **Easy Integration**: Unified APIs and interfaces

## Best Practices

### Model Selection Guide
```python
def select_yolo_model(requirements):
    if requirements['speed'] == 'highest':
        return 'YOLOv8n'
    elif requirements['accuracy'] == 'highest':
        return 'YOLOv8x'
    elif requirements['balanced']:
        return 'YOLOv8m'
    elif requirements['mobile']:
        return 'YOLOv8n'
    elif requirements['edge_device']:
        return 'YOLOv8s'
```

### Training Tips
1. **Start Small**: Begin with YOLOv8n for prototyping
2. **Gradual Scaling**: Increase model size based on performance needs
3. **Data Quality**: Focus on high-quality annotations
4. **Hyperparameter Tuning**: Use grid search or automated tools
5. **Monitoring**: Track training metrics and validation performance
