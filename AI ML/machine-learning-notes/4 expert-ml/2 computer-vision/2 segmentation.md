# Image Segmentation

## Overview
Image segmentation is a computer vision task that involves partitioning an image into multiple segments or regions, where each pixel is assigned a label. It provides pixel-level understanding of the image content.

## Types of Segmentation

### Semantic Segmentation
- **Definition**: Assigns each pixel a class label (e.g., car, road, sky)
- **Characteristic**: All instances of the same class get the same label
- **Output**: Single mask per class

### Instance Segmentation
- **Definition**: Identifies and segments individual object instances
- **Characteristic**: Different instances of the same class get different labels
- **Output**: Separate mask for each object instance

### Panoptic Segmentation
- **Definition**: Combines semantic and instance segmentation
- **Characteristic**: Every pixel gets both class and instance ID
- **Output**: Unified representation of stuff (background) and things (objects)

## Popular Architectures

### U-Net
```python
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        # Encoder (downsampling)
        self.enc1 = self.conv_block(n_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Decoder (upsampling)
        self.dec4 = self.conv_block(512 + 256, 256)
        self.dec3 = self.conv_block(256 + 128, 128)
        self.dec2 = self.conv_block(128 + 64, 64)
        self.dec1 = nn.Conv2d(64, n_classes, 1)
        
        self.pool = nn.MaxPool2d(2)
        self.upconv4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
    
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Decoder with skip connections
        dec4 = self.upconv4(enc4)
        dec4 = torch.cat([dec4, enc3], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc2], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc1], dim=1)
        dec2 = self.dec2(dec2)
        
        return self.dec1(dec2)
```

### DeepLab
- **Key Feature**: Atrous (dilated) convolutions for multi-scale context
- **ASPP**: Atrous Spatial Pyramid Pooling for capturing objects at multiple scales

```python
# DeepLabV3+ implementation example
import torchvision.models.segmentation as seg_models

model = seg_models.deeplabv3_resnet101(pretrained=True, num_classes=21)
model.eval()

# Inference
input_tensor = torch.rand(1, 3, 512, 512)
with torch.no_grad():
    output = model(input_tensor)
    predictions = output['out']
```

### Mask R-CNN (Instance Segmentation)
```python
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn

# Load pre-trained Mask R-CNN
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Inference
image = torch.rand(3, 800, 600)
predictions = model([image])

# Extract results
boxes = predictions[0]['boxes']
labels = predictions[0]['labels']
scores = predictions[0]['scores']
masks = predictions[0]['masks']  # Segmentation masks

print(f"Detected {len(boxes)} instances")
```

### Transformer-based Models
#### Segmenter (Vision Transformer for Segmentation)
```python
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image

processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

# Load and process image
image = Image.open("path/to/image.jpg")
inputs = processor(images=image, return_tensors="pt")

# Inference
outputs = model(**inputs)
logits = outputs.logits

# Resize to original image size
upsampled_logits = nn.functional.interpolate(
    logits,
    size=image.size[::-1],
    mode="bilinear",
    align_corners=False,
)

# Get predicted segmentation map
predicted_segmentation = upsampled_logits.argmax(dim=1)
```

## Loss Functions

### Cross-Entropy Loss
```python
import torch.nn as nn

# Standard cross-entropy for semantic segmentation
criterion = nn.CrossEntropyLoss()
loss = criterion(predictions, targets)
```

### Dice Loss
```python
def dice_loss(pred, target, smooth=1):
    """Dice loss for segmentation tasks"""
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()
```

### Focal Loss
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

## Evaluation Metrics

### IoU (Intersection over Union)
```python
def calculate_iou(pred_mask, true_mask):
    """Calculate IoU for binary masks"""
    intersection = torch.logical_and(pred_mask, true_mask).sum()
    union = torch.logical_or(pred_mask, true_mask).sum()
    iou = intersection.float() / union.float()
    return iou

def mean_iou(pred_masks, true_masks, num_classes):
    """Calculate mean IoU across all classes"""
    ious = []
    for i in range(num_classes):
        pred_i = (pred_masks == i)
        true_i = (true_masks == i)
        iou = calculate_iou(pred_i, true_i)
        ious.append(iou)
    return torch.stack(ious).mean()
```

### Pixel Accuracy
```python
def pixel_accuracy(pred, target):
    """Calculate pixel-wise accuracy"""
    correct = (pred == target).sum()
    total = target.numel()
    return correct.float() / total.float()
```

## Data Preprocessing

### Data Augmentation
```python
import albumentations as A

# Segmentation-specific augmentation
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.5),
    A.ElasticTransform(p=0.3),
    A.GridDistortion(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
], additional_targets={'mask': 'mask'})

# Apply transformation
transformed = transform(image=image, mask=mask)
augmented_image = transformed['image']
augmented_mask = transformed['mask']
```

### Data Loading
```python
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        
        image = np.array(image)
        mask = np.array(mask)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return torch.FloatTensor(image), torch.LongTensor(mask)

# Create data loader
dataset = SegmentationDataset(image_paths, mask_paths, transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
```

## Training Pipeline
```python
import torch.optim as optim

# Model, loss, and optimizer
model = UNet(n_channels=3, n_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (images, masks) in enumerate(dataloader):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}')
```

## Applications

### Medical Imaging
- **Organ Segmentation**: Heart, liver, brain structures
- **Tumor Detection**: Cancer identification and boundary delineation
- **Pathology**: Cell and tissue analysis

### Autonomous Vehicles
- **Road Segmentation**: Drivable area identification
- **Lane Detection**: Lane boundary segmentation
- **Traffic Sign Recognition**: Sign boundary detection

### Satellite Imaging
- **Land Use Classification**: Urban, forest, agricultural areas
- **Environmental Monitoring**: Deforestation, urban growth
- **Disaster Assessment**: Flood and damage mapping

### Industrial Applications
- **Quality Control**: Defect detection in manufacturing
- **Agriculture**: Crop monitoring and disease detection
- **Robotics**: Object manipulation and navigation

## Popular Datasets
- **COCO**: Instance segmentation with 80 categories
- **Cityscapes**: Urban street scenes for autonomous driving
- **Pascal VOC**: 20 object categories with segmentation masks
- **ADE20K**: Scene parsing with 150 semantic categories
- **Medical**: Various medical imaging datasets (ISBI, BraTS, etc.)

## Best Practices
1. **Start with pre-trained models** and fine-tune on your dataset
2. **Use appropriate data augmentation** that preserves mask-image correspondence
3. **Balance your dataset** across different classes and object sizes
4. **Experiment with loss functions** based on class imbalance
5. **Use multi-scale training** for better performance across object sizes
6. **Monitor validation metrics** to prevent overfitting
7. **Consider ensemble methods** for improved accuracy

## Common Challenges
- **Class Imbalance**: Unequal pixel distribution across classes
- **Small Objects**: Tiny objects that are hard to segment
- **Boundary Accuracy**: Precise object boundaries
- **Real-time Requirements**: Fast inference for applications
- **Limited Annotations**: Expensive pixel-level labeling

## Tools and Frameworks
- **Segmentation Models Pytorch**: Pre-trained segmentation models
- **MMSegmentation**: OpenMMLab's segmentation toolbox
- **TorchVision**: PyTorch's computer vision library
- **Detectron2**: Facebook's detection and segmentation platform
- **Labelme**: Annotation tool for segmentation masks

## Future Directions
- **Vision Transformers**: Transformer-based segmentation models
- **Few-shot Segmentation**: Learning with limited labeled data
- **Weakly Supervised**: Learning from image-level labels
- **3D Segmentation**: Extending to volumetric data
- **Real-time Segmentation**: Efficient models for mobile deployment

## Resources
- **Papers**: "U-Net", "DeepLab", "Mask R-CNN", "Segmenter"
- **Datasets**: COCO, Cityscapes, Pascal VOC, ADE20K
- **Libraries**: PyTorch, TensorFlow, OpenCV
- **Annotation Tools**: Labelme, CVAT, Supervisely
- **Benchmarks**: COCO Panoptic Challenge, Cityscapes Benchmark
