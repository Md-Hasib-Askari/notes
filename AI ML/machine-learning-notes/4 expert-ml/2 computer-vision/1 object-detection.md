# Object Detection

## Overview
Object detection is a computer vision task that involves identifying and localizing objects within images or videos. It combines classification (what objects are present) with localization (where they are located).

## Key Concepts
- **Bounding Box**: Rectangular box that encloses an object (x, y, width, height)
- **Classification**: Identifying the class/category of detected objects
- **Localization**: Determining the spatial location of objects
- **IoU (Intersection over Union)**: Metric to evaluate bounding box overlap
- **Non-Maximum Suppression (NMS)**: Technique to remove duplicate detections

## Popular Architectures

### Two-Stage Detectors
#### R-CNN Family
- **R-CNN**: Region-based CNN (slow, requires external region proposals)
- **Fast R-CNN**: Faster training and inference with RoI pooling
- **Faster R-CNN**: End-to-end with Region Proposal Network (RPN)

#### Implementation Example (Faster R-CNN)
```python
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load pre-trained model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Prepare image
image = torch.rand(3, 800, 600)  # Random image tensor
predictions = model([image])

# Extract results
boxes = predictions[0]['boxes']
labels = predictions[0]['labels']
scores = predictions[0]['scores']

print(f"Detected {len(boxes)} objects")
```

### One-Stage Detectors
#### YOLO (You Only Look Once)
- **YOLOv1-v8**: Evolution of real-time object detection
- **Key Advantage**: Fast inference, single forward pass
- **Grid-based**: Divides image into grid cells

#### Implementation Example (YOLOv5)
```python
import torch

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Inference
results = model('path/to/image.jpg')

# Display results
results.show()  # Display image with detections
results.print()  # Print detection info

# Access results
detections = results.pandas().xyxy[0]  # Pandas DataFrame
print(detections)
```

#### SSD (Single Shot Detector)
- **Multi-scale Detection**: Uses feature maps of different scales
- **Default Boxes**: Pre-defined anchor boxes at different scales

### Modern Architectures
#### DETR (Detection Transformer)
```python
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Load and process image
image = Image.open("path/to/image.jpg")
inputs = processor(images=image, return_tensors="pt")

# Inference
outputs = model(**inputs)

# Post-process
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs, target_sizes=target_sizes, threshold=0.9
)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(f"Detected {model.config.id2label[label.item()]} with confidence {round(score.item(), 3)} at location {box}")
```

## Evaluation Metrics

### Average Precision (AP)
- **mAP**: Mean Average Precision across all classes
- **AP@IoU**: AP at specific IoU thresholds (e.g., AP@0.5, AP@0.75)
- **COCO Metrics**: AP@[0.5:0.95] (average over IoU thresholds 0.5 to 0.95)

### Precision and Recall
```python
# Precision = TP / (TP + FP)
# Recall = TP / (TP + FN)
# F1-Score = 2 * (Precision * Recall) / (Precision + Recall)

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    
    # Calculate intersection
    xi1, yi1 = max(x1, x1_p), max(y1, y1_p)
    xi2, yi2 = min(x2, x2_p), min(y2, y2_p)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    
    intersection = (xi2 - xi1) * (yi2 - yi1)
    
    # Calculate union
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_p - x1_p) * (y2_p - y1_p)
    union = area1 + area2 - intersection
    
    return intersection / union
```

## Data Preprocessing

### Data Augmentation
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=15, p=0.5),
    A.Resize(640, 640),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Apply transformation
transformed = transform(image=image, bboxes=bboxes, class_labels=labels)
```

### Annotation Formats
- **PASCAL VOC**: XML format with bounding box coordinates
- **COCO**: JSON format with extensive metadata
- **YOLO**: Text files with normalized coordinates

## Training Strategies

### Transfer Learning
```python
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load pre-trained model
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Modify for custom number of classes
num_classes = 10  # Including background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features, num_classes
)
```

### Loss Functions
- **Classification Loss**: Cross-entropy for object classes
- **Localization Loss**: L1 or Smooth L1 loss for bounding box regression
- **Objectness Loss**: Binary classification for object presence

## Applications
- **Autonomous Vehicles**: Pedestrian and vehicle detection
- **Surveillance**: Security monitoring and threat detection
- **Medical Imaging**: Tumor and anomaly detection
- **Retail**: Inventory management and checkout automation
- **Sports Analytics**: Player and ball tracking
- **Manufacturing**: Quality control and defect detection

## Popular Datasets
- **COCO**: 80 object categories, 330K images
- **Pascal VOC**: 20 object categories, detection and segmentation
- **Open Images**: 600 categories, millions of images
- **ImageNet**: Object detection challenge dataset

## Best Practices
1. **Start with pre-trained models** and fine-tune on your dataset
2. **Use appropriate data augmentation** to improve generalization
3. **Balance your dataset** across different object classes
4. **Tune hyperparameters** like learning rate, IoU thresholds, and NMS
5. **Monitor validation metrics** to prevent overfitting
6. **Use multi-scale training** for better performance across object sizes

## Common Challenges
- **Small Object Detection**: Objects that occupy few pixels
- **Occlusion**: Partially hidden objects
- **Class Imbalance**: Unequal distribution of object classes
- **Real-time Requirements**: Balancing speed vs accuracy
- **Domain Adaptation**: Models trained on one domain applied to another

## Tools and Frameworks
- **YOLOv5/YOLOv8**: Ultralytics implementation
- **Detectron2**: Facebook's detection platform
- **MMDetection**: OpenMMLab's detection toolbox
- **TensorFlow Object Detection API**: Google's framework
- **Roboflow**: Data management and annotation platform

## Future Directions
- **Vision Transformers**: DETR and transformer-based detection
- **3D Object Detection**: Extending to 3D space
- **Video Object Detection**: Temporal consistency in videos
- **Few-shot Detection**: Learning with limited labeled data
- **Edge Deployment**: Efficient models for mobile and embedded devices

## Resources
- **Papers**: "You Only Look Once", "Faster R-CNN", "DETR"
- **Datasets**: COCO, Pascal VOC, Open Images
- **Competitions**: COCO Detection Challenge
- **Libraries**: PyTorch, TensorFlow, OpenCV
- **Annotation Tools**: LabelImg, Roboflow, VGG Image Annotator
