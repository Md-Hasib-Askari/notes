# Exercises: YOLO Practical Implementation

## Exercise 1: Load YOLOv5 and Detect Objects in Images/Videos

### Objective
Learn to use pre-trained YOLOv5 models for object detection on various media types including images, videos, and real-time webcam feeds.

### Prerequisites
- Python 3.7+ installed
- Basic understanding of Python programming
- Familiarity with command line/terminal usage

### Step-by-Step Implementation

#### Setup Environment
```bash
# Create and activate virtual environment
python -m venv yolo_env
source yolo_env/bin/activate  # On Windows: yolo_env\Scripts\activate

# Install required packages
pip install torch torchvision torchaudio
pip install ultralytics
pip install opencv-python
pip install matplotlib
```

#### Method 1: Using PyTorch Hub (Recommended for Beginners)
```python
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Load pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set model to evaluation mode
model.eval()

# Option 1: Detect objects in a single image
def detect_image(image_path):
    # Load image
    img = Image.open(image_path)
    
    # Run inference
    results = model(img)
    
    # Display results
    results.show()  # Shows image with bounding boxes
    
    # Get detection data
    detections = results.pandas().xyxy[0]  # pandas DataFrame
    print(detections)
    
    # Save results
    results.save('runs/detect/exp')

# Option 2: Detect objects in video
def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB (YOLOv5 expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = model(frame_rgb)
        
        # Convert results back to BGR for OpenCV display
        annotated_frame = results.render()[0]
        frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        
        # Display frame
        cv2.imshow('YOLOv5 Detection', frame_bgr)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Option 3: Real-time webcam detection
def detect_webcam():
    cap = cv2.VideoCapture(0)  # 0 for default camera
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = model(frame_rgb)
        
        # Render results
        annotated_frame = results.render()[0]
        frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        
        # Display frame
        cv2.imshow('Webcam YOLOv5', frame_bgr)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # Test with image
    detect_image('path/to/your/image.jpg')
    
    # Test with video
    detect_video('path/to/your/video.mp4')
    
    # Test with webcam
    detect_webcam()
```

#### Method 2: Using Command Line Interface
```bash
# Clone YOLOv5 repository
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt

# Detect objects in image
python detect.py --source path/to/image.jpg --weights yolov5s.pt

# Detect objects in video
python detect.py --source path/to/video.mp4 --weights yolov5s.pt

# Real-time webcam detection
python detect.py --source 0 --weights yolov5s.pt

# Batch processing multiple images
python detect.py --source path/to/images/ --weights yolov5s.pt

# Save results with confidence threshold
python detect.py --source image.jpg --weights yolov5s.pt --conf 0.4 --save-txt --save-conf
```

### Advanced Options and Customization

#### Custom Confidence and NMS Thresholds
```python
# Adjust confidence threshold
model.conf = 0.4  # confidence threshold (0-1)
model.iou = 0.45  # NMS IoU threshold (0-1)

# Filter specific classes (COCO dataset classes)
model.classes = [0, 1, 2]  # person, bicycle, car only
```

#### Processing Results Programmatically
```python
def process_detections(results):
    # Extract detection information
    detections = results.pandas().xyxy[0]
    
    for index, detection in detections.iterrows():
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        confidence = detection['confidence']
        class_name = detection['name']
        
        print(f"Detected {class_name} with confidence {confidence:.2f} at [{x1}, {y1}, {x2}, {y2}]")
        
        # Custom processing based on detected objects
        if class_name == 'person' and confidence > 0.7:
            print("High confidence person detection!")
```

### Expected Outcomes
- Successfully load and run YOLOv5 model
- Detect common objects (people, cars, animals, etc.) in images
- Process video files frame by frame
- Implement real-time object detection using webcam
- Understand confidence thresholds and result filtering

## Exercise 2: Train YOLO on Custom Dataset (1-2 Classes)

### Objective
Learn to prepare custom datasets and train YOLOv5 models on specific object classes relevant to your application.

### Dataset Preparation

#### Step 1: Collect Images
```python
import os
import requests
from PIL import Image

def download_images(urls, save_dir):
    """Download images from URLs"""
    os.makedirs(save_dir, exist_ok=True)
    
    for i, url in enumerate(urls):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(f"{save_dir}/image_{i:04d}.jpg", 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded image {i+1}/{len(urls)}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")

# Example: Create dataset directory structure
def create_dataset_structure(dataset_name):
    structure = [
        f"{dataset_name}/images/train",
        f"{dataset_name}/images/val",
        f"{dataset_name}/labels/train",
        f"{dataset_name}/labels/val"
    ]
    
    for path in structure:
        os.makedirs(path, exist_ok=True)
    
    print(f"Created dataset structure for {dataset_name}")
```

#### Step 2: Create Dataset Configuration
```yaml
# custom_dataset.yaml
path: /path/to/your/dataset  # dataset root directory
train: images/train  # train images (relative to path)
val: images/val  # val images (relative to path)

# Number of classes
nc: 2

# Class names
names: ['apple', 'orange']  # example: fruit detection
```

#### Step 3: Convert Annotations to YOLO Format
```python
import json
import os
from PIL import Image

def convert_coco_to_yolo(coco_json_path, images_dir, labels_dir):
    """Convert COCO format annotations to YOLO format"""
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create class mapping
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    class_to_idx = {name: idx for idx, name in enumerate(categories.values())}
    
    # Process each image
    for image_info in coco_data['images']:
        image_id = image_info['id']
        image_width = image_info['width']
        image_height = image_info['height']
        image_filename = image_info['file_name']
        
        # Get annotations for this image
        image_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
        
        # Create YOLO format label file
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)
        
        with open(label_path, 'w') as label_file:
            for ann in image_annotations:
                # Convert COCO bbox to YOLO format
                x, y, w, h = ann['bbox']
                x_center = (x + w/2) / image_width
                y_center = (y + h/2) / image_height
                width = w / image_width
                height = h / image_height
                
                # Get class index
                category_name = categories[ann['category_id']]
                class_idx = class_to_idx[category_name]
                
                # Write YOLO format line
                label_file.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def convert_voc_to_yolo(voc_xml_path, image_width, image_height, class_names):
    """Convert Pascal VOC XML to YOLO format"""
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(voc_xml_path)
    root = tree.getroot()
    
    yolo_annotations = []
    
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in class_names:
            continue
            
        class_idx = class_names.index(class_name)
        
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        # Convert to YOLO format
        x_center = (xmin + xmax) / 2.0 / image_width
        y_center = (ymin + ymax) / 2.0 / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height
        
        yolo_annotations.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_annotations
```

### Training Process

#### Step 1: Training Command
```bash
# Basic training command
python train.py --data custom_dataset.yaml --weights yolov5s.pt --epochs 100 --img 640 --batch-size 16

# Advanced training with custom parameters
python train.py \
    --data custom_dataset.yaml \
    --weights yolov5s.pt \
    --epochs 300 \
    --img 640 \
    --batch-size 16 \
    --name custom_fruit_detector \
    --save-period 50 \
    --patience 50 \
    --device 0
```

#### Step 2: Monitor Training Progress
```python
# View training results
import matplotlib.pyplot as plt
import pandas as pd

def plot_training_results(results_path):
    """Plot training metrics"""
    results = pd.read_csv(results_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot losses
    axes[0, 0].plot(results['epoch'], results['train/box_loss'], label='Box Loss')
    axes[0, 0].plot(results['epoch'], results['train/obj_loss'], label='Objectness Loss')
    axes[0, 0].plot(results['epoch'], results['train/cls_loss'], label='Classification Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].legend()
    
    # Plot mAP
    axes[0, 1].plot(results['epoch'], results['metrics/mAP_0.5'], label='mAP@0.5')
    axes[0, 1].plot(results['epoch'], results['metrics/mAP_0.5:0.95'], label='mAP@0.5:0.95')
    axes[0, 1].set_title('Mean Average Precision')
    axes[0, 1].legend()
    
    # Plot precision and recall
    axes[1, 0].plot(results['epoch'], results['metrics/precision'], label='Precision')
    axes[1, 0].plot(results['epoch'], results['metrics/recall'], label='Recall')
    axes[1, 0].set_title('Precision and Recall')
    axes[1, 0].legend()
    
    # Plot F1 score
    axes[1, 1].plot(results['epoch'], results['metrics/precision'] * results['metrics/recall'] * 2 / 
                   (results['metrics/precision'] + results['metrics/recall']))
    axes[1, 1].set_title('F1 Score')
    
    plt.tight_layout()
    plt.show()

# Usage
plot_training_results('runs/train/custom_fruit_detector/results.csv')
```

#### Step 3: Evaluate Trained Model
```bash
# Validate model performance
python val.py --data custom_dataset.yaml --weights runs/train/custom_fruit_detector/weights/best.pt --img 640

# Test on new images
python detect.py --source test_images/ --weights runs/train/custom_fruit_detector/weights/best.pt --conf 0.4
```

### Expected Outcomes
- Successfully prepare custom dataset in YOLO format
- Train YOLOv5 model on 1-2 custom classes
- Monitor training progress and metrics
- Achieve reasonable detection accuracy on custom objects
- Understand overfitting and validation strategies

## Exercise 3: Image Annotation using LabelImg or Roboflow

### Objective
Learn to create high-quality annotations for custom datasets using popular annotation tools.

### Option 1: LabelImg (Desktop Application)

#### Installation and Setup
```bash
# Install LabelImg
pip install labelImg

# Or install from source
git clone https://github.com/tzutalin/labelImg.git
cd labelImg
pip install pyqt5 lxml
make qt5py3
python labelImg.py
```

#### Using LabelImg
1. **Launch Application**: Run `labelImg` or `python labelImg.py`
2. **Open Directory**: Click "Open Dir" and select your images folder
3. **Set Save Directory**: Click "Change Save Dir" and select labels folder
4. **Choose Format**: Select "YOLO" format from the menu
5. **Create Annotations**:
   - Press 'W' or click "Create RectBox" to start drawing
   - Draw bounding box around object
   - Select or type class name
   - Press 'D' to go to next image
   - Press 'A' to go to previous image

#### Best Practices for Annotation
```python
# Verification script for LabelImg annotations
import os
import cv2
import numpy as np

def visualize_yolo_annotations(image_path, label_path, class_names):
    """Visualize YOLO format annotations"""
    # Load image
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    # Read annotations
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            # Parse YOLO format
            class_id, x_center, y_center, w, h = map(float, line.strip().split())
            
            # Convert to pixel coordinates
            x_center *= width
            y_center *= height
            w *= width
            h *= height
            
            # Calculate bounding box corners
            x1 = int(x_center - w/2)
            y1 = int(y_center - h/2)
            x2 = int(x_center + w/2)
            y2 = int(y_center + h/2)
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add class label
            class_name = class_names[int(class_id)]
            cv2.putText(image, class_name, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return image

def check_annotation_quality(dataset_path, class_names):
    """Check annotation quality and statistics"""
    images_path = os.path.join(dataset_path, 'images')
    labels_path = os.path.join(dataset_path, 'labels')
    
    total_images = len([f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    annotated_images = len([f for f in os.listdir(labels_path) if f.endswith('.txt')])
    
    print(f"Total images: {total_images}")
    print(f"Annotated images: {annotated_images}")
    print(f"Annotation coverage: {annotated_images/total_images*100:.1f}%")
    
    # Count objects per class
    class_counts = {name: 0 for name in class_names}
    
    for label_file in os.listdir(labels_path):
        if label_file.endswith('.txt'):
            with open(os.path.join(labels_path, label_file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    class_id = int(line.split()[0])
                    class_counts[class_names[class_id]] += 1
    
    print("\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} objects")
```

### Option 2: Roboflow (Web-based Platform)

#### Setup and Usage
1. **Create Account**: Sign up at roboflow.com
2. **Create Project**: Start new object detection project
3. **Upload Images**: Drag and drop images to upload
4. **Annotate Online**:
   - Click on image to start annotating
   - Draw bounding boxes around objects
   - Assign class labels
   - Use keyboard shortcuts for efficiency

#### Roboflow API Integration
```python
from roboflow import Roboflow

# Initialize Roboflow
rf = Roboflow(api_key="your_api_key")
project = rf.workspace().project("your_project_name")
dataset = project.version(1).download("yolov5")

# The dataset will be downloaded in YOLOv5 format
print(f"Dataset downloaded to: {dataset.location}")

# Access dataset configuration
with open(f"{dataset.location}/data.yaml", 'r') as f:
    print(f.read())
```

#### Roboflow Advanced Features
```python
# Data augmentation and preprocessing
def setup_roboflow_augmentations():
    """Example augmentation configuration"""
    augmentations = {
        "resize": {"width": 640, "height": 640},
        "rotation": {"degrees": 15},
        "brightness": {"percent": 20},
        "blur": {"pixels": 1},
        "noise": {"percent": 5},
        "flip": {"horizontal": True},
        "crop": {"percent": 10}
    }
    return augmentations

# Export to different formats
def export_dataset_formats(project):
    """Export dataset in multiple formats"""
    formats = ['yolov5', 'coco', 'voc', 'tensorflow']
    
    for format_type in formats:
        try:
            dataset = project.version(1).download(format_type)
            print(f"Exported {format_type} to {dataset.location}")
        except Exception as e:
            print(f"Error exporting {format_type}: {e}")
```

### Annotation Quality Guidelines

#### Best Practices
1. **Consistent Bounding Boxes**: Include the entire object, avoid cutting off parts
2. **Tight Boxes**: Minimize background inclusion while capturing full object
3. **Occlusion Handling**: Annotate visible parts of partially occluded objects
4. **Edge Cases**: Include objects at image boundaries
5. **Difficult Objects**: Annotate challenging cases (blurry, small, etc.)

#### Quality Control Script
```python
import cv2
import numpy as np
from pathlib import Path

def analyze_annotation_quality(dataset_path):
    """Analyze annotation quality metrics"""
    images_dir = Path(dataset_path) / 'images'
    labels_dir = Path(dataset_path) / 'labels'
    
    issues = []
    
    for image_path in images_dir.glob('*.jpg'):
        label_path = labels_dir / f"{image_path.stem}.txt"
        
        if not label_path.exists():
            issues.append(f"Missing annotation: {image_path.name}")
            continue
            
        # Load image
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        
        # Check annotations
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            try:
                parts = line.strip().split()
                if len(parts) != 5:
                    issues.append(f"Invalid format in {label_path.name}, line {i+1}")
                    continue
                    
                class_id, x_center, y_center, w, h = map(float, parts)
                
                # Check coordinate validity
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                    issues.append(f"Invalid center coordinates in {label_path.name}, line {i+1}")
                    
                if not (0 < w <= 1 and 0 < h <= 1):
                    issues.append(f"Invalid dimensions in {label_path.name}, line {i+1}")
                    
                # Check if bounding box is reasonable size
                if w * width < 10 or h * height < 10:
                    issues.append(f"Very small object in {label_path.name}, line {i+1}")
                    
            except ValueError as e:
                issues.append(f"Parse error in {label_path.name}, line {i+1}: {e}")
    
    print(f"Found {len(issues)} annotation issues:")
    for issue in issues[:10]:  # Show first 10 issues
        print(f"- {issue}")
    
    if len(issues) > 10:
        print(f"... and {len(issues) - 10} more issues")
```

### Expected Outcomes
- Successfully annotate 50-100 images using LabelImg or Roboflow
- Create high-quality bounding box annotations
- Understand annotation best practices and quality control
- Generate YOLO-format dataset ready for training
- Implement annotation verification and quality checking procedures

These exercises provide hands-on experience with the core YOLO workflow: using pre-trained models, training custom models, and creating quality training datasets through proper annotation techniques.
