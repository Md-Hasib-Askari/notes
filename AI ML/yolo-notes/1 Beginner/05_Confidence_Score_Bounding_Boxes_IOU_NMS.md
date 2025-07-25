# Confidence Score, Bounding Boxes, IOU, NMS

## Confidence Score

### Definition and Purpose
The confidence score is a numerical value between 0 and 1 that represents the model's certainty about the presence of an object within a predicted bounding box. It combines two important factors: the probability that an object exists in the bounding box and how well the predicted bounding box aligns with the actual object.

### Mathematical Representation
In YOLO, confidence score is calculated as:
**Confidence = Pr(Object) × IOU(pred, truth)**

Where:
- Pr(Object) is the probability that an object exists in the grid cell
- IOU(pred, truth) is the Intersection over Union between predicted and ground truth boxes

### Practical Significance
- **Threshold filtering**: Objects with confidence scores below a certain threshold (e.g., 0.5) are typically discarded
- **Ranking predictions**: Higher confidence scores indicate more reliable detections
- **Model calibration**: Well-calibrated models produce confidence scores that accurately reflect prediction reliability

### Applications in Practice
During inference, confidence scores help filter out false positives and uncertain predictions, ensuring that only high-quality detections are retained for further processing.

## Bounding Boxes

### Structure and Representation
Bounding boxes are rectangular regions that define the spatial location of detected objects. They are typically represented using four parameters:

#### Common Formats:
1. **YOLO Format**: (x_center, y_center, width, height) - normalized to [0,1]
2. **Pascal VOC Format**: (x_min, y_min, x_max, y_max) - absolute pixel coordinates
3. **COCO Format**: (x_min, y_min, width, height) - absolute pixel coordinates

### Coordinate Systems
- **Normalized coordinates**: Values between 0 and 1, relative to image dimensions
- **Absolute coordinates**: Actual pixel values in the image
- **Grid-relative coordinates**: Coordinates relative to the grid cell (used internally in YOLO)

### Prediction Process
In YOLO, each grid cell predicts bounding box coordinates as offsets from anchor boxes or grid cell positions. The model learns to:
- Predict the center coordinates of the bounding box
- Predict the width and height of the bounding box
- Apply activation functions (like sigmoid) to constrain coordinate values

### Quality Metrics
The quality of bounding box predictions is measured using metrics like IoU, which determines how well the predicted box aligns with the ground truth box.

## IOU (Intersection over Union)

### Mathematical Definition
IOU measures the overlap between two bounding boxes and is calculated as:

**IOU = Area of Intersection / Area of Union**

Where:
- Area of Intersection: Overlapping area between predicted and ground truth boxes
- Area of Union: Total area covered by both boxes combined

### Calculation Steps
1. **Find intersection coordinates**: Determine the overlapping rectangular region
2. **Calculate intersection area**: Width × Height of overlapping region
3. **Calculate union area**: Area of box1 + Area of box2 - Intersection area
4. **Compute ratio**: Intersection area ÷ Union area

### IOU Values and Interpretation
- **IOU = 0**: No overlap between boxes
- **IOU = 0.5**: Moderate overlap, commonly used as threshold for positive detections
- **IOU = 1**: Perfect overlap, identical boxes
- **IOU ≥ 0.7**: Generally considered good localization
- **IOU < 0.3**: Poor localization, usually treated as negative

### Applications
- **Training**: Used in loss functions to measure localization accuracy
- **Evaluation**: Standard metric for assessing detection quality
- **NMS**: Used to identify overlapping detections for suppression
- **Anchor matching**: Determines which anchors are responsible for detecting specific objects

## NMS (Non-Maximum Suppression)

### Problem Addressed
Object detection models often produce multiple overlapping bounding boxes for the same object. NMS eliminates redundant detections by keeping only the most confident prediction for each object.

### Algorithm Steps
1. **Sort by confidence**: Arrange all detections in descending order of confidence scores
2. **Select highest confidence**: Pick the detection with the highest confidence score
3. **Calculate IOU**: Compute IOU between the selected detection and all remaining detections
4. **Suppress overlapping boxes**: Remove detections with IOU above a threshold (typically 0.4-0.6)
5. **Repeat**: Continue the process with remaining detections until all are processed

### Key Parameters
- **Confidence threshold**: Minimum confidence score for a detection to be considered
- **IOU threshold**: Maximum allowed overlap before suppression (typically 0.4-0.6)
- **Maximum detections**: Maximum number of objects to detect per image

### Variants of NMS
- **Hard NMS**: Completely removes overlapping detections above threshold
- **Soft NMS**: Reduces confidence scores of overlapping detections instead of removing them
- **DIoU NMS**: Uses Distance-IoU instead of standard IoU for better suppression

### Implementation Considerations
- **Class-wise NMS**: Apply NMS separately for each object class
- **Global NMS**: Apply NMS across all classes (less common)
- **Batch processing**: Efficient implementation for processing multiple images

### Impact on Performance
Proper NMS configuration significantly affects:
- **Precision**: Reduces false positives by eliminating duplicate detections
- **Recall**: May reduce recall if threshold is too aggressive
- **Inference speed**: Adds computational overhead but essential for clean results

These four concepts work together to form the foundation of modern object detection systems, with each playing a crucial role in producing accurate and reliable detection results.
