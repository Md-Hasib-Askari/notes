# Classification vs Detection vs Segmentation

## Overview

Computer vision tasks can be broadly categorized into three main types based on the level of detail they provide about objects in images: Classification, Detection, and Segmentation. Each serves different purposes and has varying levels of complexity.

## Image Classification

### Definition
Image classification is the task of assigning a single label or category to an entire image. It answers the question "What is in this image?" but doesn't provide information about where objects are located.

### Characteristics
- **Input**: Single image
- **Output**: Class label (e.g., "cat", "dog", "car")
- **Complexity**: Lowest among the three tasks
- **Use Cases**: Photo tagging, medical diagnosis from full images, content filtering

### Example
Given an image of a dog in a park, classification would simply output "dog" as the predicted class.

## Object Detection

### Definition
Object detection combines classification and localization by identifying what objects are present in an image and where they are located. It answers "What objects are in this image and where are they?"

### Characteristics
- **Input**: Single image
- **Output**: Multiple bounding boxes with class labels and confidence scores
- **Complexity**: Medium complexity
- **Information Provided**: Object class + rectangular bounding box coordinates
- **Use Cases**: Autonomous driving, surveillance, inventory management

### Example
Given an image with multiple objects, detection would output: "dog at coordinates (100,150,250,300)", "tree at coordinates (50,75,150,400)", etc.

## Image Segmentation

### Definition
Segmentation provides pixel-level understanding of images by classifying each pixel in the image. It's the most detailed form of image analysis.

### Types of Segmentation

#### Semantic Segmentation
- **Purpose**: Classifies each pixel into predefined categories
- **Output**: Pixel-wise class labels
- **Limitation**: Doesn't distinguish between different instances of the same class
- **Example**: All pixels belonging to "person" are labeled the same, regardless of how many people are in the image

#### Instance Segmentation
- **Purpose**: Identifies and segments individual object instances
- **Output**: Separate masks for each object instance
- **Advantage**: Can distinguish between different instances of the same class
- **Example**: Separate masks for "person1", "person2", "person3"

### Characteristics
- **Input**: Single image
- **Output**: Pixel-wise classification map or instance masks
- **Complexity**: Highest among the three tasks
- **Use Cases**: Medical imaging, autonomous driving (precise road segmentation), photo editing

## Comparison Summary

| Aspect | Classification | Detection | Segmentation |
|--------|---------------|-----------|--------------|
| **Question Answered** | What? | What + Where? | What + Where + Exact Shape? |
| **Output Granularity** | Image-level | Object-level | Pixel-level |
| **Computational Cost** | Low | Medium | High |
| **Accuracy Requirements** | Basic | Moderate | High |
| **Spatial Information** | None | Bounding boxes | Precise boundaries |

## Evolution and Relationships

These tasks build upon each other in complexity:
1. **Classification** forms the foundation by learning to recognize object features
2. **Detection** adds spatial awareness by locating objects within images
3. **Segmentation** provides the most detailed understanding by determining exact object boundaries

Modern deep learning frameworks often combine these approaches, with models like Mask R-CNN performing detection and segmentation simultaneously, while YOLO focuses primarily on efficient detection.
