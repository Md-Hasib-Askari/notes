# Image Analysis for Microscopy

## Basic Image Processing with PIL/Pillow
Process microscopy images using Python's PIL library.

```python
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path):
    """Load and preprocess microscopy image"""
    
    # Load image
    image = Image.open(image_path)
    
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    
    # Apply median filter to reduce noise
    image = image.filter(ImageFilter.MedianFilter(size=3))
    
    return image

def threshold_image(image, threshold=128):
    """Apply binary threshold to image"""
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Apply threshold
    binary = (img_array > threshold).astype(np.uint8) * 255
    
    return Image.fromarray(binary)

def enhance_image_quality(image):
    """Enhance image quality for analysis"""
    
    # Convert to numpy for processing
    img_array = np.array(image)
    
    # Histogram equalization
    hist, bins = np.histogram(img_array.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    
    # Apply equalization
    equalized = np.interp(img_array.flatten(), bins[:-1], cdf_normalized)
    equalized = equalized.reshape(img_array.shape).astype(np.uint8)
    
    return Image.fromarray(equalized)

# Example usage
image = load_and_preprocess_image('bacteria_microscopy.jpg')
enhanced = enhance_image_quality(image)
binary = threshold_image(enhanced, threshold=120)
```

## Cell Counting and Colony Counting
Automated counting of bacterial cells and colonies.

```python
from scipy import ndimage
from skimage import measure, morphology
from skimage.feature import blob_dog
import cv2

def count_bacterial_cells(image, min_area=20, max_area=200):
    """Count individual bacterial cells in microscopy image"""
    
    # Convert PIL to numpy
    img_array = np.array(image)
    
    # Binary threshold
    binary = img_array < 100  # Assuming dark cells on light background
    
    # Remove small noise
    cleaned = morphology.remove_small_objects(binary, min_size=min_area)
    
    # Label connected components
    labeled = measure.label(cleaned)
    
    # Filter by area
    props = measure.regionprops(labeled)
    valid_cells = []
    
    for prop in props:
        if min_area <= prop.area <= max_area:
            valid_cells.append(prop)
    
    cell_count = len(valid_cells)
    
    return cell_count, valid_cells, labeled

def count_colonies(image, min_radius=10, max_radius=50):
    """Count bacterial colonies on agar plates"""
    
    img_array = np.array(image)
    
    # Use blob detection for circular colonies
    blobs = blob_dog(img_array, min_sigma=min_radius/3, max_sigma=max_radius/3, 
                     threshold=0.1, overlap=0.5)
    
    # Filter blobs by size
    valid_colonies = []
    for blob in blobs:
        y, x, r = blob
        if min_radius <= r * 1.414 <= max_radius:  # Convert sigma to radius
            valid_colonies.append(blob)
    
    colony_count = len(valid_colonies)
    
    return colony_count, valid_colonies

def visualize_detection(original_image, detections, detection_type='cells'):
    """Visualize detection results"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Detection overlay
    axes[1].imshow(original_image, cmap='gray')
    
    if detection_type == 'cells':
        # Overlay cell boundaries
        for cell in detections:
            y, x = cell.centroid
            axes[1].plot(x, y, 'r+', markersize=8)
    
    elif detection_type == 'colonies':
        # Overlay colony circles
        for blob in detections:
            y, x, r = blob
            circle = plt.Circle((x, y), r*1.414, color='red', fill=False, linewidth=2)
            axes[1].add_patch(circle)
    
    axes[1].set_title(f'Detected {detection_type.title()}')
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig

# Example usage
cell_count, cell_props, labeled_img = count_bacterial_cells(binary)
colony_count, colony_blobs = count_colonies(image)
visualize_detection(image, cell_props, 'cells')
```

## Measuring Cellular Structures
Measure size and morphology of bacterial cells.

```python
def measure_cell_morphology(labeled_image, props_list):
    """Measure morphological parameters of cells"""
    
    measurements = []
    
    for prop in props_list:
        # Basic measurements
        area = prop.area
        perimeter = prop.perimeter
        major_axis = prop.major_axis_length
        minor_axis = prop.minor_axis_length
        
        # Derived measurements
        circularity = 4 * np.pi * area / (perimeter ** 2)
        aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0
        elongation = 1 - (minor_axis / major_axis) if major_axis > 0 else 0
        
        measurements.append({
            'area': area,
            'perimeter': perimeter,
            'major_axis': major_axis,
            'minor_axis': minor_axis,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'elongation': elongation,
            'centroid': prop.centroid
        })
    
    return pd.DataFrame(measurements)

def analyze_cell_size_distribution(measurements_df):
    """Analyze cell size distribution"""
    
    stats = {
        'mean_area': measurements_df['area'].mean(),
        'std_area': measurements_df['area'].std(),
        'mean_length': measurements_df['major_axis'].mean(),
        'mean_width': measurements_df['minor_axis'].mean(),
        'mean_aspect_ratio': measurements_df['aspect_ratio'].mean()
    }
    
    # Plot size distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    measurements_df['area'].hist(bins=20, ax=axes[0,0])
    axes[0,0].set_title('Cell Area Distribution')
    axes[0,0].set_xlabel('Area (pixels²)')
    
    measurements_df['aspect_ratio'].hist(bins=20, ax=axes[0,1])
    axes[0,1].set_title('Aspect Ratio Distribution')
    axes[0,1].set_xlabel('Aspect Ratio')
    
    axes[1,0].scatter(measurements_df['major_axis'], measurements_df['minor_axis'])
    axes[1,0].set_title('Length vs Width')
    axes[1,0].set_xlabel('Major Axis (pixels)')
    axes[1,0].set_ylabel('Minor Axis (pixels)')
    
    measurements_df['circularity'].hist(bins=20, ax=axes[1,1])
    axes[1,1].set_title('Circularity Distribution')
    axes[1,1].set_xlabel('Circularity')
    
    plt.tight_layout()
    
    return stats, fig

# Example analysis
measurements = measure_cell_morphology(labeled, valid_cells)
stats, plot = analyze_cell_size_distribution(measurements)
```

## Automated Image Analysis Workflows
Create automated pipelines for batch image processing.

```python
def batch_process_images(image_directory, output_directory):
    """Process multiple images automatically"""
    
    from pathlib import Path
    
    input_path = Path(image_directory)
    output_path = Path(output_directory)
    output_path.mkdir(exist_ok=True)
    
    results_summary = []
    
    # Process each image
    for image_file in input_path.glob('*.jpg'):
        try:
            # Load and process
            image = load_and_preprocess_image(image_file)
            enhanced = enhance_image_quality(image)
            binary = threshold_image(enhanced)
            
            # Count cells
            cell_count, cell_props, labeled = count_bacterial_cells(binary)
            
            # Measure morphology
            measurements = measure_cell_morphology(labeled, cell_props)
            
            # Save results
            results_file = output_path / f"{image_file.stem}_results.csv"
            measurements.to_csv(results_file, index=False)
            
            # Save processed image
            processed_file = output_path / f"{image_file.stem}_processed.png"
            binary.save(processed_file)
            
            # Add to summary
            results_summary.append({
                'filename': image_file.name,
                'cell_count': cell_count,
                'mean_area': measurements['area'].mean() if len(measurements) > 0 else 0,
                'mean_aspect_ratio': measurements['aspect_ratio'].mean() if len(measurements) > 0 else 0
            })
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(output_path / 'batch_summary.csv', index=False)
    
    return summary_df

# Run batch processing
summary = batch_process_images('./microscopy_images', './analysis_results')
```

These image analysis tools enable automated quantification of bacterial cells and colonies from microscopy images.
