# Week 12: Custom Training Mastery

## Training Pipeline

### Data Loading and Preprocessing
**Dataset Structure Setup**:
```python
# Standard YOLO dataset structure
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml

# dataset.yaml configuration
path: /path/to/dataset
train: images/train
val: images/val
test: images/test
nc: 3  # number of classes
names: ['person', 'car', 'bicycle']
```

**Custom Dataset Class**:
```python
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class YOLODataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, img_size=640):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_size = img_size
        self.image_files = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load labels
        label_path = os.path.join(self.label_dir, 
                                 self.image_files[idx].replace('.jpg', '.txt'))
        
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, x, y, w, h = map(float, line.strip().split())
                    boxes.append([class_id, x, y, w, h])
        
        # Apply transformations
        if self.transform:
            image, boxes = self.transform(image, boxes)
        
        # Resize image
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, torch.tensor(boxes)

# Data loading with augmentations
def get_data_loaders(train_dir, val_dir, batch_size=16, img_size=640):
    train_transform = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomScale(scale_range=(0.8, 1.2)),
        ColorJitter(brightness=0.2, contrast=0.2),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = Compose([
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = YOLODataset(train_dir, transform=train_transform, img_size=img_size)
    val_dataset = YOLODataset(val_dir, transform=val_transform, img_size=img_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=yolo_collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=yolo_collate_fn, num_workers=4)
    
    return train_loader, val_loader
```

### Hyperparameter Tuning
**Key Hyperparameters**:
```python
# Training hyperparameters
hyperparams = {
    # Learning rate settings
    'lr0': 0.01,                    # initial learning rate
    'lrf': 0.01,                    # final learning rate factor
    'momentum': 0.937,              # SGD momentum
    'weight_decay': 0.0005,         # optimizer weight decay
    
    # Warmup settings
    'warmup_epochs': 3,             # warmup epochs
    'warmup_momentum': 0.8,         # warmup initial momentum
    'warmup_bias_lr': 0.1,          # warmup initial bias lr
    
    # Loss weights
    'box': 7.5,                     # box loss gain
    'cls': 0.5,                     # cls loss gain
    'dfl': 1.5,                     # dfl loss gain
    'fl_gamma': 0.0,                # focal loss gamma
    
    # Augmentation settings
    'hsv_h': 0.015,                 # hue augmentation
    'hsv_s': 0.7,                   # saturation augmentation
    'hsv_v': 0.4,                   # value augmentation
    'degrees': 0.0,                 # rotation degrees
    'translate': 0.1,               # translation
    'scale': 0.5,                   # scaling
    'shear': 0.0,                   # shear degrees
    'perspective': 0.0,             # perspective
    'flipud': 0.0,                  # vertical flip probability
    'fliplr': 0.5,                  # horizontal flip probability
    'mosaic': 1.0,                  # mosaic probability
    'mixup': 0.0,                   # mixup probability
    'copy_paste': 0.0,              # copy paste probability
}
```

**Hyperparameter Search**:
```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr0 = trial.suggest_float('lr0', 1e-5, 1e-1, log=True)
    momentum = trial.suggest_float('momentum', 0.8, 0.99)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    box_gain = trial.suggest_float('box_gain', 1.0, 10.0)
    cls_gain = trial.suggest_float('cls_gain', 0.1, 2.0)
    
    # Train model with suggested hyperparameters
    model = YOLO('yolov8n.pt')
    results = model.train(
        data='dataset.yaml',
        epochs=50,
        lr0=lr0,
        momentum=momentum,
        weight_decay=weight_decay,
        box=box_gain,
        cls=cls_gain,
        verbose=False
    )
    
    # Return validation mAP
    return results.results_dict['metrics/mAP50-95(B)']

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

### Training Monitoring and Logging
**Weights & Biases Integration**:
```python
import wandb

# Initialize wandb
wandb.init(
    project="yolo-custom-training",
    config={
        "model": "yolov8n",
        "dataset": "custom_dataset",
        "epochs": 300,
        "batch_size": 16,
        "learning_rate": 0.01
    }
)

# Custom training loop with logging
def train_with_logging(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.937)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = compute_loss(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Log batch metrics
            if batch_idx % 100 == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch,
                    "batch": batch_idx
                })
        
        # Validation
        val_metrics = validate(model, val_loader)
        
        # Log epoch metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader),
            "val_mAP": val_metrics['mAP'],
            "val_precision": val_metrics['precision'],
            "val_recall": val_metrics['recall'],
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        scheduler.step()
        
        # Save checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss / len(train_loader),
            }, f'checkpoint_epoch_{epoch}.pt')
```

### Validation Strategies
**K-Fold Cross Validation**:
```python
from sklearn.model_selection import KFold

def k_fold_validation(dataset_path, k=5, epochs=100):
    # Load all image paths
    image_paths = glob.glob(os.path.join(dataset_path, 'images', '*.jpg'))
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(image_paths)):
        print(f"Training Fold {fold + 1}/{k}")
        
        # Split data
        train_paths = [image_paths[i] for i in train_idx]
        val_paths = [image_paths[i] for i in val_idx]
        
        # Create temporary dataset splits
        create_fold_split(train_paths, val_paths, fold)
        
        # Train model
        model = YOLO('yolov8n.pt')
        results = model.train(
            data=f'fold_{fold}_dataset.yaml',
            epochs=epochs,
            project=f'fold_{fold}',
            name='train'
        )
        
        # Validate
        val_results = model.val()
        fold_results.append(val_results.results_dict['metrics/mAP50-95(B)'])
    
    # Calculate average performance
    avg_map = np.mean(fold_results)
    std_map = np.std(fold_results)
    
    print(f"K-Fold Results: {avg_map:.3f} ± {std_map:.3f}")
    return fold_results
```

## Common Issues & Solutions

### Overfitting Prevention
**Early Stopping Implementation**:
```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
```

**Regularization Techniques**:
```python
# Dropout in training
model.train()
for module in model.modules():
    if isinstance(module, nn.Dropout):
        module.p = 0.2  # Increase dropout probability

# L2 Regularization
optimizer = torch.optim.SGD(
    model.parameters(), 
    lr=0.01, 
    weight_decay=0.0005  # L2 regularization
)

# Data augmentation
strong_augment = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.2),
    RandomRotation(degrees=10),
    ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    RandomScale(scale_range=(0.7, 1.3)),
    RandomCrop(crop_ratio=0.8),
    Mosaic(p=0.8),
    MixUp(p=0.3)
])
```

### Class Imbalance Handling
**Weighted Loss Function**:
```python
def compute_weighted_loss(outputs, targets, class_weights):
    # Calculate class frequencies
    class_counts = torch.bincount(targets[:, 0].long())
    total_samples = len(targets)
    
    # Compute weights (inverse frequency)
    weights = total_samples / (len(class_counts) * class_counts.float())
    weights = weights / weights.sum() * len(class_counts)  # Normalize
    
    # Apply weights to loss
    classification_loss = weighted_crossentropy(outputs, targets, weights)
    
    return classification_loss

# Focal Loss for severe imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
```

**Sampling Strategies**:
```python
from torch.utils.data import WeightedRandomSampler

def create_balanced_sampler(dataset):
    # Calculate class frequencies
    class_counts = {}
    for _, labels in dataset:
        for label in labels:
            class_id = int(label[0])
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    # Calculate sample weights
    total_samples = sum(class_counts.values())
    class_weights = {k: total_samples / v for k, v in class_counts.items()}
    
    # Assign weight to each sample
    sample_weights = []
    for _, labels in dataset:
        max_weight = 0
        for label in labels:
            class_id = int(label[0])
            max_weight = max(max_weight, class_weights[class_id])
        sample_weights.append(max_weight)
    
    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler
```

### Low mAP Troubleshooting
**Systematic Debugging Approach**:
```python
def diagnose_low_map(model, val_loader):
    model.eval()
    diagnosis = {
        'false_positives': 0,
        'false_negatives': 0,
        'localization_errors': 0,
        'classification_errors': 0,
        'confidence_issues': 0
    }
    
    with torch.no_grad():
        for images, targets in val_loader:
            predictions = model(images)
            
            # Analyze predictions vs targets
            for pred, target in zip(predictions, targets):
                # Check for various error types
                diagnosis.update(analyze_prediction_errors(pred, target))
    
    # Print diagnosis
    print("mAP Diagnosis:")
    for error_type, count in diagnosis.items():
        print(f"{error_type}: {count}")
    
    return diagnosis

def analyze_prediction_errors(pred, target):
    # Implement detailed error analysis
    errors = {}
    
    # Match predictions to ground truth
    matched_preds, unmatched_preds = match_predictions(pred, target)
    
    # Analyze unmatched predictions (false positives)
    errors['false_positives'] = len(unmatched_preds)
    
    # Analyze missed targets (false negatives)
    errors['false_negatives'] = count_missed_targets(matched_preds, target)
    
    # Analyze localization accuracy
    errors['localization_errors'] = count_localization_errors(matched_preds)
    
    # Analyze classification accuracy
    errors['classification_errors'] = count_classification_errors(matched_preds)
    
    return errors
```

### Training Instability Fixes
**Gradient Clipping**:
```python
# Gradient clipping to prevent exploding gradients
max_norm = 10.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

# Mixed precision training for stability
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for images, targets in train_loader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(images)
        loss = compute_loss(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    scaler.step(optimizer)
    scaler.update()
```

**Learning Rate Scheduling**:
```python
# Adaptive learning rate reduction
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',
    factor=0.5,
    patience=10,
    min_lr=1e-6
)

# Warmup + Cosine Annealing
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * epoch / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = 0.5 * self.base_lr * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
```

## Achieving >90% mAP Project Guidelines

### Dataset Quality Requirements
```python
# Minimum dataset requirements for high mAP
dataset_requirements = {
    'min_images_per_class': 1000,
    'min_total_images': 5000,
    'annotation_quality': 'high',  # IoU > 0.95 with ground truth
    'data_diversity': {
        'lighting_conditions': ['bright', 'dim', 'artificial'],
        'backgrounds': ['simple', 'complex', 'cluttered'],
        'object_scales': ['small', 'medium', 'large'],
        'orientations': ['0°', '90°', '180°', '270°'],
        'weather_conditions': ['sunny', 'cloudy', 'rainy']
    }
}
```

### Training Recipe for High mAP
```python
def high_map_training_recipe():
    # Progressive training strategy
    stages = [
        {
            'epochs': 100,
            'img_size': 320,
            'batch_size': 32,
            'lr': 0.01,
            'augmentation': 'light'
        },
        {
            'epochs': 100,
            'img_size': 640,
            'batch_size': 16,
            'lr': 0.005,
            'augmentation': 'medium'
        },
        {
            'epochs': 100,
            'img_size': 640,
            'batch_size': 8,
            'lr': 0.001,
            'augmentation': 'heavy'
        }
    ]
    
    for stage in stages:
        print(f"Training stage: {stage}")
        # Implement progressive training
        
# Test Time Augmentation for inference
def test_time_augmentation(model, image):
    augmentations = [
        lambda x: x,  # Original
        lambda x: torch.flip(x, [3]),  # Horizontal flip
        lambda x: torch.rot90(x, 1, [2, 3]),  # 90° rotation
        lambda x: torch.rot90(x, 3, [2, 3]),  # 270° rotation
    ]
    
    predictions = []
    for aug in augmentations:
        augmented = aug(image)
        pred = model(augmented)
        predictions.append(pred)
    
    # Ensemble predictions
    return ensemble_predictions(predictions)
```

### Validation and Testing Protocol
```python
def comprehensive_evaluation(model, test_loader):
    metrics = {
        'mAP50': 0,
        'mAP50-95': 0,
        'precision': 0,
        'recall': 0,
        'f1_score': 0,
        'per_class_ap': {},
        'confusion_matrix': None
    }
    
    all_predictions = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for images, targets in test_loader:
            predictions = model(images)
            all_predictions.extend(predictions)
            all_targets.extend(targets)
    
    # Calculate comprehensive metrics
    metrics = calculate_detection_metrics(all_predictions, all_targets)
    
    # Generate detailed report
    generate_evaluation_report(metrics)
    
    return metrics
```

## Key Success Factors

1. **High-Quality Data**: Clean annotations, diverse samples
2. **Progressive Training**: Start simple, gradually increase complexity
3. **Proper Validation**: Use appropriate metrics and validation strategies
4. **Hyperparameter Optimization**: Systematic tuning approach
5. **Monitoring and Debugging**: Track metrics and diagnose issues
6. **Ensemble Methods**: Combine multiple models or predictions
7. **Test-Time Augmentation**: Boost inference performance
8. **Domain-Specific Optimization**: Tailor approach to specific use case
