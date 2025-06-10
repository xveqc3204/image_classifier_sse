# Image Classifier SSE - Object Detection Based

A system that classifies environments as **Chemical** or **Biological** based on **detected objects** in images, perfect for sensor systems that need to determine lab environment types.

## Overview

This system takes images with COCO-format object detection annotations and classifies the overall environment by analyzing the detected objects, their confidence scores, and spatial properties. It's designed for sensor systems that receive object detection data and need to determine environment type.

**How it works:**
1. **Input**: Object detection results (bounding boxes + labels + confidence scores)
2. **Analysis**: Score each detected object as chemical/biological/neutral indicator
3. **Aggregation**: Combine object scores using weighted algorithms
4. **Output**: "Chemical" or "Biological" environment classification

## Key Features

- **Object Detection Based**: Works with your existing COCO detection data
- **Dual Classification Methods**: Neural network + rule-based approaches
- **Explainable Predictions**: Shows which objects influenced the decision
- **Confidence Scoring**: Uses detection confidence scores for better accuracy
- **Configurable Weights**: Easy to adjust object importance
- **Sensor-Ready**: Perfect for real-time sensor systems

## System Architecture

```
COCO Annotations → Object Analysis → Feature Extraction → Classification
      ↓                    ↓                ↓               ↓
  Objects +         Score Objects      Numerical         Binary
  Confidence    (Chemical/Bio/Neutral)   Features      Classification
```

## Installation

1. **Setup project directory**:
```bash
mkdir image_classifier_sse
cd image_classifier_sse
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Directory structure**:
```
image_classifier_sse/
├── data/
│   ├── raw/
│   │   ├── images/
│   │   └── annotations/
│   │       └── _annotations.coco.json
│   └── processed/
├── src/
│   ├── object_analyzer.py      # NEW: Object analysis engine
│   ├── data_preprocessing.py   # Modified for object detection
│   ├── model.py               # Feature-based neural network
│   ├── train.py               # Training with comparison
│   └── inference.py           # Object-based inference
├── models/
├── config.yaml
└── README.md
```

## Quick Start

### 1. Prepare Your COCO Data

Place your COCO JSON annotation files in the data directory:

```bash
data/raw/annotations/_annotations.coco.json  # Your object detection annotations
data/raw/images/                             # Corresponding images (optional)
```

### 2. Process Object Detection Data

Extract object features from your COCO annotations:

```bash
python src/data_preprocessing.py
```

This will:
- Parse COCO object detection annotations
- Analyze detected objects and their properties
- Create object-based feature datasets
- Generate comprehensive visualizations
- Split data into train/validation/test sets

### 3. Train the Classifier

Train both neural network and rule-based classifiers:

```bash
python src/train.py
```

Features:
- **Dual approach**: Neural network + rule-based comparison
- **Object feature learning**: Learns from detection patterns
- **Automatic comparison**: Shows which method works better
- **Feature importance**: Shows which object types matter most

### 4. Make Predictions

#### From COCO Annotations (Batch)
```bash
python src/inference.py --coco data/raw/annotations/_annotations.coco.json
```

#### Specific Image
```bash
python src/inference.py --coco data/raw/annotations/_annotations.coco.json --image-id 123
```

#### Web Interface
```bash
python src/inference.py --interface
```

#### Rule-Based Only
```bash
python src/inference.py --coco data/raw/annotations/_annotations.coco.json --rule-based
```

## Object Scoring System

### Object Categories & Weights

**Strong Chemical Indicators (weight: 1.0)**
- Chemical substance, Gas mask, Gas tank, Hazmat suit

**Moderate Chemical Indicators (weight: 0.6-0.7)**
- Fume hood, Ventilation, Eye protection

**Strong Biological Indicators (weight: 1.0)**
- Biohazard symbol, Petri dish, Pipette

**Moderate Biological Indicators (weight: 0.5-0.6)**
- Glassware, Lab coat

**Neutral Objects (weight: 0.0)**
- Computer, Tech equipment, Gloves

### Classification Algorithm

```python
# For each detected object:
object_score = detection_confidence × object_weight

# Aggregate scores:
chemical_total = sum(chemical_object_scores)
biological_total = sum(biological_object_scores)

# Final classification:
bio_probability = biological_total / (chemical_total + biological_total)
classification = "Biological" if bio_probability > 0.5 else "Chemical"
```

## Model Comparison

The system includes **two classification approaches**:

### 1. Rule-Based Classifier
- **Fast**: Immediate predictions
- **Interpretable**: Clear decision logic  
- **No training needed**: Works out of the box
- **Baseline performance**: Good starting point

### 2. Neural Network Classifier
- **Feature learning**: Learns complex patterns
- **Higher accuracy**: Often beats rule-based
- **Requires training**: Needs labeled data
- **14 input features**: Object counts, confidence scores, areas, ratios

## Understanding the Output

### Prediction Results
```
Classification: Biological
Confidence: 0.847 (84.7%)
Method: Neural Network
Objects detected: 5
Dominant objects: [petri dish, pipette, biohazard symbol]
Explanation: Detected 3 biological-related objects. Biological indicators have higher weighted confidence.
```

### Detailed Analysis
- **Object breakdown**: Which objects were detected
- **Confidence scores**: How confident the detector was
- **Weighted scores**: How each object influenced the decision
- **Method used**: Neural network vs rule-based
- **Feature importance**: Which patterns the model learned

## Configuration

Customize object weights and behavior in `config.yaml`:

```yaml
# Object weights for classification
object_weights:
  gas_mask: 1.0          # Strong chemical indicator
  petri_dish: 1.0        # Strong biological indicator
  lab_coat: 0.5          # Moderate indicator (both domains)
  computer: 0.0          # Neutral object

# Model settings
model:
  hidden_sizes: [64, 32, 16]
  dropout: 0.3
  
# Training settings  
training:
  epochs: 100
  learning_rate: 0.001
```

## Use Cases

Perfect for:

### 1. **Sensor Systems**
- Real-time environment classification
- Automated safety monitoring
- Equipment detection and categorization

### 2. **Laboratory Management**
- Automatic lab type identification
- Safety compliance monitoring
- Equipment inventory tracking

### 3. **Research Applications**
- Large-scale image dataset analysis
- Automated content categorization
- Environmental monitoring

## Advantages Over Image Classification

| Aspect | Object Detection Based (Ours) | Whole Image Classification |
|--------|------------------------------|---------------------------|
| **Explainability** | ✅ Shows which objects influenced decision | ❌ Black box predictions |
| **Data Efficiency** | ✅ Works with existing COCO data | ❌ Needs new labeled images |
| **Precision** | ✅ Focuses on specific objects | ❌ Considers irrelevant background |
| **Sensor Integration** | ✅ Perfect for detection systems | ❌ Requires full image processing |
| **Real-time Performance** | ✅ Fast feature-based prediction | ❌ Slower CNN inference |

## Performance Expectations

- **Accuracy**: 85-95% (depending on data quality)
- **Processing Speed**: ~1000 predictions/second
- **Training Time**: 5-15 minutes
- **Memory Usage**: Low (feature-based, not image-based)
- **Explainability**: High (shows object contributions)

## Advanced Features

### Feature Engineering
The system extracts 14 key features from object detections:
- Object counts by type
- Average/maximum confidence scores
- Weighted object scores
- Area ratios
- Object type ratios

### Model Comparison
Automatic comparison between approaches:
```bash
=== Model Comparison ===
Neural Network vs Rule-Based:
  Accuracy: 0.9200 vs 0.8800 (Δ+0.0400)
  Precision: 0.9100 vs 0.8650 (Δ+0.0450)
  F1-Score: 0.9150 vs 0.8750 (Δ+0.0400)
```

### Visualization Tools
- Object distribution analysis
- Feature correlation heatmaps
- Classification confidence distributions
- Detection pattern analysis

## API Usage

```python
from object_analyzer import ObjectAnalyzer
from inference import SSEObjectPredictor

# Initialize predictor
predictor = SSEObjectPredictor('models/best_model.pth')

# Predict from detections
detections = [
    {'category_name': 'gas_mask', 'confidence': 0.9, 'bbox': [100,100,50,60], 'area': 3000},
    {'category_name': 'hazmat_suit', 'confidence': 0.85, 'bbox': [200,150,80,120], 'area': 9600}
]

prediction, confidence = predictor.predict_from_detections(detections)
print(f"Environment: {prediction} (confidence: {confidence:.3f})")
```

## Troubleshooting

### Common Issues

1. **No objects detected**: Returns "Unknown" classification
2. **Low confidence**: Check object detection quality
3. **Poor accuracy**: Adjust object weights in config
4. **Missing categories**: Add new objects to analyzer

### Performance Tips

- **High-quality detections**: Better input = better results
- **Balanced data**: Ensure both classes in training
- **Weight tuning**: Adjust object importance based on domain knowledge
- **Confidence thresholds**: Filter low-confidence detections

## Future Enhancements

1. **Spatial analysis**: Consider object locations and relationships
2. **Temporal patterns**: Analyze object sequences over time
3. **Multi-class**: Extend beyond binary classification
4. **Active learning**: Improve with new data
5. **Edge deployment**: Optimize for embedded systems

## License

MIT License - see LICENSE file for details.

## Support

For questions about object detection based classification:
1. Check the object analyzer configuration
2. Review the training comparison results
3. Examine object detection quality
4. Consider domain-specific weight adjustments

**Remember**: Good object detection is more important than perfect algorithms! once
- **Comprehensive Evaluation**: Detailed metrics and visualizations

## Installation

1. **Clone or create the project directory**:
```bash
mkdir image_classifier_sse
cd image_classifier_sse
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up the directory structure**:
```
image_classifier_sse/
├── data/
│   ├── raw/
│   │   ├── images/
│   │   └── annotations/
│   └── processed/
├── src/
├── models/
├── notebooks/
├── requirements.txt
├── config.yaml
└── README.md
```

## Quick Start

### 1. Prepare Your Data

Place your COCO JSON annotation files and images in the appropriate directories:

```bash
# Your COCO annotation files
data/raw/annotations/_annotations.coco.json

# Your images
data/raw/images/your_image_files.jpg
```

### 2. Data Preprocessing

Process the COCO annotations and create the binary classification dataset:

```bash
python src/data_preprocessing.py
```

This will:
- Parse COCO annotations
- Classify images as Chemical/Biological/Neutral
- Create train/validation/test splits
- Generate data analysis visualizations

### 3. Train the Model

Start training with the default configuration:

```bash
python src/train.py
```

Training features:
- **Transfer Learning**: Uses pre-trained ResNet50
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rate
- **Real-time Monitoring**: Loss and accuracy tracking
- **Model Checkpointing**: Saves best performing model

### 4. Make Predictions

#### Single Image Prediction
```bash
python src/inference.py --image path/to/your/image.jpg
```

#### Batch Processing
```bash
python src/inference.py --batch path/to/image/directory
```

#### Web Interface
```bash
python src/inference.py --interface
```

## Model Architecture

### Transfer Learning Approach

The model uses a pre-trained CNN backbone (ResNet50 by default) with a custom classification head:

```
Input Image (224x224x3)
↓
Pre-trained CNN Backbone (ResNet50)
↓
Global Average Pooling
↓
Dense Layer (512 neurons) + ReLU + Dropout
↓
Dense Layer (256 neurons) + ReLU + Dropout
↓
Output Layer (1 neuron) + Sigmoid
↓
Probability Score (0.0 = Chemical, 1.0 = Biological)
```

### Category Mapping

**Biological Categories:**
- Biohazard symbol
- Petri dish
- Pipette
- Glassware
- Lab coat

**Chemical Categories:**
- Chemical substance
- Fume hood
- Gas mask
- Gas tank
- Hazmat suit
- Ventilation
- Toxic sign
- Eye protection

**Neutral Categories:**
- Computer
- Tech equipment
- Gloves

## Configuration

Modify `config.yaml` to customize the training process:

```yaml
# Model settings
model:
  architecture: "resnet50"  # or "efficientnet_b0"
  pretrained: true
  dropout: 0.5

# Training settings
training:
  epochs: 50
  learning_rate: 0.001
  batch_size: 32
  early_stopping_patience: 10

# Data settings
data:
  image_size: [224, 224]
  train_split: 0.7
  val_split: 0.2
  test_split: 0.1
```

## Understanding the Output

### Prediction Results

The model outputs:
1. **Predicted Class**: "Chemical" or "Biological"
2. **Confidence Score**: 0.0 to 1.0 (how certain the model is)
3. **Probability**: Raw sigmoid output

Example output:
```
Prediction: Biological
Confidence: 0.847 (84.7%)
Interpretation: The model is 84.7% confident this image shows biological-related content
```

### Evaluation Metrics

The model reports several metrics:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under the ROC curve

## Project Structure Explained

```
image_classifier_sse/
├── data/
│   ├── raw/                    # Original COCO data
│   └── processed/              # Processed binary classification data
│       ├── train/              # Training images
│       ├── val/                # Validation images
│       └── test/               # Test images
├── src/
│   ├── data_preprocessing.py   # COCO to binary classification converter
│   ├── model.py               # Model architecture and data loading
│   ├── train.py               # Training script
│   └── inference.py           # Prediction and web interface
├── models/
│   └── saved_models/          # Trained model checkpoints
├── notebooks/                 # Jupyter notebooks for analysis
├── requirements.txt           # Python dependencies
├── config.yaml               # Configuration file
└── README.md                 # This file
```

## Advanced Usage

### Custom Model Architecture

To use a different backbone model, modify `config.yaml`:

```yaml
model:
  architecture: "efficientnet_b0"  # Instead of "resnet50"
```

### Hyperparameter Tuning

Key parameters to experiment with:
- **Learning Rate**: Start with 0.001, try 0.0001 or 0.01
- **Batch Size**: 16, 32, or 64 depending on GPU memory
- **Dropout**: 0.3 to 0.7 for regularization
- **Image Size**: 224x224 (standard) or 256x256 for more detail

### Data Augmentation

The training includes automatic data augmentation:
- Random horizontal flips
- Random rotations (±15°)
- Color jittering
- Random cropping

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in `config.yaml`
   - Use smaller image size

2. **Low Accuracy**:
   - Check data quality and labeling
   - Increase training epochs
   - Try different learning rates

3. **Overfitting**:
   - Increase dropout rate
   - Add more training data
   - Use stronger data augmentation

4. **Class Imbalance**:
   - Check data distribution with preprocessing script
   - Use weighted loss functions
   - Collect more data for minority class

### Performance Tips

- **GPU Usage**: The model automatically uses GPU if available
- **Data Loading**: Uses multiple workers for faster data loading
- **Mixed Precision**: Consider using automatic mixed precision for faster training

## Model Performance

Expected performance on well-balanced datasets:
- **Accuracy**: 85-95%
- **Training Time**: 10-30 minutes (depending on dataset size and hardware)
- **Inference Speed**: ~50-100 images/second on GPU

## Contributing

To improve the model:

1. **Add More Categories**: Extend the category mapping in `config.yaml`
2. **Improve Architecture**: Experiment with different backbone models
3. **Data Quality**: Ensure consistent and accurate labeling
4. **Evaluation**: Add more comprehensive evaluation metrics

## Use Cases

This classifier can be used for:
- **Content Categorization**: Automatically sort images by domain
- **Equipment Inventory**: Categorize equipment in images
- **Educational Content**: Create training materials for classification
- **Research Organization**: Sort research images by field

## Technical Details

### Why Transfer Learning?

1. **Pre-trained Features**: Models already understand basic visual patterns
2. **Data Efficiency**: Works well with smaller datasets
3. **Faster Training**: Convergence in fewer epochs
4. **Better Performance**: Often outperforms training from scratch

### Binary Classification Choice

- **Simplicity**: Easy to understand and validate
- **Practical**: Matches real-world categorization needs
- **Extensible**: Can be extended to multi-class later

### Model Selection Rationale

**ResNet50**:
- Proven architecture
- Good balance of accuracy and speed
- Widely supported

**EfficientNet**:
- Better accuracy per parameter
- More efficient
- Slightly more complex

## Future Enhancements

Potential improvements:
1. **Multi-class Classification**: Predict specific object types
2. **Object Detection**: Locate and classify multiple objects
3. **Confidence Calibration**: Better uncertainty estimation
4. **Model Compression**: Smaller models for mobile deployment
5. **Active Learning**: Iteratively improve with new data

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues:
1. Check the troubleshooting section
2. Review the configuration options
3. Examine the training logs and visualizations
4. Consider the data quality and distribution

Remember: Good data is more important than a perfect model!