"""
Model definition for Image Classifier SSE - Object Detection Based
Uses object detection features for classification instead of raw images
"""

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import yaml
import joblib
import os

class ObjectFeaturesDataset(Dataset):
    """Dataset class for object detection features"""
    
    def __init__(self, csv_path, scaler=None, fit_scaler=False):
        """
        Args:
            csv_path: Path to CSV file with object features
            scaler: Pre-fitted scaler (for val/test sets)
            fit_scaler: Whether to fit a new scaler (for train set)
        """
        # Load data
        self.df = pd.read_csv(csv_path)
        
        # Define feature columns (exclude metadata and labels)
        self.feature_columns = [
            'total_objects', 'chemical_objects', 'biological_objects', 'neutral_objects',
            'avg_chemical_confidence', 'avg_biological_confidence',
            'max_chemical_confidence', 'max_biological_confidence',
            'chemical_weighted_score', 'biological_weighted_score',
            'chemical_area_ratio', 'biological_area_ratio',
            'chemical_object_ratio', 'biological_object_ratio'
        ]
        
        # Extract features and labels
        self.features = self.df[self.feature_columns].values.astype(np.float32)
        self.labels = self.df['label_binary'].values.astype(np.float32)
        
        # Handle scaling
        if fit_scaler:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        elif scaler is not None:
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)
        else:
            self.scaler = None
        
        # Convert to tensors
        self.features = torch.FloatTensor(self.features)
        self.labels = torch.FloatTensor(self.labels)
        
        print(f"Loaded dataset: {len(self.features)} samples, {self.features.shape[1]} features")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
    def get_feature_names(self):
        return self.feature_columns

class ObjectClassifier(nn.Module):
    """Neural network for classifying based on object detection features"""
    
    def __init__(self, input_size=14, hidden_sizes=[64, 32, 16], dropout=0.3):
        """
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout rate
        """
        super(ObjectClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Create hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze()

class SimpleRuleBasedClassifier:
    """Simple rule-based classifier using object detection features"""
    
    def __init__(self, threshold=0.5):
        """
        Args:
            threshold: Decision threshold for biological vs chemical
        """
        self.threshold = threshold
        self.feature_columns = [
            'total_objects', 'chemical_objects', 'biological_objects', 'neutral_objects',
            'avg_chemical_confidence', 'avg_biological_confidence',
            'max_chemical_confidence', 'max_biological_confidence',
            'chemical_weighted_score', 'biological_weighted_score',
            'chemical_area_ratio', 'biological_area_ratio',
            'chemical_object_ratio', 'biological_object_ratio'
        ]
    
    def predict(self, features_df):
        """
        Predict using simple rules
        
        Args:
            features_df: DataFrame with object features
            
        Returns:
            predictions: Array of predictions (0=Chemical, 1=Biological)
            probabilities: Array of prediction probabilities
        """
        predictions = []
        probabilities = []
        
        for _, row in features_df.iterrows():
            chemical_score = row['chemical_weighted_score']
            biological_score = row['biological_weighted_score']
            
            # Avoid division by zero
            total_score = chemical_score + biological_score
            if total_score == 0:
                prob = 0.5  # Neutral when no decisive objects
            else:
                prob = biological_score / total_score
            
            prediction = 1 if prob > self.threshold else 0
            
            predictions.append(prediction)
            probabilities.append(prob)
        
        return np.array(predictions), np.array(probabilities)
    
    def evaluate(self, features_df, true_labels):
        """Evaluate the rule-based classifier"""
        predictions, probabilities = self.predict(features_df)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions),
            'recall': recall_score(true_labels, predictions),
            'f1_score': f1_score(true_labels, predictions),
            'auc': roc_auc_score(true_labels, probabilities)
        }
        
        return metrics

def create_data_loaders(config):
    """Create data loaders for object features"""
    
    data_dir = config['data']['processed_data_path']
    batch_size = config['data']['batch_size']
    
    # Load datasets
    train_dataset = ObjectFeaturesDataset(
        os.path.join(data_dir, 'train_features.csv'),
        fit_scaler=True
    )
    
    val_dataset = ObjectFeaturesDataset(
        os.path.join(data_dir, 'val_features.csv'),
        scaler=train_dataset.scaler
    )
    
    test_dataset = ObjectFeaturesDataset(
        os.path.join(data_dir, 'test_features.csv'),
        scaler=train_dataset.scaler
    )
    
    # Save scaler for later use
    scaler_path = os.path.join(data_dir, 'feature_scaler.pkl')
    joblib.dump(train_dataset.scaler, scaler_path)
    print(f"Saved feature scaler to {scaler_path}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")  
    print(f"  Test: {len(test_dataset)} samples")
    print(f"  Features: {len(train_dataset.get_feature_names())}")
    
    return train_loader, val_loader, test_loader, train_dataset.scaler

def create_model(config):
    """Create the neural network model"""
    
    model_config = config['model']
    
    # Get input size from feature columns
    input_size = 14  # Number of object features
    
    model = ObjectClassifier(
        input_size=input_size,
        hidden_sizes=model_config.get('hidden_sizes', [64, 32, 16]),
        dropout=model_config['dropout']
    )
    
    return model

class ModelEvaluator:
    """Class for evaluating model performance on object features"""
    
    def __init__(self, model, device, scaler=None):
        self.model = model
        self.device = device
        self.scaler = scaler
    
    def evaluate(self, data_loader, threshold=0.5):
        """Evaluate model on given data loader"""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        total_loss = 0
        
        criterion = nn.BCELoss()
        
        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                
                # Convert to predictions
                predictions = (outputs > threshold).float()
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(outputs.cpu().numpy())
                total_loss += loss.item()
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        auc = roc_auc_score(all_labels, all_probabilities)
        
        avg_loss = total_loss / len(data_loader)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'loss': avg_loss
        }
        
        return metrics, all_predictions, all_labels, all_probabilities
    
    def evaluate_single_features(self, features_array, threshold=0.5):
        """Evaluate on a single set of features"""
        self.model.eval()
        
        # Ensure features are scaled if scaler exists
        if self.scaler is not None:
            features_array = self.scaler.transform(features_array.reshape(1, -1))
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_array).to(self.device)
        if len(features_tensor.shape) == 1:
            features_tensor = features_tensor.unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(features_tensor)
            probability = output.item()
        
        prediction = 1 if probability > threshold else 0
        class_name = "Biological" if prediction == 1 else "Chemical"
        confidence = probability if prediction == 1 else (1 - probability)
        
        return class_name, confidence, probability

def compare_models(config):
    """Compare neural network vs rule-based classifier"""
    print("\n=== Model Comparison ===")
    
    data_dir = config['data']['processed_data_path']
    
    # Load test data
    test_df = pd.read_csv(os.path.join(data_dir, 'test_features.csv'))
    true_labels = test_df['label_binary'].values
    
    # Test rule-based classifier
    print("\n--- Rule-Based Classifier ---")
    rule_classifier = SimpleRuleBasedClassifier(threshold=0.5)
    rule_metrics = rule_classifier.evaluate(test_df, true_labels)
    
    for metric, value in rule_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return rule_classifier, rule_metrics

def print_model_summary(model):
    """Print model architecture summary"""
    print("Model Architecture:")
    print(f"  Type: Object Features Neural Network")
    print(f"  Input size: 14 features")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    print(f"\n  Architecture:")
    for i, layer in enumerate(model.network):
        print(f"    {i}: {layer}")

# Test the model creation
if __name__ == "__main__":
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Test rule-based classifier
    print("Testing rule-based classifier...")
    rule_classifier = SimpleRuleBasedClassifier()
    
    # Create sample features
    sample_features = pd.DataFrame({
        'total_objects': [3, 2],
        'chemical_objects': [2, 0],
        'biological_objects': [0, 2],
        'neutral_objects': [1, 0],
        'chemical_weighted_score': [1.5, 0.0],
        'biological_weighted_score': [0.0, 1.8],
        'avg_chemical_confidence': [0.8, 0.0],
        'avg_biological_confidence': [0.0, 0.9],
        'max_chemical_confidence': [0.9, 0.0],
        'max_biological_confidence': [0.0, 0.95],
        'chemical_area_ratio': [0.7, 0.0],
        'biological_area_ratio': [0.0, 0.8],
        'chemical_object_ratio': [1.0, 0.0],
        'biological_object_ratio': [0.0, 1.0]
    })
    
    predictions, probabilities = rule_classifier.predict(sample_features)
    print(f"Predictions: {predictions}")
    print(f"Probabilities: {probabilities}")
    
    # Test neural network model
    print(f"\nTesting neural network model...")
    model = create_model(config)
    print_model_summary(model)
    
    # Test forward pass
    dummy_features = torch.randn(2, 14)  # Batch of 2, 14 features
    output = model(dummy_features)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_features.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output values: {output.detach().numpy()}")
        
    # Custom classifier head
    self.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes)
    )
        
        # For binary classification, we use sigmoid activation
    self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Extract features using the backbone
        features = self.backbone(x)
        
        # Classification
        logits = self.classifier(features)
        
        # Apply sigmoid for binary classification
        probabilities = self.sigmoid(logits)
        
        return probabilities

def get_transforms(image_size=(224, 224), augment=True):
    """Get image transforms for training and validation"""
    
    # Basic transforms (always applied)
    basic_transforms = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225]    # ImageNet stds
        )
    ]
    
    if augment:
        # Training transforms with augmentation
        train_transforms = transforms.Compose([
            transforms.Resize((int(image_size[0] * 1.1), int(image_size[1] * 1.1))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return train_transforms
    else:
        # Validation/test transforms (no augmentation)
        return transforms.Compose(basic_transforms)

def create_data_loaders(config):
    """Create data loaders for training, validation, and testing"""
    
    data_dir = config['data']['processed_data_path']
    batch_size = config['data']['batch_size']
    image_size = tuple(config['data']['image_size'])
    
    # Define transforms
    train_transform = get_transforms(image_size, augment=True)
    val_transform = get_transforms(image_size, augment=False)
    
    # Create datasets
    train_dataset = SSEDataset(
        os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    
    val_dataset = SSEDataset(
        os.path.join(data_dir, 'val'),
        transform=val_transform
    )
    
    test_dataset = SSEDataset(
        os.path.join(data_dir, 'test'),
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Validation: {len(val_dataset)} images")
    print(f"  Test: {len(test_dataset)} images")
    
    return train_loader, val_loader, test_loader

def create_model(config):
    """Create the model based on configuration"""
    
    model_config = config['model']
    
    model = SSEImageClassifier(
        architecture=model_config['architecture'],
        pretrained=model_config['pretrained'],
        num_classes=model_config['num_classes'],
        dropout=model_config['dropout']
    )
    
    return model

class ModelEvaluator:
    """Class for evaluating model performance"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def evaluate(self, data_loader, threshold=0.5):
        """Evaluate model on given data loader"""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        total_loss = 0
        
        criterion = nn.BCELoss()
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.float().to(self.device)
                
                # Forward pass
                outputs = self.model(images).squeeze()
                loss = criterion(outputs, labels)
                
                # Convert to predictions
                predictions = (outputs > threshold).float()
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(outputs.cpu().numpy())
                total_loss += loss.item()
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        auc = roc_auc_score(all_labels, all_probabilities)
        
        avg_loss = total_loss / len(data_loader)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'loss': avg_loss
        }
        
        return metrics, all_predictions, all_labels, all_probabilities

def print_model_summary(model, input_size=(3, 224, 224)):
    """Print model architecture summary"""
    print("Model Architecture:")
    print(f"  Backbone: {model.architecture}")
    print(f"  Input size: {input_size}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")

# Test the model creation
if __name__ == "__main__":
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = create_model(config)
    print_model_summary(model)
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output value: {output.item():.4f}")
    print(f"  Predicted class: {'Biological' if output.item() > 0.5 else 'Chemical'}")
    
"""
Model definition for Lab Safety Image Classifier
Transfer learning approach using pre-trained CNN
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import yaml

class LabSafetyDataset(Dataset):
    """Custom Dataset class for lab safety images"""
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Directory containing 'biological' and 'chemical' subdirectories
            transform: Optional transform to be applied on images
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.classes = ['biological', 'chemical']
        
        # Load all image paths and labels
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

class LabSafetyClassifier(nn.Module):
    """Lab Safety Image Classifier using transfer learning"""
    
    def __init__(self, architecture='resnet50', pretrained=True, num_classes=1, dropout=0.5):
        """
        Args:
            architecture: Base architecture ('resnet50', 'efficientnet_b0', etc.)
            pretrained: Whether to use pretrained weights
            num_classes: Number of output classes (1 for binary classification)
            dropout: Dropout rate
        """
        super(LabSafetyClassifier, self).__init__()
        
        self.architecture = architecture
        
        # Load pre-trained model
        if architecture == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            # Replace the final layer
            self.backbone.fc = nn.Identity()
            
        elif architecture == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            # Replace the final layer
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # For binary classification, we use sigmoid activation
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Extract features using the backbone
        features = self.backbone(x)
        
        # Classification
        logits = self.classifier(features)
        
        # Apply sigmoid for binary classification
        probabilities = self.sigmoid(logits)
        
        return probabilities

def get_transforms(image_size=(224, 224), augment=True):
    """Get image transforms for training and validation"""
    
    # Basic transforms (always applied)
    basic_transforms = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225]    # ImageNet stds
        )
    ]
    
    if augment:
        # Training transforms with augmentation
        train_transforms = transforms.Compose([
            transforms.Resize((int(image_size[0] * 1.1), int(image_size[1] * 1.1))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return train_transforms
    else:
        # Validation/test transforms (no augmentation)
        return transforms.Compose(basic_transforms)

def create_data_loaders(config):
    """Create data loaders for training, validation, and testing"""
    
    data_dir = config['data']['processed_data_path']
    batch_size = config['data']['batch_size']
    image_size = tuple(config['data']['image_size'])
    
    # Define transforms
    train_transform = get_transforms(image_size, augment=True)
    val_transform = get_transforms(image_size, augment=False)
    
    # Create datasets
    train_dataset = LabSafetyDataset(
        os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    
    val_dataset = LabSafetyDataset(
        os.path.join(data_dir, 'val'),
        transform=val_transform
    )
    
    test_dataset = LabSafetyDataset(
        os.path.join(data_dir, 'test'),
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Validation: {len(val_dataset)} images")
    print(f"  Test: {len(test_dataset)} images")
    
    return train_loader, val_loader, test_loader

def create_model(config):
    """Create the model based on configuration"""
    
    model_config = config['model']
    
    model = LabSafetyClassifier(
        architecture=model_config['architecture'],
        pretrained=model_config['pretrained'],
        num_classes=model_config['num_classes'],
        dropout=model_config['dropout']
    )
    
    return model

class ModelEvaluator:
    """Class for evaluating model performance"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def evaluate(self, data_loader, threshold=0.5):
        """Evaluate model on given data loader"""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        total_loss = 0
        
        criterion = nn.BCELoss()
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.float().to(self.device)
                
                # Forward pass
                outputs = self.model(images).squeeze()
                loss = criterion(outputs, labels)
                
                # Convert to predictions
                predictions = (outputs > threshold).float()
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(outputs.cpu().numpy())
                total_loss += loss.item()
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        auc = roc_auc_score(all_labels, all_probabilities)
        
        avg_loss = total_loss / len(data_loader)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'loss': avg_loss
        }
        
        return metrics, all_predictions, all_labels, all_probabilities

def print_model_summary(model, input_size=(3, 224, 224)):
    """Print model architecture summary"""
    print("Model Architecture:")
    print(f"  Backbone: {model.architecture}")
    print(f"  Input size: {input_size}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")

# Test the model creation
if __name__ == "__main__":
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = create_model(config)
    print_model_summary(model)
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output value: {output.item():.4f}")
    print(f"  Predicted class: {'Biological' if output.item() > 0.5 else 'Chemical'}")