"""
Training script for Image Classifier SSE - Object Detection Based
Trains on object detection features instead of raw images
"""

import os
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
import pandas as pd

from model import create_model, create_data_loaders, ModelEvaluator, compare_models, print_model_summary

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

class SSETrainer:
    """Main trainer class for the object detection based SSE classifier"""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize trainer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create model
        self.model = create_model(self.config)
        self.model.to(self.device)
        
        # Print model summary
        print_model_summary(self.model)
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader, self.scaler = create_data_loaders(self.config)
        
        # Setup training components
        self.setup_training()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rate': []
        }
    
    def setup_training(self):
        """Setup optimizer, loss function, and schedulers"""
        train_config = self.config['training']
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Optimizer
        if train_config['optimizer'].lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=train_config['learning_rate'],
                weight_decay=1e-4  # L2 regularization
            )
        elif train_config['optimizer'].lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=train_config['learning_rate'],
                momentum=0.9,
                weight_decay=1e-4
            )
        else:
            raise ValueError(f"Unsupported optimizer: {train_config['optimizer']}")
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=train_config['reduce_lr_patience'],
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=train_config['early_stopping_patience'],
            min_delta=0.001
        )
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (features, labels) in enumerate(self.train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # Print progress occasionally
            if batch_idx % 10 == 0 and batch_idx > 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = correct_predictions / total_samples
        
        return epoch_loss, epoch_accuracy
    
    def validate_epoch(self):
        """Validate for one epoch"""
        evaluator = ModelEvaluator(self.model, self.device, self.scaler)
        metrics, _, _, _ = evaluator.evaluate(self.val_loader)
        
        return metrics['loss'], metrics['accuracy']
    
    def train(self):
        """Main training loop"""
        print(f"\nStarting training for {self.config['training']['epochs']} epochs...")
        
        # First, compare with rule-based baseline
        print("\n=== Baseline Comparison ===")
        rule_classifier, rule_metrics = compare_models(self.config)
        print(f"Rule-based baseline accuracy: {rule_metrics['accuracy']:.4f}")
        
        start_time = time.time()
        best_val_accuracy = 0.0
        
        for epoch in range(self.config['training']['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}")
            print("-" * 50)
            
            # Train
            train_loss, train_accuracy = self.train_epoch()
            
            # Validate
            val_loss, val_accuracy = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step(val_accuracy)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_accuracy'].append(train_accuracy)
            self.history['val_accuracy'].append(val_accuracy)
            self.history['learning_rate'].append(current_lr)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_model('best_model.pth')
                print(f"New best validation accuracy: {best_val_accuracy:.4f}")
                
                # Compare with baseline
                if best_val_accuracy > rule_metrics['accuracy']:
                    improvement = best_val_accuracy - rule_metrics['accuracy']
                    print(f"Neural network beats baseline by {improvement:.4f}!")
            
            # Early stopping
            if self.early_stopping(val_accuracy, self.model):
                print(f"Early stopping triggered after epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.2f} minutes")
        print(f"Best validation accuracy: {best_val_accuracy:.4f}")
        print(f"Baseline accuracy: {rule_metrics['accuracy']:.4f}")
        
        # Save final model and history
        self.save_model('final_model.pth')
        self.save_training_history()
        
        return best_val_accuracy, rule_metrics['accuracy']
    
    def save_model(self, filename):
        """Save model checkpoint"""
        save_dir = self.config['output']['model_save_path']
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler': self.scaler,
            'config': self.config,
            'history': self.history
        }
        
        torch.save(checkpoint, os.path.join(save_dir, filename))
        print(f"Model saved as {filename}")
    
    def load_model_safely(self, model_path):
        """Safely load model checkpoint with proper error handling"""
        try:
            # Try with safe globals first (PyTorch 2.6+ compatible)
            with torch.serialization.safe_globals([
                torch.serialization.DefaultSafeGlobals,
                'sklearn.preprocessing._data.StandardScaler'
            ]):
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"Safe loading failed: {e}")
            try:
                # Fallback: load with weights_only=False (less secure but works)
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                print("Warning: Loaded model with weights_only=False for compatibility")
            except Exception as e2:
                print(f"Model loading failed completely: {e2}")
                return None
        
        return checkpoint
    
    def save_training_history(self):
        """Save training history as JSON and plot"""
        save_dir = self.config['output']['model_save_path']
        
        # Save as JSON
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Create training plots
        self.plot_training_history()
    
    def plot_training_history(self):
        """Plot training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, self.history['train_accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, self.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate plot
        ax3.plot(epochs, self.history['learning_rate'], 'g-', linewidth=2)
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Feature importance visualization (if available)
        try:
            self.plot_feature_importance(ax4)
        except:
            # Fallback: show accuracy comparison
            ax4.bar(['Training', 'Validation'], 
                   [self.history['train_accuracy'][-1], self.history['val_accuracy'][-1]],
                   color=['blue', 'red'], alpha=0.7)
            ax4.set_title('Final Accuracy Comparison')
            ax4.set_ylabel('Accuracy')
            ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save plot
        save_dir = self.config['output']['model_save_path']
        plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, ax):
        """Plot feature importance from the trained model"""
        # Get feature names
        feature_names = [
            'Total Objects', 'Chemical Objects', 'Biological Objects', 'Neutral Objects',
            'Avg Chemical Conf', 'Avg Biological Conf', 'Max Chemical Conf', 'Max Biological Conf',
            'Chemical Score', 'Biological Score', 'Chemical Area', 'Biological Area',
            'Chemical Ratio', 'Biological Ratio'
        ]
        
        # Get weights from first layer (simplified importance measure)
        with torch.no_grad():
            first_layer = self.model.network[0]  # First linear layer
            weights = first_layer.weight.abs().mean(dim=0).cpu().numpy()
        
        # Sort by importance
        importance_order = np.argsort(weights)[::-1]
        sorted_features = [feature_names[i] for i in importance_order[:10]]  # Top 10
        sorted_weights = weights[importance_order[:10]]
        
        ax.barh(range(len(sorted_features)), sorted_weights, alpha=0.7)
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features)
        ax.set_title('Feature Importance (Top 10)')
        ax.set_xlabel('Average Weight Magnitude')
    
    def evaluate_test_set(self):
        """Evaluate the model on the test set"""
        print("\nEvaluating on test set...")
        
        # Load best model with safe loading
        best_model_path = os.path.join(self.config['output']['model_save_path'], 'best_model.pth')
        if os.path.exists(best_model_path):
            print("Loading best model for testing...")
            checkpoint = self.load_model_safely(best_model_path)
            
            if checkpoint is not None:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("✓ Successfully loaded best model for testing")
            else:
                print("⚠ Warning: Could not load best model, using current model state")
        else:
            print("⚠ Warning: Best model not found, using current model state")
        
        # Neural network evaluation
        evaluator = ModelEvaluator(self.model, self.device, self.scaler)
        nn_metrics, predictions, labels, probabilities = evaluator.evaluate(self.test_loader)
        
        print(f"\n=== Neural Network Test Results ===")
        print(f"  Accuracy: {nn_metrics['accuracy']:.4f}")
        print(f"  Precision: {nn_metrics['precision']:.4f}")
        print(f"  Recall: {nn_metrics['recall']:.4f}")
        print(f"  F1-Score: {nn_metrics['f1_score']:.4f}")
        print(f"  AUC: {nn_metrics['auc']:.4f}")
        print(f"  Loss: {nn_metrics['loss']:.4f}")
        
        # Rule-based evaluation on test set
        data_dir = self.config['data']['processed_data_path']
        test_df = pd.read_csv(os.path.join(data_dir, 'test_features.csv'))
        
        from model import SimpleRuleBasedClassifier
        rule_classifier = SimpleRuleBasedClassifier()
        rule_metrics = rule_classifier.evaluate(test_df, test_df['label_binary'].values)
        
        print(f"\n=== Rule-Based Test Results ===")
        print(f"  Accuracy: {rule_metrics['accuracy']:.4f}")
        print(f"  Precision: {rule_metrics['precision']:.4f}")
        print(f"  Recall: {rule_metrics['recall']:.4f}")
        print(f"  F1-Score: {rule_metrics['f1_score']:.4f}")
        print(f"  AUC: {rule_metrics['auc']:.4f}")
        
        # Compare models
        print(f"\n=== Model Comparison ===")
        print(f"Neural Network vs Rule-Based:")
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc']:
            nn_val = nn_metrics[metric]
            rule_val = rule_metrics[metric]
            improvement = nn_val - rule_val
            print(f"  {metric.capitalize()}: {nn_val:.4f} vs {rule_val:.4f} (Δ{improvement:+.4f})")
        
        # Plot confusion matrices
        self.plot_confusion_matrices(labels, predictions, test_df, rule_classifier)
        
        return nn_metrics, rule_metrics
    
    def plot_confusion_matrices(self, nn_labels, nn_predictions, test_df, rule_classifier):
        """Plot confusion matrices for both models"""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        # Get rule-based predictions
        rule_predictions, _ = rule_classifier.predict(test_df)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Neural Network confusion matrix
        cm_nn = confusion_matrix(nn_labels, nn_predictions)
        sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Chemical', 'Biological'],
                   yticklabels=['Chemical', 'Biological'], ax=ax1)
        ax1.set_title('Neural Network - Confusion Matrix')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Rule-based confusion matrix
        cm_rule = confusion_matrix(test_df['label_binary'].values, rule_predictions)
        sns.heatmap(cm_rule, annot=True, fmt='d', cmap='Greens', 
                   xticklabels=['Chemical', 'Biological'],
                   yticklabels=['Chemical', 'Biological'], ax=ax2)
        ax2.set_title('Rule-Based - Confusion Matrix')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        # Save confusion matrices
        save_dir = self.config['output']['model_save_path']
        plt.savefig(os.path.join(save_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to run training"""
    print("Image Classifier SSE - Object Detection Based Training")
    print("=" * 60)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA devices available: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available, using CPU")
    
    # Check if processed data exists
    if not os.path.exists('data/processed/train_features.csv'):
        print("\nError: Processed feature data not found!")
        print("Please run data_preprocessing.py first to create the object features dataset.")
        return
    
    # Create trainer
    trainer = SSETrainer()
    
    # Start training
    best_val_acc, baseline_acc = trainer.train()
    
    # Evaluate on test set
    nn_metrics, rule_metrics = trainer.evaluate_test_set()
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final test accuracy (NN): {nn_metrics['accuracy']:.4f}")
    print(f"Final test accuracy (Rule): {rule_metrics['accuracy']:.4f}")
    
    if nn_metrics['accuracy'] > rule_metrics['accuracy']:
        improvement = nn_metrics['accuracy'] - rule_metrics['accuracy']
        print(f"Neural network improved over rule-based by: {improvement:.4f}")
    else:
        print("Rule-based classifier performed better - consider using it instead!")

if __name__ == "__main__":
    main()