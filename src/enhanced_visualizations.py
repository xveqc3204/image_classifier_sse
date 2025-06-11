"""
Enhanced Visualization Module for Image Classifier SSE - Object Detection Based
Provides comprehensive analysis visualizations including confusion matrix and precision-recall curves
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report, precision_recall_curve,
    average_precision_score, roc_curve, auc, precision_score, recall_score
)
from typing import List, Dict, Tuple, Optional
import os

class EnhancedVisualizer:
    """Enhanced visualization class for comprehensive model analysis"""
    
    def __init__(self, save_dir: str = "visualizations", class_names: List[str] = ["Chemical", "Biological"]):
        """
        Initialize visualizer
        
        Args:
            save_dir: Directory to save visualizations
            class_names: Names of the classes
        """
        self.save_dir = save_dir
        self.class_names = class_names
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style for better looking plots
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_comprehensive_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_proba: np.ndarray, model_name: str = "Model",
                                  save_filename: str = None) -> None:
        """
        Create comprehensive analysis plot with multiple visualizations
        
        Args:
            y_true: True labels (0 or 1)
            y_pred: Predicted labels (0 or 1)
            y_proba: Prediction probabilities
            model_name: Name of the model for titles
            save_filename: Filename to save the plot
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Create a 2x3 grid for subplots
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Confusion Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_confusion_matrix(y_true, y_pred, ax1, model_name)
        
        # 2. Precision-Recall Curve
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_precision_recall_curve(y_true, y_proba, ax2, model_name)
        
        # 3. ROC Curve
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_roc_curve(y_true, y_proba, ax3, model_name)
        
        # 4. Prediction Distribution
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_prediction_distribution(y_true, y_proba, ax4, model_name)
        
        # 5. Classification Metrics Bar Chart
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_metrics_bar_chart(y_true, y_pred, y_proba, ax5, model_name)
        
        # 6. Confidence vs Accuracy
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_confidence_vs_accuracy(y_true, y_pred, y_proba, ax6, model_name)
        
        plt.suptitle(f'{model_name} - Comprehensive Performance Analysis', fontsize=16, fontweight='bold')
        
        if save_filename:
            plt.savefig(os.path.join(self.save_dir, save_filename), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             ax: plt.Axes, model_name: str) -> None:
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotations with both counts and percentages
        annotations = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
        
        # Plot heatmap
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_title(f'{model_name} - Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        # Add accuracy in the title
        accuracy = np.trace(cm) / np.sum(cm)
        ax.text(0.5, 1.02, f'Accuracy: {accuracy:.3f}', 
                transform=ax.transAxes, ha='center', fontweight='bold')
    
    def _plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                                    ax: plt.Axes, model_name: str) -> None:
        """Plot precision-recall curve"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        
        # Plot curve
        ax.plot(recall, precision, color='darkorange', lw=2, 
                label=f'PR curve (AP = {avg_precision:.3f})')
        
        # Plot baseline (random classifier)
        baseline = np.sum(y_true) / len(y_true)
        ax.axhline(y=baseline, color='navy', linestyle='--', 
                  label=f'Random baseline (AP = {baseline:.3f})')
        
        # Find optimal threshold (F1 score)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last element
        optimal_threshold = thresholds[optimal_idx]
        
        ax.plot(recall[optimal_idx], precision[optimal_idx], 'ro', markersize=8,
                label=f'Optimal threshold = {optimal_threshold:.3f}')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'{model_name} - Precision-Recall Curve')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
    
    def _plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                       ax: plt.Axes, model_name: str) -> None:
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot curve
        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        
        # Plot diagonal (random classifier)
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random classifier (AUC = 0.5)')
        
        # Find optimal threshold (Youden's J statistic)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8,
                label=f'Optimal threshold = {optimal_threshold:.3f}')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{model_name} - ROC Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
    
    def _plot_prediction_distribution(self, y_true: np.ndarray, y_proba: np.ndarray,
                                    ax: plt.Axes, model_name: str) -> None:
        """Plot distribution of prediction probabilities"""
        # Separate probabilities by true class
        class_0_proba = y_proba[y_true == 0]
        class_1_proba = y_proba[y_true == 1]
        
        # Plot histograms
        ax.hist(class_0_proba, bins=30, alpha=0.7, label=f'True {self.class_names[0]}', 
                color='red', density=True)
        ax.hist(class_1_proba, bins=30, alpha=0.7, label=f'True {self.class_names[1]}', 
                color='blue', density=True)
        
        # Add decision threshold line
        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, 
                  label='Decision threshold (0.5)')
        
        ax.set_xlabel('Prediction Probability')
        ax.set_ylabel('Density')
        ax.set_title(f'{model_name} - Prediction Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_metrics_bar_chart(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_proba: np.ndarray, ax: plt.Axes, model_name: str) -> None:
        """Plot classification metrics as bar chart"""
        # Calculate metrics
        accuracy = np.mean(y_true == y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        auc_score = auc(*roc_curve(y_true, y_proba)[:2])
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        values = [accuracy, precision, recall, f1, auc_score]
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightsteelblue']
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylim([0, 1.1])
        ax.set_ylabel('Score')
        ax.set_title(f'{model_name} - Classification Metrics')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels if needed
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_confidence_vs_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   y_proba: np.ndarray, ax: plt.Axes, model_name: str) -> None:
        """Plot confidence vs accuracy to show calibration"""
        # Create confidence bins
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_confidences = []
        bin_accuracies = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this confidence bin
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                
                bin_accuracies.append(accuracy_in_bin)
                bin_confidences.append(avg_confidence_in_bin)
                bin_counts.append(np.sum(in_bin))
            else:
                bin_accuracies.append(0)
                bin_confidences.append((bin_lower + bin_upper) / 2)
                bin_counts.append(0)
        
        # Plot reliability diagram
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        
        # Plot bins with size proportional to count
        sizes = np.array(bin_counts) * 100 / len(y_true)  # Scale for visibility
        scatter = ax.scatter(bin_confidences, bin_accuracies, s=sizes, 
                           alpha=0.7, c=bin_counts, cmap='viridis', edgecolors='black')
        
        # Add colorbar for counts
        cbar = plt.colorbar(scatter, ax=ax, label='Number of predictions')
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'{model_name} - Reliability Diagram')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    def plot_model_comparison(self, results_dict: Dict[str, Dict], 
                            save_filename: str = "model_comparison.png") -> None:
        """
        Compare multiple models side by side
        
        Args:
            results_dict: Dictionary with model_name: {y_true, y_pred, y_proba} 
            save_filename: Filename to save the comparison plot
        """
        n_models = len(results_dict)
        fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 10))
        
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        for idx, (model_name, results) in enumerate(results_dict.items()):
            y_true = results['y_true']
            y_pred = results['y_pred']
            y_proba = results['y_proba']
            
            # Confusion matrix
            self._plot_confusion_matrix(y_true, y_pred, axes[0, idx], model_name)
            
            # Precision-recall curve
            self._plot_precision_recall_curve(y_true, y_proba, axes[1, idx], model_name)
        
        plt.tight_layout()
        
        if save_filename:
            plt.savefig(os.path.join(self.save_dir, save_filename), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_object_analysis_enhanced(self, detections: List[Dict], 
                                    prediction: str, confidence: float,
                                    detailed_analysis: Dict,
                                    save_filename: str = None) -> None:
        """
        Enhanced object analysis visualization for object detection based classifier
        
        Args:
            detections: List of object detections
            prediction: Predicted class
            confidence: Confidence score
            detailed_analysis: Detailed analysis from object analyzer
            save_filename: Filename to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Object counts by type with enhanced styling
        chem_count = detailed_analysis['chemical_indicators']['count']
        bio_count = detailed_analysis['biological_indicators']['count']
        
        bars = ax1.bar(['Chemical', 'Biological'], [chem_count, bio_count], 
                      color=['#FF6B6B', '#4ECDC4'], alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax1.set_title('Object Counts by Type', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Objects', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Enhanced prediction visualization
        colors = ['#FF6B6B', '#4ECDC4']
        pred_values = [1-confidence if prediction == 'Biological' else confidence,
                      confidence if prediction == 'Biological' else 1-confidence]
        
        bars = ax2.bar(['Chemical', 'Biological'], pred_values, 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add confidence text with better styling
        pred_color = colors[1] if prediction == 'Biological' else colors[0]
        ax2.text(0.5, 0.95, f'{prediction}: {confidence:.1%}', 
                transform=ax2.transAxes, ha='center', va='top', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor=pred_color, alpha=0.3),
                fontsize=14, fontweight='bold')
        
        ax2.set_title('Classification Result', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Confidence', fontsize=12)
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Weighted scores comparison with gradient
        chem_score = detailed_analysis['chemical_indicators']['weighted_score']
        bio_score = detailed_analysis['biological_indicators']['weighted_score']
        
        bars = ax3.bar(['Chemical Score', 'Biological Score'], [chem_score, bio_score],
                      color=['#FF6B6B', '#4ECDC4'], alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add score labels
        for bar, score in zip(bars, [chem_score, bio_score]):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{score:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax3.set_title('Weighted Object Scores', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Weighted Score', fontsize=12)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Enhanced dominant objects visualization
        dominant_objects = detailed_analysis['dominant_objects'][:6]  # Top 6
        if dominant_objects:
            objects = [obj['object'] for obj in dominant_objects]
            confidences = [obj['confidence'] for obj in dominant_objects]
            weights = [obj.get('weight', 0) for obj in dominant_objects]
            
            # Create color map based on object type and weight
            colors_bars = []
            for obj_name, weight in zip(objects, weights):
                if weight >= 0.7:  # Strong indicator
                    colors_bars.append('#FF4444' if 'chemical' in obj_name.lower() or 
                                     any(chem in obj_name.lower() for chem in ['gas', 'hazmat', 'chemical']) 
                                     else '#2E8B57')
                elif weight > 0:  # Moderate indicator
                    colors_bars.append('#FF7F7F' if 'chemical' in obj_name.lower() or 
                                     any(chem in obj_name.lower() for chem in ['gas', 'hazmat', 'chemical'])
                                     else '#90EE90')
                else:  # Neutral
                    colors_bars.append('#D3D3D3')
            
            bars = ax4.barh(objects, confidences, color=colors_bars, alpha=0.8, 
                           edgecolor='black', linewidth=1)
            
            # Add confidence labels
            for bar, conf in zip(bars, confidences):
                width = bar.get_width()
                ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                        f'{conf:.2f}', ha='left', va='center', fontweight='bold', fontsize=10)
            
            ax4.set_title('Dominant Objects (Top 6)', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Detection Confidence', fontsize=12)
            ax4.set_xlim(0, 1.1)
            ax4.grid(True, alpha=0.3, axis='x')
            
            # Add legend for colors
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#FF4444', label='Strong Chemical'),
                Patch(facecolor='#2E8B57', label='Strong Biological'),
                Patch(facecolor='#FF7F7F', label='Moderate Chemical'),
                Patch(facecolor='#90EE90', label='Moderate Biological'),
                Patch(facecolor='#D3D3D3', label='Neutral')
            ]
            ax4.legend(handles=legend_elements, loc='lower right', fontsize=9)
            
        else:
            ax4.text(0.5, 0.5, 'No objects detected', 
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=14, fontweight='bold')
            ax4.set_title('No Objects Detected', fontsize=14, fontweight='bold')
        
        # Overall styling
        plt.suptitle(f'Object Detection Analysis - {prediction} Classification', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_filename:
            plt.savefig(os.path.join(self.save_dir, save_filename), dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print detailed explanation
        print(f"\n{'='*60}")
        print(f"DETAILED CLASSIFICATION ANALYSIS")
        print(f"{'='*60}")
        print(f"Classification: {prediction}")
        print(f"Confidence: {confidence:.3f} ({confidence:.1%})")
        print(f"Total objects detected: {detailed_analysis['total_objects_detected']}")
        print(f"Method: {detailed_analysis.get('method', 'Rule-Based')}")
        print(f"\nExplanation: {detailed_analysis['confidence_explanation']}")
        print(f"{'='*60}")