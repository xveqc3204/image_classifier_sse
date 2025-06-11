"""
Enhanced Inference script for Image Classifier SSE - Object Detection Based
Now includes comprehensive visualizations with confusion matrix and precision-recall curves
"""

import os
import sys
import torch
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import gradio as gr
import json
import joblib
import torch.serialization

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import create_model, SimpleRuleBasedClassifier, ModelEvaluator
from object_analyzer import ObjectAnalyzer
from enhanced_visualizations import EnhancedVisualizer

class SSEObjectPredictor:
    """Enhanced class for making predictions based on object detection results"""
    
    def __init__(self, model_path=None, config_path="config.yaml", use_neural_network=True):
        """
        Initialize enhanced predictor
        
        Args:
            model_path: Path to saved neural network model (optional)
            config_path: Path to configuration file
            use_neural_network: Whether to use neural network or rule-based classifier
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize object analyzer
        self.analyzer = ObjectAnalyzer(config_path)
        
        # Initialize enhanced visualizer
        self.visualizer = EnhancedVisualizer(save_dir="inference_visualizations")
        
        # Decision threshold
        self.threshold = self.config['output']['threshold']
        
        # Set up models
        self.use_neural_network = use_neural_network
        self.neural_model = None
        self.scaler = None
        self.rule_classifier = SimpleRuleBasedClassifier(threshold=self.threshold)
        
        if use_neural_network and model_path:
            self.load_neural_model(model_path)
        
        # Class names
        self.class_names = ['Chemical', 'Biological']
    
    def load_neural_model(self, model_path):
        """Load neural network model from checkpoint"""
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found: {model_path}")
            print("Falling back to rule-based classifier")
            self.use_neural_network = False
            return
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        with torch.serialization.safe_globals([torch.serialization.safe_globals]):
            checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create and load model
        self.neural_model = create_model(self.config)
        self.neural_model.load_state_dict(checkpoint['model_state_dict'])
        self.neural_model.to(self.device)
        self.neural_model.eval()
        
        # Load scaler
        self.scaler = checkpoint.get('scaler', None)
        if self.scaler is None:
            # Try to load scaler separately
            scaler_path = os.path.join(
                self.config['data']['processed_data_path'], 
                'feature_scaler.pkl'
            )
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
        
        print(f"Neural network model loaded from: {model_path}")
    
    def predict_from_detections(self, detections, return_detailed=False):
        """
        Make prediction from object detection results
        
        Args:
            detections: List of detection dictionaries with keys:
                       'category_name', 'confidence', 'bbox', 'area'
            return_detailed: Whether to return detailed analysis
            
        Returns:
            prediction: Predicted class name
            confidence: Confidence score
            detailed_analysis: Detailed analysis (if return_detailed=True)
        """
        
        if self.use_neural_network and self.neural_model is not None:
            return self._predict_neural_network(detections, return_detailed)
        else:
            return self._predict_rule_based(detections, return_detailed)
    
    def _predict_neural_network(self, detections, return_detailed=False):
        """Predict using neural network"""
        # Extract features
        features = self.analyzer.extract_object_features(detections)
        
        # Convert to array (in correct order)
        feature_array = np.array([
            features['total_objects'], features['chemical_objects'], 
            features['biological_objects'], features['neutral_objects'],
            features['avg_chemical_confidence'], features['avg_biological_confidence'],
            features['max_chemical_confidence'], features['max_biological_confidence'],
            features['chemical_weighted_score'], features['biological_weighted_score'],
            features['chemical_area_ratio'], features['biological_area_ratio'],
            features['chemical_object_ratio'], features['biological_object_ratio']
        ], dtype=np.float32)
        
        # Use neural network evaluator
        evaluator = ModelEvaluator(self.neural_model, self.device, self.scaler)
        prediction, confidence, probability = evaluator.evaluate_single_features(
            feature_array, threshold=self.threshold
        )
        
        if return_detailed:
            # Get additional analysis from rule-based method
            _, _, detailed_analysis = self.analyzer.classify_image_advanced(detections)
            detailed_analysis['method'] = 'Neural Network'
            detailed_analysis['raw_probability'] = probability
            return prediction, confidence, detailed_analysis
        
        return prediction, confidence
    
    def _predict_rule_based(self, detections, return_detailed=False):
        """Predict using rule-based classifier"""
        if return_detailed:
            return self.analyzer.classify_image_advanced(detections)
        else:
            prediction, confidence = self.analyzer.classify_image_simple(detections)
            return prediction, confidence
    
    def predict_from_coco_annotation(self, annotation_file, image_id):
        """
        Predict from COCO annotation file for a specific image
        
        Args:
            annotation_file: Path to COCO JSON file
            image_id: ID of image in COCO dataset
            
        Returns:
            prediction, confidence, detections_used
        """
        # Load COCO data
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create category mapping
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Find annotations for this image
        detections = []
        for ann in coco_data['annotations']:
            if ann['image_id'] == image_id:
                bbox = ann['bbox']
                area = bbox[2] * bbox[3] if len(bbox) >= 4 else ann.get('area', 0)
                
                detection = {
                    'category_name': categories[ann['category_id']],
                    'confidence': 1.0,  # COCO ground truth
                    'bbox': bbox,
                    'area': area
                }
                detections.append(detection)
        
        if not detections:
            return "Unknown", 0.0, []
        
        prediction, confidence = self.predict_from_detections(detections)
        return prediction, confidence, detections
    
    def batch_predict_from_coco(self, annotation_file):
        """
        Predict for all images in a COCO annotation file
        
        Args:
            annotation_file: Path to COCO JSON file
            
        Returns:
            DataFrame with results
        """
        print(f"Processing COCO file: {annotation_file}")
        
        # Load COCO data
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create mappings
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        images = {img['id']: img['file_name'] for img in coco_data['images']}
        
        # Group annotations by image
        image_detections = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in image_detections:
                image_detections[image_id] = []
            
            bbox = ann['bbox']
            area = bbox[2] * bbox[3] if len(bbox) >= 4 else ann.get('area', 0)
            
            detection = {
                'category_name': categories[ann['category_id']],
                'confidence': 1.0,
                'bbox': bbox,
                'area': area
            }
            image_detections[image_id].append(detection)
        
        # Process all images
        results = []
        for image_id, detections in image_detections.items():
            prediction, confidence, detailed = self.predict_from_detections(
                detections, return_detailed=True
            )
            
            result = {
                'image_id': image_id,
                'image_filename': images.get(image_id, f"image_{image_id}"),
                'prediction': prediction,
                'confidence': confidence,
                'num_objects': len(detections),
                'method': detailed.get('method', 'Rule-Based'),
                'dominant_objects': ', '.join([obj['object'] for obj in detailed['dominant_objects'][:3]])
            }
            results.append(result)
        
        return pd.DataFrame(results)
    
    def enhanced_visualize_prediction(self, detections, prediction=None, confidence=None, 
                                    save_path=None, show_detailed=True):
        """
        Enhanced visualization of the prediction with comprehensive analysis
        
        Args:
            detections: List of object detections
            prediction: Predicted class (will compute if None)
            confidence: Confidence score (will compute if None)
            save_path: Path to save visualization
            show_detailed: Whether to show detailed analysis
        """
        if prediction is None or confidence is None:
            prediction, confidence, detailed = self.predict_from_detections(
                detections, return_detailed=True
            )
        else:
            _, _, detailed = self.predict_from_detections(detections, return_detailed=True)
        
        if show_detailed:
            # Use enhanced visualizer for comprehensive analysis
            self.visualizer.plot_object_analysis_enhanced(
                detections, prediction, confidence, detailed, save_path
            )
        else:
            # Use original visualization method
            self.visualize_prediction(detections, prediction, confidence, save_path)
        
        return prediction, confidence
    
    def visualize_prediction(self, detections, prediction=None, confidence=None, save_path=None):
        """
        Original visualization method (kept for compatibility)
        """
        if prediction is None or confidence is None:
            prediction, confidence, detailed = self.predict_from_detections(
                detections, return_detailed=True
            )
        else:
            _, _, detailed = self.predict_from_detections(detections, return_detailed=True)
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Object counts by type
        chem_count = detailed['chemical_indicators']['count']
        bio_count = detailed['biological_indicators']['count']
        
        ax1.bar(['Chemical', 'Biological'], [chem_count, bio_count], 
                color=['red', 'blue'], alpha=0.7)
        ax1.set_title('Object Counts by Type')
        ax1.set_ylabel('Number of Objects')
        
        # 2. Prediction result
        colors = ['red', 'blue']
        color = colors[1] if prediction == 'Biological' else colors[0]
        
        ax2.bar(['Chemical', 'Biological'], 
                [1-confidence if prediction == 'Biological' else confidence,
                 confidence if prediction == 'Biological' else 1-confidence],
                color=['red', 'blue'], alpha=0.7)
        ax2.set_title(f'Prediction: {prediction}')
        ax2.set_ylabel('Confidence')
        ax2.set_ylim(0, 1)
        
        # Add confidence text
        ax2.text(0.5, 0.95, f'{prediction}: {confidence:.1%}', 
                transform=ax2.transAxes, ha='center', va='top', 
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
                fontsize=12, fontweight='bold')
        
        # 3. Weighted scores comparison
        chem_score = detailed['chemical_indicators']['weighted_score']
        bio_score = detailed['biological_indicators']['weighted_score']
        
        ax3.bar(['Chemical Score', 'Biological Score'], [chem_score, bio_score],
                color=['red', 'blue'], alpha=0.7)
        ax3.set_title('Weighted Scores')
        ax3.set_ylabel('Score')
        
        # 4. Dominant objects
        dominant_objects = detailed['dominant_objects'][:5]  # Top 5
        if dominant_objects:
            objects = [obj['object'] for obj in dominant_objects]
            confidences = [obj['confidence'] for obj in dominant_objects]
            
            # Color bars by object type
            colors_bars = []
            for obj_name in objects:
                if obj_name in self.analyzer.object_types:
                    obj_type = self.analyzer.object_types[obj_name]
                    if obj_type == 0.0:  # Chemical
                        colors_bars.append('red')
                    elif obj_type == 1.0:  # Biological
                        colors_bars.append('blue')
                    else:  # Neutral
                        colors_bars.append('gray')
                else:
                    colors_bars.append('gray')
            
            ax4.barh(objects, confidences, color=colors_bars, alpha=0.7)
            ax4.set_title('Dominant Objects (Top 5)')
            ax4.set_xlabel('Detection Confidence')
        else:
            ax4.text(0.5, 0.5, 'No objects detected', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('No Objects Detected')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print detailed explanation
        print(f"\n=== Prediction Analysis ===")
        print(f"Classification: {prediction}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Total objects detected: {detailed['total_objects_detected']}")
        print(f"Explanation: {detailed['confidence_explanation']}")
        print(f"Method: {detailed.get('method', 'Rule-Based')}")
        
        return prediction, confidence
    
    def batch_analysis_with_visualization(self, annotation_file, save_summary=True):
        """
        Perform batch analysis with comprehensive visualizations
        
        Args:
            annotation_file: Path to COCO annotation file
            save_summary: Whether to save summary visualizations
            
        Returns:
            DataFrame with results and creates summary visualizations
        """
        print(f"Performing comprehensive batch analysis...")
        
        # Get batch predictions
        results_df = self.batch_predict_from_coco(annotation_file)
        
        print(f"Analysis complete! Results summary:")
        print(f"   Total images processed: {len(results_df)}")
        
        # Summary statistics
        prediction_counts = results_df['prediction'].value_counts()
        print(f"   Prediction distribution:")
        for pred, count in prediction_counts.items():
            percentage = (count / len(results_df)) * 100
            print(f"     {pred}: {count} ({percentage:.1f}%)")
        
        print(f"   Average confidence: {results_df['confidence'].mean():.3f}")
        print(f"   Average objects per image: {results_df['num_objects'].mean():.1f}")
        
        if save_summary:
            self._create_batch_summary_visualizations(results_df, annotation_file)
        
        return results_df
    
    def _create_batch_summary_visualizations(self, results_df, annotation_file):
        """Create summary visualizations for batch analysis"""
        print(f"Creating batch summary visualizations...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Prediction distribution pie chart
        prediction_counts = results_df['prediction'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4', '#95A5A6']  # Red, teal, gray
        
        wedges, texts, autotexts = ax1.pie(prediction_counts.values, labels=prediction_counts.index, 
                                          autopct='%1.1f%%', colors=colors[:len(prediction_counts)],
                                          startangle=90, explode=[0.05]*len(prediction_counts))
        
        # Enhance text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        ax1.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
        
        # 2. Confidence distribution histogram
        for pred_class in prediction_counts.index:
            class_confidences = results_df[results_df['prediction'] == pred_class]['confidence']
            color = '#FF6B6B' if pred_class == 'Chemical' else '#4ECDC4' if pred_class == 'Biological' else '#95A5A6'
            ax2.hist(class_confidences, bins=20, alpha=0.7, label=pred_class, 
                    color=color, edgecolor='black', linewidth=0.5)
        
        ax2.set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Confidence Score', fontsize=12)
        ax2.set_ylabel('Number of Images', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Objects per image vs confidence
        scatter = ax3.scatter(results_df['num_objects'], results_df['confidence'], 
                             c=results_df['prediction'].map({'Chemical': 0, 'Biological': 1, 'Unknown': 0.5}),
                             cmap='RdYlBu', alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        
        ax3.set_title('Objects per Image vs Confidence', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Number of Objects Detected', fontsize=12)
        ax3.set_ylabel('Confidence Score', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Prediction Type', fontsize=11)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['Chemical', 'Unknown', 'Biological'])
        
        # 4. Method usage (if applicable)
        if 'method' in results_df.columns:
            method_counts = results_df['method'].value_counts()
            bars = ax4.bar(method_counts.index, method_counts.values, 
                          color=['#3498DB', '#E74C3C'], alpha=0.8, 
                          edgecolor='black', linewidth=1)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{int(height)}', ha='center', va='bottom', 
                        fontweight='bold', fontsize=12)
            
            ax4.set_title('Classification Method Usage', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Number of Images', fontsize=12)
            ax4.grid(True, alpha=0.3, axis='y')
        else:
            # Show average confidence by prediction type
            avg_conf_by_pred = results_df.groupby('prediction')['confidence'].mean()
            bars = ax4.bar(avg_conf_by_pred.index, avg_conf_by_pred.values,
                          color=['#FF6B6B', '#4ECDC4', '#95A5A6'][:len(avg_conf_by_pred)], 
                          alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add value labels
            for bar, conf in zip(bars, avg_conf_by_pred.values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{conf:.3f}', ha='center', va='bottom', 
                        fontweight='bold', fontsize=12)
            
            ax4.set_title('Average Confidence by Prediction', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Average Confidence', fontsize=12)
            ax4.set_ylim(0, 1.1)
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Batch Analysis Summary - {os.path.basename(annotation_file)}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save summary
        summary_path = os.path.join(self.visualizer.save_dir, 
                                   f"batch_summary_{os.path.basename(annotation_file).replace('.json', '')}.png")
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Summary visualization saved to: {summary_path}")

def create_enhanced_gradio_interface(predictor):
    """Create enhanced Gradio web interface with visualization options"""
    
    def predict_from_objects_text(objects_text, use_nn, show_detailed_viz):
        """
        Enhanced function for Gradio interface with visualization options
        """
        if not objects_text.strip():
            return "Please enter detected objects", None, "", None
        
        try:
            # Parse objects text (same as before)
            lines = objects_text.strip().split('\n')
            detections = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Parse format: "object_name confidence" or just "object_name"
                parts = line.split()
                if len(parts) >= 2:
                    obj_name = ' '.join(parts[:-1])
                    try:
                        confidence = float(parts[-1])
                    except:
                        obj_name = line
                        confidence = 0.9
                else:
                    obj_name = line
                    confidence = 0.9
                
                detection = {
                    'category_name': obj_name,
                    'confidence': confidence,
                    'bbox': [0, 0, 100, 100],  # Dummy bbox
                    'area': 10000  # Dummy area
                }
                detections.append(detection)
            
            # Set predictor mode
            predictor.use_neural_network = use_nn and predictor.neural_model is not None
            
            # Make prediction
            prediction, confidence, detailed = predictor.predict_from_detections(
                detections, return_detailed=True
            )
            
            # Create result text
            result_text = f"""
            **Prediction:** {prediction}
            **Confidence:** {confidence:.1%}
            **Method:** {detailed.get('method', 'Rule-Based')}
            
            **Objects Analyzed:** {len(detections)}
            - Chemical objects: {detailed['chemical_indicators']['count']}
            - Biological objects: {detailed['biological_indicators']['count']}
            
            **Explanation:** {detailed['confidence_explanation']}
            """
            
            # Create objects breakdown
            objects_breakdown = "**Detected Objects:**\n"
            for detection in detections:
                obj_name = detection['category_name']
                obj_conf = detection['confidence']
                obj_type = predictor.analyzer.object_types.get(obj_name, 0.5)
                type_name = "Chemical" if obj_type == 0.0 else "Biological" if obj_type == 1.0 else "Neutral"
                objects_breakdown += f"- {obj_name} ({obj_conf:.2f}) → {type_name}\n"
            
            # Create visualization
            visualization_plot = None
            if show_detailed_viz:
                try:
                    # Create a temporary plot and return it
                    import matplotlib
                    matplotlib.use('Agg')  # Use non-interactive backend
                    
                    # Use enhanced visualization
                    predictor.enhanced_visualize_prediction(
                        detections, prediction, confidence, 
                        save_path="temp_gradio_viz.png", show_detailed=True
                    )
                    
                    visualization_plot = "temp_gradio_viz.png"
                except Exception as e:
                    print(f"Visualization error: {e}")
                    visualization_plot = None
            
            return result_text, prediction, objects_breakdown, visualization_plot
            
        except Exception as e:
            return f"Error processing objects: {str(e)}", None, "", None
    
    # Create enhanced interface
    with gr.Blocks(title="Enhanced Image Classifier SSE - Object Detection Based") as interface:
        gr.Markdown("# Enhanced Object-Based Environment Classifier")
        gr.Markdown("""
        This enhanced system classifies environments as **Chemical** or **Biological** based on detected objects.
        Enter the objects detected in your image (one per line) with optional confidence scores.
        
        **Features:**
        - Neural Network and Rule-Based classification
        - Enhanced visualizations with detailed analysis
        - Comprehensive object analysis
        """)
        
        with gr.Row():
            with gr.Column():
                objects_input = gr.Textbox(
                    label="Detected Objects (one per line)",
                    placeholder="gas mask 0.95\nhazmat suit 0.87\nchemical substance 0.82",
                    lines=8,
                    value="gas mask 0.95\nhazmat suit 0.87\nchemical substance 0.82"
                )
                
                with gr.Row():
                    use_nn_checkbox = gr.Checkbox(
                        label="Use Neural Network (if available)",
                        value=True
                    )
                    
                    show_viz_checkbox = gr.Checkbox(
                        label="Show Enhanced Visualization",
                        value=True
                    )
                
                predict_btn = gr.Button("Classify Environment", variant="primary", size="lg")
            
            with gr.Column():
                result_output = gr.Markdown(label="Classification Result")
                classification_output = gr.Label(label="Prediction")
                objects_breakdown = gr.Markdown(label="Objects Analysis")
                
                # Visualization output
                visualization_output = gr.Image(label="Analysis Visualization", type="filepath")
        
        # Example inputs
        gr.Examples(
            examples=[
                ["gas mask 0.95\nhazmat suit 0.87\nchemical substance 0.82", True, True],
                ["petri dish 0.92\npipette 0.88\nbiohazard symbol 0.85", True, True],
                ["lab coat 0.80\ncomputer 0.75\ngloves 0.70", False, False],
                ["chemical substance 0.90\npetri dish 0.85\nlab coat 0.75", True, True],  # Mixed case
            ],
            inputs=[objects_input, use_nn_checkbox, show_viz_checkbox],
            label="Example Inputs"
        )
        
        predict_btn.click(
            fn=predict_from_objects_text,
            inputs=[objects_input, use_nn_checkbox, show_viz_checkbox],
            outputs=[result_output, classification_output, objects_breakdown, visualization_output]
        )
    
    return interface

def main():
    """Enhanced main function for inference with comprehensive analysis options"""
    parser = argparse.ArgumentParser(description='Enhanced Image Classifier SSE - Object Detection Based Inference')
    parser.add_argument('--model', type=str, 
                       default='models/saved_models/best_model.pth',
                       help='Path to trained neural network model')
    parser.add_argument('--coco', type=str,
                       help='Path to COCO annotation file for batch prediction')
    parser.add_argument('--image-id', type=int,
                       help='Specific image ID to predict (requires --coco)')
    parser.add_argument('--interface', action='store_true',
                       help='Launch enhanced Gradio web interface')
    parser.add_argument('--rule-based', action='store_true',
                       help='Use only rule-based classifier (no neural network)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--enhanced-viz', action='store_true',
                       help='Use enhanced visualizations (default: True)')
    parser.add_argument('--batch-analysis', action='store_true',
                       help='Perform comprehensive batch analysis with visualizations')
    
    args = parser.parse_args()
    
    # Create enhanced predictor
    print("Initializing enhanced predictor...")
    use_nn = not args.rule_based
    predictor = SSEObjectPredictor(
        model_path=args.model if use_nn else None,
        config_path=args.config,
        use_neural_network=use_nn
    )
    
    if use_nn and predictor.neural_model is None:
        print("Warning: Neural network model not available, using rule-based classifier")
    
    if args.interface:
        # Launch enhanced Gradio interface
        print("Launching enhanced web interface...")
        interface = create_enhanced_gradio_interface(predictor)
        interface.launch(share=True)
        
    elif args.coco:
        if args.image_id is not None:
            # Single image prediction with enhanced visualization
            print(f"Predicting for image ID {args.image_id} in {args.coco}")
            prediction, confidence, detections = predictor.predict_from_coco_annotation(
                args.coco, args.image_id
            )
            
            print(f"\nResults for Image ID {args.image_id}:")
            print(f"   Prediction: {prediction}")
            print(f"   Confidence: {confidence:.3f} ({confidence:.1%})")
            print(f"   Objects detected: {len(detections)}")
            
            if detections:
                print("   Detected objects:")
                for det in detections:
                    print(f"     - {det['category_name']} (conf: {det['confidence']:.2f})")
                
                # Enhanced visualization
                if args.enhanced_viz:
                    print("Creating enhanced visualization...")
                    predictor.enhanced_visualize_prediction(
                        detections, prediction, confidence, 
                        save_path=f"prediction_image_{args.image_id}.png"
                    )
                else:
                    predictor.visualize_prediction(detections, prediction, confidence)
            
        elif args.batch_analysis:
            # Comprehensive batch analysis
            print(f"Performing comprehensive batch analysis...")
            results_df = predictor.batch_analysis_with_visualization(args.coco)
            
            # Save detailed results
            output_path = f"comprehensive_analysis_{os.path.basename(args.coco).replace('.json', '')}.csv"
            results_df.to_csv(output_path, index=False)
            print(f"Detailed results saved to: {output_path}")
            
        else:
            # Standard batch prediction
            print(f"Processing all images in {args.coco}")
            results_df = predictor.batch_predict_from_coco(args.coco)
            
            print(f"\nBatch Results:")
            print(f"   Total images processed: {len(results_df)}")
            
            # Summary statistics
            prediction_counts = results_df['prediction'].value_counts()
            print(f"\n   Prediction distribution:")
            for pred, count in prediction_counts.items():
                percentage = (count / len(results_df)) * 100
                print(f"     {pred}: {count} ({percentage:.1f}%)")
            
            print(f"\n   Average confidence: {results_df['confidence'].mean():.3f}")
            print(f"   Average objects per image: {results_df['num_objects'].mean():.1f}")
            
            # Save results
            output_path = f"batch_predictions_{os.path.basename(args.coco).replace('.json', '')}.csv"
            results_df.to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_path}")
            
            # Show sample results
            print(f"\nSample results:")
            print(results_df[['image_filename', 'prediction', 'confidence', 'num_objects']].head(10))
    
    else:
        print("Please specify --coco, --interface, or use --help for more information")
        
        # Show enhanced example usage
        print(f"\nEnhanced usage examples:")
        print(f"   # Launch enhanced web interface:")
        print(f"   python src/inference.py --interface")
        print(f"   ")
        print(f"   # Comprehensive batch analysis:")
        print(f"   python src/inference.py --coco data/raw/annotations/_annotations.coco.json --batch-analysis")
        print(f"   ")
        print(f"   # Single image with enhanced visualization:")
        print(f"   python src/inference.py --coco data/raw/annotations/_annotations.coco.json --image-id 123 --enhanced-viz")
        print(f"   ")
        print(f"   # Standard batch prediction:")
        print(f"   python src/inference.py --coco data/raw/annotations/_annotations.coco.json")

if __name__ == "__main__":
    main()