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
    
def visualize_prediction(self, detections, prediction=None, confidence=None, save_path=None):
    """
    Visualize the prediction based on detected objects
    
    Args:
        detections: List of object detections
        prediction: Predicted class (will compute if None)
        confidence: Confidence score (will compute if None)
        save_path: Path to save visualization
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

def create_gradio_interface(predictor):
    """Create Gradio web interface for object-based predictions"""
    
    def predict_from_objects_text(objects_text, use_nn):
        """
        Function for Gradio interface - predict from text input of objects
        
        Args:
            objects_text: Text description of detected objects (one per line)
            use_nn: Whether to use neural network or rule-based
        """
        if not objects_text.strip():
            return "Please enter detected objects", None, ""
        
        try:
            # Parse objects text
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
            
            return result_text, prediction, objects_breakdown
            
        except Exception as e:
            return f"Error processing objects: {str(e)}", None, ""
    
    # Create interface
    with gr.Blocks(title="Image Classifier SSE - Object Detection Based") as interface:
        gr.Markdown("# Object-Based Environment Classifier")
        gr.Markdown("""
        This system classifies environments as **Chemical** or **Biological** based on detected objects.
        Enter the objects detected in your image (one per line) with optional confidence scores.
        """)
        
        with gr.Row():
            with gr.Column():
                objects_input = gr.Textbox(
                    label="Detected Objects (one per line)",
                    placeholder="gas mask 0.95\nhazmat suit 0.87\nchemical substance 0.82",
                    lines=8,
                    value="gas mask 0.95\nhazmat suit 0.87\nchemical substance 0.82"
                )
                
                use_nn_checkbox = gr.Checkbox(
                    label="Use Neural Network (if available)",
                    value=True
                )
                
                predict_btn = gr.Button("Classify Environment", variant="primary")
            
            with gr.Column():
                result_output = gr.Markdown(label="Classification Result")
                classification_output = gr.Label(label="Prediction")
                objects_breakdown = gr.Markdown(label="Objects Analysis")
        
        # Example inputs
        gr.Examples(
            examples=[
                ["gas mask 0.95\nhazmat suit 0.87\nchemical substance 0.82", True],
                ["petri dish 0.92\npipette 0.88\nbiohazard symbol 0.85", True],
                ["lab coat 0.80\ncomputer 0.75\ngloves 0.70", False],
            ],
            inputs=[objects_input, use_nn_checkbox],
            label="Example Inputs"
        )
        
        predict_btn.click(
            fn=predict_from_objects_text,
            inputs=[objects_input, use_nn_checkbox],
            outputs=[result_output, classification_output, objects_breakdown]
        )
    
    return interface

def main():
    """Main function for inference"""
    parser = argparse.ArgumentParser(description='Image Classifier SSE - Object Detection Based Inference')
    parser.add_argument('--model', type=str, 
                       default='models/saved_models/best_model.pth',
                       help='Path to trained neural network model')
    parser.add_argument('--coco', type=str,
                       help='Path to COCO annotation file for batch prediction')
    parser.add_argument('--image-id', type=int,
                       help='Specific image ID to predict (requires --coco)')
    parser.add_argument('--interface', action='store_true',
                       help='Launch Gradio web interface')
    parser.add_argument('--rule-based', action='store_true',
                       help='Use only rule-based classifier (no neural network)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--compare', action='store_true',
                       help='Compare neural network vs rule-based on test set')
    
    args = parser.parse_args()
    
    # Create predictor
    print("Initializing predictor...")
    use_nn = not args.rule_based
    predictor = SSEObjectPredictor(
        model_path=args.model if use_nn else None,
        config_path=args.config,
        use_neural_network=use_nn
    )
    
    if use_nn and predictor.neural_model is None:
        print("Neural network model not available, using rule-based classifier")
    
    if args.interface:
        # Launch Gradio interface
        print("Launching web interface...")
        interface = create_gradio_interface(predictor)
        interface.launch(share=True)
        
    elif args.coco:
        if args.image_id is not None:
            # Single image prediction from COCO
            print(f"Predicting for image ID {args.image_id} in {args.coco}")
            prediction, confidence, detections = predictor.predict_from_coco_annotation(
                args.coco, args.image_id
            )
            
            print(f"\nResults for Image ID {args.image_id}:")
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence:.3f} ({confidence:.1%})")
            print(f"Objects detected: {len(detections)}")
            
            if detections:
                print("Detected objects:")
                for det in detections:
                    print(f"  - {det['category_name']} (conf: {det['confidence']:.2f})")
                
                # Visualize
                predictor.visualize_prediction(detections, prediction, confidence)
            
        else:
            # Batch prediction from COCO
            print(f"Processing all images in {args.coco}")
            results_df = predictor.batch_predict_from_coco(args.coco)
            
            print(f"\nBatch Results:")
            print(f"Total images processed: {len(results_df)}")
            
            # Summary statistics
            prediction_counts = results_df['prediction'].value_counts()
            print(f"\nPrediction distribution:")
            for pred, count in prediction_counts.items():
                percentage = (count / len(results_df)) * 100
                print(f"  {pred}: {count} ({percentage:.1f}%)")
            
            print(f"\nAverage confidence: {results_df['confidence'].mean():.3f}")
            print(f"Average objects per image: {results_df['num_objects'].mean():.1f}")
            
            # Save results
            output_path = f"batch_predictions_{os.path.basename(args.coco).replace('.json', '')}.csv"
            results_df.to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_path}")
            
            # Show sample results
            print(f"\nSample results:")
            print(results_df[['image_filename', 'prediction', 'confidence', 'num_objects']].head(10))
    
    elif args.compare:
        # Compare models on test set
        print("Comparing neural network vs rule-based classifier...")
        
        # Load test data
        data_dir = predictor.config['data']['processed_data_path']
        test_csv = os.path.join(data_dir, 'test_features.csv')
        
        if not os.path.exists(test_csv):
            print(f"Error: Test data not found at {test_csv}")
            print("Please run data_preprocessing.py first")
            return
        
        test_df = pd.read_csv(test_csv)
        print(f"Test set size: {len(test_df)} images")
        
        # Test both methods
        results = []
        for _, row in test_df.iterrows():
            # Reconstruct detections from features (simplified)
            detections = []  # This is a limitation - we'd need original detections
            
            # For comparison, we'll use the saved features directly
            # This is a simplification for demonstration
            pass
        
        print("Note: Full comparison requires original detection data")
        print("Consider running the training script which includes model comparison")
    
    else:
        print("Please specify --coco, --interface, or --compare")
        print("Use --help for more information")
        
        # Show example usage
        print(f"\nExample usage:")
        print(f"  # Launch web interface:")
        print(f"  python {__file__} --interface")
        print(f"  ")
        print(f"  # Predict from COCO file:")
        print(f"  python {__file__} --coco data/annotations/_annotations.coco.json")
        print(f"  ")
        print(f"  # Predict specific image:")
        print(f"  python {__file__} --coco data/annotations/_annotations.coco.json --image-id 123")
        print(f"  ")
        print(f"  # Use only rule-based classifier:")
        print(f"  python {__file__} --coco data/annotations/_annotations.coco.json --rule-based")

if __name__ == "__main__":
    main()
    
"""
Inference script for Image Classifier SSE - Object Detection Based
Use trained model to classify images based on their detected objects
"""

import os
import torch
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import gradio as gr
import json
import joblib

from model import create_model, SimpleRuleBasedClassifier, ModelEvaluator
from object_analyzer import ObjectAnalyzer

class SSEObjectPredictor:
    """Class for making predictions based on object detection results"""
    
    def __init__(self, model_path=None, config_path="config.yaml", use_neural_network=True):
        """
        Initialize predictor
        
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
    
    def visualize_prediction(self, detections, prediction=None, confidence=None, save_path=None):
        """
        Visualize the prediction based on detected objects
        
        Args:
            detections: List of object detections
            prediction: Predicted class (will compute if None)
            confidence: Confidence score (will compute if None)
            save_path: Path to save visualization
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

def create_gradio_interface(predictor):
    """Create Gradio web interface for object-based predictions"""
    
    def predict_from_objects_text(objects_text, use_nn):
        """
        Function for Gradio interface - predict from text input of objects
        
        Args:
            objects_text: Text description of detected objects (one per line)
            use_nn: Whether to use neural network or rule-based
        """
        if not objects_text.strip():
            return "Please enter detected objects", None, ""
        
        try:
            # Parse objects text
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
            
            return result_text, prediction, objects_breakdown
            
        except Exception as e:
            return f"Error processing objects: {str(e)}", None, ""
    
    # Create interface
    with gr.Blocks(title="Image Classifier SSE - Object Detection Based") as interface:
        gr.Markdown("# Object-Based Environment Classifier")
        gr.Markdown("""
        This system classifies environments as **Chemical** or **Biological** based on detected objects.
        Enter the objects detected in your image (one per line) with optional confidence scores.
        """)
        
        with gr.Row():
            with gr.Column():
                objects_input = gr.Textbox(
                    label="Detected Objects (one per line)",
                    placeholder="gas mask 0.95\nhazmat suit 0.87\nchemical substance 0.82",
                    lines=8,
                    value="gas mask 0.95\nhazmat suit 0.87\nchemical substance 0.82"
                )
                
                use_nn_checkbox = gr.Checkbox(
                    label="Use Neural Network (if available)",
                    value=True
                )
                
                predict_btn = gr.Button("Classify Environment", variant="primary")
            
            with gr.Column():
                result_output = gr.Markdown(label="Classification Result")
                classification_output = gr.Label(label="Prediction")
                objects_breakdown = gr.Markdown(label="Objects Analysis")
        
        # Example inputs
        gr.Examples(
            examples=[
                ["gas mask 0.95\nhazmat suit 0.87\nchemical substance 0.82", True],
                ["petri dish 0.92\npipette 0.88\nbiohazard symbol 0.85", True],
                ["lab coat 0.80\ncomputer 0.75\ngloves 0.70", False],
            ],
            inputs=[objects_input, use_nn_checkbox],
            label="Example Inputs"
        )
        
        predict_btn.click(
            fn=predict_from_objects_text,
            inputs=[objects_input, use_nn_checkbox],
            outputs=[result_output, classification_output, objects_breakdown]
        )
    
    return interface

def main():
    """Main function for inference"""
    parser = argparse.ArgumentParser(description='Image Classifier SSE - Object Detection Based Inference')
    parser.add_argument('--model', type=str, 
                       default='models/saved_models/best_model.pth',
                       help='Path to trained neural network model')
    parser.add_argument('--coco', type=str,
                       help='Path to COCO annotation file for batch prediction')
    parser.add_argument('--image-id', type=int,
                       help='Specific image ID to predict (requires --coco)')
    parser.add_argument('--interface', action='store_true',
                       help='Launch Gradio web interface')
    parser.add_argument('--rule-based', action='store_true',
                       help='Use only rule-based classifier (no neural network)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--compare', action='store_true',
                       help='Compare neural network vs rule-based on test set')
    
    args = parser.parse_args()
    
    # Create predictor
    print("Initializing predictor...")
    use_nn = not args.rule_based
    predictor = SSEObjectPredictor(
        model_path=args.model if use_nn else None,
        config_path=args.config,
        use_neural_network=use_nn
    )
    
    if use_nn and predictor.neural_model is None:
        print("Neural network model not available, using rule-based classifier")
    
    if args.interface:
        # Launch Gradio interface
        print("Launching web interface...")
        interface = create_gradio_interface(predictor)
        interface.launch(share=True)
        
    elif args.coco:
        if args.image_id is not None:
            # Single image prediction from COCO
            print(f"Predicting for image ID {args.image_id} in {args.coco}")
            prediction, confidence, detections = predictor.predict_from_coco_annotation(
                args.coco, args.image_id
            )
            
            print(f"\nResults for Image ID {args.image_id}:")
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence:.3f} ({confidence:.1%})")
            print(f"Objects detected: {len(detections)}")
            
            if detections:
                print("Detected objects:")
                for det in detections:
                    print(f"  - {det['category_name']} (conf: {det['confidence']:.2f})")
                
                # Visualize
                predictor.visualize_prediction(detections, prediction, confidence)
            
        else:
            # Batch prediction from COCO
            print(f"Processing all images in {args.coco}")
            results_df = predictor.batch_predict_from_coco(args.coco)
            
            print(f"\nBatch Results:")
            print(f"Total images processed: {len(results_df)}")
            
            # Summary statistics
            prediction_counts = results_df['prediction'].value_counts()
            print(f"\nPrediction distribution:")
            for pred, count in prediction_counts.items():
                percentage = (count / len(results_df)) * 100
                print(f"  {pred}: {count} ({percentage:.1f}%)")
            
            print(f"\nAverage confidence: {results_df['confidence'].mean():.3f}")
            print(f"Average objects per image: {results_df['num_objects'].mean():.1f}")
            
            # Save results
            output_path = f"batch_predictions_{os.path.basename(args.coco).replace('.json', '')}.csv"
            results_df.to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_path}")
            
            # Show sample results
            print(f"\nSample results:")
            print(results_df[['image_filename', 'prediction', 'confidence', 'num_objects']].head(10))
    
    elif args.compare:
        # Compare models on test set
        print("Comparing neural network vs rule-based classifier...")
        
        # Load test data
        data_dir = predictor.config['data']['processed_data_path']
        test_csv = os.path.join(data_dir, 'test_features.csv')
        
        if not os.path.exists(test_csv):
            print(f"Error: Test data not found at {test_csv}")
            print("Please run data_preprocessing.py first")
            return
        
        test_df = pd.read_csv(test_csv)
        print(f"Test set size: {len(test_df)} images")
        
        print("Note: Full comparison requires original detection data")
        print("Consider running the training script which includes model comparison")
    
    else:
        print("Please specify --coco, --interface, or --compare")
        print("Use --help for more information")
        
        # Show example usage
        print(f"\nExample usage:")
        print(f"  # Launch web interface:")
        print(f"  python src/inference.py --interface")
        print(f"  ")
        print(f"  # Predict from COCO file:")
        print(f"  python src/inference.py --coco data/raw/annotations/_annotations.coco.json")
        print(f"  ")
        print(f"  # Predict specific image:")
        print(f"  python src/inference.py --coco data/raw/annotations/_annotations.coco.json --image-id 123")
        print(f"  ")
        print(f"  # Use only rule-based classifier:")
        print(f"  python src/inference.py --coco data/raw/annotations/_annotations.coco.json --rule-based")

if __name__ == "__main__":
    main()
    
"""
Inference script for Image Classifier SSE
Use trained model to classify new images
"""

import os
import torch
import yaml
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import gradio as gr
from model import create_model, get_transforms

class SSEPredictor:
    """Class for making predictions with trained model"""
    
    def __init__(self, model_path, config_path="config.yaml"):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to saved model checkpoint
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = create_model(self.config)
        self.model.to(self.device)
        
        # Load trained weights
        self.load_model(model_path)
        
        # Setup transforms
        image_size = tuple(self.config['data']['image_size'])
        self.transform = get_transforms(image_size, augment=False)
        
        # Class names
        self.class_names = ['Chemical', 'Biological']
        
        # Decision threshold
        self.threshold = self.config['output']['threshold']
    
    def load_model(self, model_path):
        """Load model from checkpoint"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from: {model_path}")
    
    def preprocess_image(self, image_path):
        """Preprocess image for inference"""
        # Load and convert image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            # Handle PIL Image object (for Gradio)
            image = image_path.convert('RGB')
        
        # Apply transforms
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        
        return image_tensor
    
    def predict(self, image_path, return_confidence=True):
        """
        Make prediction for a single image
        
        Args:
            image_path: Path to image or PIL Image object
            return_confidence: Whether to return confidence scores
            
        Returns:
            prediction: Predicted class name
            confidence: Confidence score (if return_confidence=True)
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            probability = output.item()
        
        # Convert to class prediction
        predicted_class_idx = int(probability > self.threshold)
        predicted_class = self.class_names[predicted_class_idx]
        
        # Calculate confidence
        confidence = probability if predicted_class_idx == 1 else (1 - probability)
        
        if return_confidence:
            return predicted_class, confidence
        else:
            return predicted_class
    
    def predict_batch(self, image_paths):
        """Make predictions for multiple images"""
        results = []
        
        for image_path in image_paths:
            try:
                prediction, confidence = self.predict(image_path)
                results.append({
                    'image_path': image_path,
                    'prediction': prediction,
                    'confidence': confidence
                })
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'prediction': 'Error',
                    'confidence': 0.0,
                    'error': str(e)
                })
        
        return results
    
    def visualize_prediction(self, image_path, save_path=None):
        """Visualize prediction with image"""
        # Make prediction
        prediction, confidence = self.predict(image_path)
        
        # Load original image
        if isinstance(image_path, str):
            image = Image.open(image_path)
        else:
            image = image_path
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Input Image')
        plt.axis('off')
        
        # Prediction
        plt.subplot(1, 2, 2)
        colors = ['red', 'blue']
        color = colors[1] if prediction == 'Biological' else colors[0]
        
        plt.bar(['Chemical', 'Biological'], 
                [1-confidence if prediction == 'Biological' else confidence,
                 confidence if prediction == 'Biological' else 1-confidence],
                color=['red', 'blue'],
                alpha=0.7)
        
        plt.title(f'Prediction: {prediction}\nConfidence: {confidence:.3f}')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        
        # Add confidence text
        plt.text(0.5, 0.95, f'{prediction}: {confidence:.1%}', 
                transform=plt.gca().transAxes, 
                ha='center', va='top', 
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
                fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return prediction, confidence

def create_gradio_interface(predictor):
    """Create Gradio web interface for the model"""
    
    def predict_and_visualize(image):
        """Function for Gradio interface"""
        if image is None:
            return "Please upload an image", None
        
        try:
            prediction, confidence = predictor.predict(image)
            
            # Create result text
            result_text = f"""
            **Prediction:** {prediction}
            **Confidence:** {confidence:.1%}
            
            **Interpretation:**
            • This image is classified as **{prediction.lower()}** related
            • The model is {confidence:.1%} confident in this prediction
            • Threshold used: {predictor.threshold}
            """
            
            return result_text, prediction
            
        except Exception as e:
            return f"Error processing image: {str(e)}", None
    
    # Create interface
    interface = gr.Interface(
        fn=predict_and_visualize,
        inputs=gr.Image(type="pil", label="Upload Image for Classification"),
        outputs=[
            gr.Markdown(label="Prediction Results"),
            gr.Label(label="Classification")
        ],
        title="Image Classifier SSE",
        description="""
        Upload an image to classify it as either:
        - **Chemical**: Related to chemical handling, storage, or safety
        - **Biological**: Related to biological research or safety
        
        The model analyzes the objects in the image to make this determination.
        """,
        examples=[
            # Add example image paths here if you have them
        ],
        theme="default"
    )
    
    return interface

def main():
    """Main function for inference"""
    parser = argparse.ArgumentParser(description='Image Classifier SSE - Inference')
    parser.add_argument('--model', type=str, 
                       default='models/saved_models/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--image', type=str, 
                       help='Path to single image for prediction')
    parser.add_argument('--batch', type=str, 
                       help='Path to directory containing images for batch prediction')
    parser.add_argument('--interface', action='store_true',
                       help='Launch Gradio web interface')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Create predictor
    print("Loading model...")
    predictor = SSEPredictor(args.model, args.config)
    print("Model loaded successfully!")
    
    if args.interface:
        # Launch Gradio interface
        print("Launching web interface...")
        interface = create_gradio_interface(predictor)
        interface.launch(share=True)
        
    elif args.image:
        # Single image prediction
        print(f"Predicting for image: {args.image}")
        prediction, confidence = predictor.predict(args.image)
        
        print(f"\nResults:")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.3f} ({confidence:.1%})")
        
        # Visualize result
        predictor.visualize_prediction(args.image)
        
    elif args.batch:
        # Batch prediction
        print(f"Processing images in directory: {args.batch}")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for file in os.listdir(args.batch):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(args.batch, file))
        
        if not image_files:
            print(f"\nSummary:")
            print(f"Chemical images: {chemical_count}")
            print(f"Biological images: {biological_count}")
            print(f"Total processed: {len(results)}")
    
    else:
        print("Please specify --image, --batch, or --interface")
        print("Use --help for more information")

if __name__ == "__main__":
    main()