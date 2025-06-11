"""
Object Analyzer for Image Classifier SSE
Analyzes detected objects and their properties to determine environment type
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import yaml

class ObjectAnalyzer:
    """
    Analyzes detected objects to classify environment as Chemical vs Biological
    
    This class takes COCO detection results and converts them into environment classifications
    by analyzing the types of objects detected, their confidence scores, and spatial properties.
    """
    
    def __init__(self, config_path="config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Object category weights and mappings
        self.setup_object_mappings()
        
    def setup_object_mappings(self):
        """
        Setup object category mappings and weights
        
        TO ADD NEW OBJECTS:
        1. Add object name and weight to appropriate category below
        2. Weights: 1.0 = strong indicator, 0.5-0.8 = moderate, 0.0 = neutral
        
        TO ADJUST WEIGHTS:
        1. Change the numerical values (0.0 to 1.0)
        2. Higher values = stronger influence on classification
        
        TO MOVE OBJECTS BETWEEN CATEGORIES:
        1. Cut object from one dictionary and paste to another
        2. Adjust weight as needed
        """
        
        # === CHEMICAL INDICATORS ===
        # Strong indicators (high confidence in classification)
        self.strong_chemical_objects = {
            "chemical substance": 1.0,
            "gas mask": 1.0, 
            "gas tank": 1.0,
            "hazmat suit": 1.0
        }
        
        # Moderate indicators (medium confidence)
        self.moderate_chemical_objects = {
            "fume_hood": 0.7,
            "ventilation": 0.6
        }
        
        # === BIOLOGICAL INDICATORS ===
        # Strong indicators (high confidence in classification)
        self.strong_biological_objects = {
            "biohazard symbol": 1.0,
            "petri dish": 1.0,
            "pipette": 1.0  
        }
        
        # Moderate indicators (medium confidence)
        self.moderate_biological_objects = {
            # Currently no moderate biological indicators
            # Add here if needed: "object_name": weight_value
        }
        
        # === NEUTRAL OBJECTS ===
        # Neutral objects (don't indicate specific environment - used in both)
        self.neutral_objects = {
            "computer": 0.0,
            "tech equipment": 0.0,
            "gloves": 0.0,
            "glassware": 0.0,      # Used in both chemical and biological labs
            "lab coat": 0.0,       # Used in both chemical and biological labs  
            "eye protection": 0.0  # Used in both chemical and biological labs
        }
        
        # Create combined mappings
        self.object_weights = {}
        self.object_weights.update(self.strong_chemical_objects)
        self.object_weights.update(self.moderate_chemical_objects)
        self.object_weights.update(self.strong_biological_objects)
        self.object_weights.update(self.moderate_biological_objects)
        self.object_weights.update(self.neutral_objects)
        
        # Create type mappings (chemical=0, biological=1, neutral=0.5)
        self.object_types = {}
        
        for obj in self.strong_chemical_objects:
            self.object_types[obj] = 0.0  # Chemical
        for obj in self.moderate_chemical_objects:
            self.object_types[obj] = 0.0  # Chemical
            
        for obj in self.strong_biological_objects:
            self.object_types[obj] = 1.0  # Biological
        for obj in self.moderate_biological_objects:
            self.object_types[obj] = 1.0  # Biological
            
        for obj in self.neutral_objects:
            self.object_types[obj] = 0.5  # Neutral
    
    def extract_object_features(self, detections: List[Dict]) -> Dict:
        """
        Extract features from object detections for a single image
        
        Args:
            detections: List of detection dictionaries with keys:
                       'category_name', 'confidence', 'bbox', 'area'
        
        Returns:
            Dictionary of features for classification
        """
        if not detections:
            return self._get_empty_features()
        
        features = {}
        
        # Basic counts
        features['total_objects'] = len(detections)
        features['chemical_objects'] = 0
        features['biological_objects'] = 0
        features['neutral_objects'] = 0
        
        # Confidence scores
        chemical_confidences = []
        biological_confidences = []
        
        # Weighted scores
        chemical_weighted_score = 0.0
        biological_weighted_score = 0.0
        
        # Area analysis
        total_area = 0
        chemical_area = 0
        biological_area = 0
        
        for detection in detections:
            obj_name = detection['category_name']
            confidence = detection['confidence']
            area = detection.get('area', 0)
            
            total_area += area
            
            # Get object type and weight
            obj_type = self.object_types.get(obj_name, 0.5)  # Default neutral
            obj_weight = self.object_weights.get(obj_name, 0.0)
            
            # Count objects by type
            if obj_type == 0.0:  # Chemical
                features['chemical_objects'] += 1
                chemical_confidences.append(confidence)
                chemical_weighted_score += confidence * obj_weight
                chemical_area += area
            elif obj_type == 1.0:  # Biological
                features['biological_objects'] += 1
                biological_confidences.append(confidence)
                biological_weighted_score += confidence * obj_weight
                biological_area += area
            else:  # Neutral
                features['neutral_objects'] += 1
        
        # Statistical features
        features['avg_chemical_confidence'] = np.mean(chemical_confidences) if chemical_confidences else 0.0
        features['avg_biological_confidence'] = np.mean(biological_confidences) if biological_confidences else 0.0
        features['max_chemical_confidence'] = max(chemical_confidences) if chemical_confidences else 0.0
        features['max_biological_confidence'] = max(biological_confidences) if biological_confidences else 0.0
        
        # Weighted scores (main classification features)
        features['chemical_weighted_score'] = chemical_weighted_score
        features['biological_weighted_score'] = biological_weighted_score
        
        # Area-based features
        features['chemical_area_ratio'] = chemical_area / total_area if total_area > 0 else 0.0
        features['biological_area_ratio'] = biological_area / total_area if total_area > 0 else 0.0
        
        # Ratio features
        total_typed_objects = features['chemical_objects'] + features['biological_objects']
        features['chemical_object_ratio'] = features['chemical_objects'] / total_typed_objects if total_typed_objects > 0 else 0.0
        features['biological_object_ratio'] = features['biological_objects'] / total_typed_objects if total_typed_objects > 0 else 0.0
        
        return features
    
    def _get_empty_features(self) -> Dict:
        """Return feature dict for images with no detections"""
        return {
            'total_objects': 0,
            'chemical_objects': 0,
            'biological_objects': 0,
            'neutral_objects': 0,
            'avg_chemical_confidence': 0.0,
            'avg_biological_confidence': 0.0,
            'max_chemical_confidence': 0.0,
            'max_biological_confidence': 0.0,
            'chemical_weighted_score': 0.0,
            'biological_weighted_score': 0.0,
            'chemical_area_ratio': 0.0,
            'biological_area_ratio': 0.0,
            'chemical_object_ratio': 0.0,
            'biological_object_ratio': 0.0
        }
    
    def classify_image_simple(self, detections: List[Dict], threshold: float = 0.5) -> Tuple[str, float]:
        """
        Simple rule-based classification using weighted scores
        
        Args:
            detections: List of object detections
            threshold: Decision threshold (0.5 = equal weight)
        
        Returns:
            Tuple of (classification, confidence)
        """
        features = self.extract_object_features(detections)
        
        chemical_score = features['chemical_weighted_score']
        biological_score = features['biological_weighted_score']
        
        # Avoid division by zero
        total_score = chemical_score + biological_score
        if total_score == 0:
            return "Unknown", 0.0
        
        # Calculate biological probability
        biological_prob = biological_score / total_score
        
        # Classify based on threshold
        if biological_prob > threshold:
            classification = "Biological"
            confidence = biological_prob
        else:
            classification = "Chemical" 
            confidence = 1 - biological_prob
        
        return classification, confidence
    
    def classify_image_advanced(self, detections: List[Dict]) -> Tuple[str, float, Dict]:
        """
        Advanced classification with detailed analysis
        
        Returns:
            Tuple of (classification, confidence, detailed_analysis)
        """
        features = self.extract_object_features(detections)
        classification, confidence = self.classify_image_simple(detections)
        
        # Detailed analysis
        analysis = {
            'total_objects_detected': features['total_objects'],
            'chemical_indicators': {
                'count': features['chemical_objects'],
                'avg_confidence': features['avg_chemical_confidence'],
                'weighted_score': features['chemical_weighted_score']
            },
            'biological_indicators': {
                'count': features['biological_objects'], 
                'avg_confidence': features['avg_biological_confidence'],
                'weighted_score': features['biological_weighted_score']
            },
            'dominant_objects': self._get_dominant_objects(detections),
            'confidence_explanation': self._explain_confidence(features, classification, confidence)
        }
        
        return classification, confidence, analysis
    
    def _get_dominant_objects(self, detections: List[Dict], top_n: int = 3) -> List[Dict]:
        """Get the most confident/important object detections"""
        if not detections:
            return []
        
        # Sort by confidence and weight
        sorted_detections = []
        for detection in detections:
            obj_name = detection['category_name']
            confidence = detection['confidence']
            weight = self.object_weights.get(obj_name, 0.0)
            importance = confidence * (1 + weight)  # Boost important objects
            
            sorted_detections.append({
                'object': obj_name,
                'confidence': confidence,
                'weight': weight,
                'importance': importance
            })
        
        # Return top N by importance
        sorted_detections.sort(key=lambda x: x['importance'], reverse=True)
        return sorted_detections[:top_n]
    
    def _explain_confidence(self, features: Dict, classification: str, confidence: float) -> str:
        """Generate human-readable explanation of the classification"""
        explanations = []
        
        if features['total_objects'] == 0:
            return "No objects detected in image"
        
        if classification == "Chemical":
            if features['chemical_objects'] > 0:
                explanations.append(f"Detected {features['chemical_objects']} chemical-related objects")
            if features['chemical_weighted_score'] > features['biological_weighted_score']:
                explanations.append("Chemical indicators have higher weighted confidence")
        else:  # Biological
            if features['biological_objects'] > 0:
                explanations.append(f"Detected {features['biological_objects']} biological-related objects")
            if features['biological_weighted_score'] > features['chemical_weighted_score']:
                explanations.append("Biological indicators have higher weighted confidence")
        
        if features['neutral_objects'] > 0:
            explanations.append(f"{features['neutral_objects']} neutral objects also detected")
        
        return ". ".join(explanations)
    
    def batch_analyze(self, images_detections: Dict[str, List[Dict]]) -> pd.DataFrame:
        """
        Analyze multiple images and return results as DataFrame
        
        Args:
            images_detections: Dict mapping image_ids to detection lists
        
        Returns:
            DataFrame with analysis results
        """
        results = []
        
        for image_id, detections in images_detections.items():
            classification, confidence, analysis = self.classify_image_advanced(detections)
            
            result = {
                'image_id': image_id,
                'classification': classification,
                'confidence': confidence,
                'total_objects': analysis['total_objects_detected'],
                'chemical_objects': analysis['chemical_indicators']['count'],
                'biological_objects': analysis['biological_indicators']['count'],
                'chemical_score': analysis['chemical_indicators']['weighted_score'],
                'biological_score': analysis['biological_indicators']['weighted_score']
            }
            
            # Add dominant objects info
            dominant = analysis['dominant_objects']
            for i, obj_info in enumerate(dominant[:3]):  # Top 3
                result[f'top_object_{i+1}'] = obj_info['object']
                result[f'top_confidence_{i+1}'] = obj_info['confidence']
            
            results.append(result)
        
        return pd.DataFrame(results)

if __name__ == "__main__":
    # Test the analyzer with updated mappings
    analyzer = ObjectAnalyzer()
    
    # Example detections for a chemical lab image
    chemical_detections = [
        {'category_name': 'gas mask', 'confidence': 0.9, 'bbox': [100, 100, 50, 60], 'area': 3000},
        {'category_name': 'hazmat suit', 'confidence': 0.85, 'bbox': [200, 150, 80, 120], 'area': 9600},
        {'category_name': 'chemical substance', 'confidence': 0.8, 'bbox': [50, 200, 30, 40], 'area': 1200},
        {'category_name': 'lab coat', 'confidence': 0.7, 'bbox': [300, 50, 100, 80], 'area': 8000}  # Now neutral
    ]
    
    # Example detections for a biological lab image  
    biological_detections = [
        {'category_name': 'petri dish', 'confidence': 0.95, 'bbox': [120, 140, 40, 40], 'area': 1600},
        {'category_name': 'pipette', 'confidence': 0.88, 'bbox': [180, 100, 20, 80], 'area': 1600},
        {'category_name': 'biohazard symbol', 'confidence': 0.9, 'bbox': [250, 200, 30, 30], 'area': 900},
        {'category_name': 'glassware', 'confidence': 0.75, 'bbox': [300, 180, 60, 100], 'area': 6000}  # Now neutral
    ]
    
    print("=== UPDATED MAPPINGS TEST ===")
    print("Chemical indicators: gas mask, hazmat suit, chemical substance, fume_hood, ventilation")
    print("Biological indicators: petri dish, pipette, biohazard symbol") 
    print("Neutral objects: lab coat, glassware, eye protection, computer, tech equipment, gloves")
    print()
    
    print("=== Chemical Lab Analysis ===")
    chem_class, chem_conf, chem_analysis = analyzer.classify_image_advanced(chemical_detections)
    print(f"Classification: {chem_class}")
    print(f"Confidence: {chem_conf:.3f}")
    print(f"Explanation: {chem_analysis['confidence_explanation']}")
    print(f"Dominant objects: {[obj['object'] for obj in chem_analysis['dominant_objects']]}")
    
    print("\n=== Biological Lab Analysis ===")
    bio_class, bio_conf, bio_analysis = analyzer.classify_image_advanced(biological_detections)
    print(f"Classification: {bio_class}")
    print(f"Confidence: {bio_conf:.3f}")
    print(f"Explanation: {bio_analysis['confidence_explanation']}")
    print(f"Dominant objects: {[obj['object'] for obj in bio_analysis['dominant_objects']]}")
    
    # Test with mixed objects
    mixed_detections = [
        {'category_name': 'petri dish', 'confidence': 0.9, 'bbox': [100, 100, 40, 40], 'area': 1600},
        {'category_name': 'gas mask', 'confidence': 0.85, 'bbox': [200, 150, 50, 60], 'area': 3000},
        {'category_name': 'lab coat', 'confidence': 0.8, 'bbox': [300, 200, 100, 120], 'area': 12000},  # Neutral
        {'category_name': 'computer', 'confidence': 0.7, 'bbox': [400, 100, 80, 60], 'area': 4800}   # Neutral
    ]
    
    print("\n=== Mixed Environment Analysis ===")
    mixed_class, mixed_conf, mixed_analysis = analyzer.classify_image_advanced(mixed_detections)
    print(f"Classification: {mixed_class}")
    print(f"Confidence: {mixed_conf:.3f}")
    print(f"Explanation: {mixed_analysis['confidence_explanation']}")
    print(f"Dominant objects: {[obj['object'] for obj in mixed_analysis['dominant_objects']]}")
    
    print("\n=== Batch Analysis Example ===")
    batch_detections = {
        'chem_lab_001': chemical_detections,
        'bio_lab_001': biological_detections,
        'mixed_lab_001': mixed_detections
    }
    
    results_df = analyzer.batch_analyze(batch_detections)
    print(results_df[['image_id', 'classification', 'confidence', 'total_objects', 'chemical_objects', 'biological_objects']])