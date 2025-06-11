"""
Data preprocessing script for Image Classifier SSE - Object Detection Based
Processes COCO format annotations to extract object detection features
"""

import json
import os
import shutil
from collections import defaultdict, Counter
import yaml
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np

from object_analyzer import ObjectAnalyzer

class COCOObjectProcessor:
    def __init__(self, config_path="config.yaml"):
        """Initialize the data processor with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize object analyzer
        self.analyzer = ObjectAnalyzer(config_path)
        
    def load_coco_data(self, annotation_file):
        """Load and parse COCO annotation file with object detection focus"""
        print(f"Loading COCO data from {annotation_file}")
        
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create category ID to name mapping
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Create image ID to filename mapping
        images = {img['id']: img['file_name'] for img in coco_data['images']}
        
        # Process annotations into detections per image
        image_detections = defaultdict(list)
        
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            category_name = categories[ann['category_id']]
            
            # Calculate bounding box area
            bbox = ann['bbox']  # [x, y, width, height]
            area = bbox[2] * bbox[3] if len(bbox) >= 4 else ann.get('area', 0)
            
            detection = {
                'category_name': category_name,
                'confidence': 1.0,  # COCO ground truth has perfect confidence
                'bbox': bbox,
                'area': area,
                'annotation_id': ann['id']
            }
            
            image_detections[image_id].append(detection)
        
        return images, image_detections, categories
    
    def analyze_detection_dataset(self, images, image_detections):
        """Analyze the dataset from object detection perspective"""
        print("Analyzing object detection dataset...")
        
        # Basic statistics
        total_images = len(images)
        total_detections = sum(len(detections) for detections in image_detections.values())
        
        print(f"\nDataset Statistics:")
        print(f"Total images: {total_images}")
        print(f"Total object detections: {total_detections}")
        print(f"Average detections per image: {total_detections/total_images:.2f}")
        
        # Object category analysis
        all_objects = []
        for detections in image_detections.values():
            for detection in detections:
                all_objects.append(detection['category_name'])
        
        object_counts = Counter(all_objects)
        print(f"\nObject category distribution:")
        for category, count in object_counts.most_common():
            percentage = (count / total_detections) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
        
        # Classify images using object analyzer
        print(f"\nClassifying images based on detected objects...")
        
        image_classifications = {}
        classification_counts = {'Chemical': 0, 'Biological': 0, 'Unknown': 0}
        confidence_scores = []
        
        for image_id, detections in image_detections.items():
            classification, confidence = self.analyzer.classify_image_simple(detections)
            image_classifications[image_id] = {
                'classification': classification,
                'confidence': confidence,
                'num_objects': len(detections)
            }
            classification_counts[classification] += 1
            confidence_scores.append(confidence)
        
        print(f"\nImage classification results:")
        for label, count in classification_counts.items():
            percentage = (count / total_images) * 100
            print(f"  {label}: {count} images ({percentage:.1f}%)")
        
        print(f"Average confidence: {np.mean(confidence_scores):.3f}")
        print(f"Confidence std: {np.std(confidence_scores):.3f}")
        
        return image_classifications, object_counts, classification_counts
    
    def create_object_features_dataset(self, images, image_detections, image_classifications):
        """Create dataset of object-based features for training"""
        print("\nCreating object features dataset...")
        
        features_data = []
        
        for image_id, detections in image_detections.items():
            # Extract features using analyzer
            features = self.analyzer.extract_object_features(detections)
            
            # Add metadata
            features['image_id'] = image_id
            features['image_filename'] = images[image_id]
            
            # Add ground truth label
            classification_info = image_classifications[image_id]
            features['label'] = classification_info['classification']
            features['confidence'] = classification_info['confidence']
            
            features_data.append(features)
        
        # Convert to DataFrame
        df = pd.DataFrame(features_data)
        
        # Remove unknown classifications for binary training
        df_binary = df[df['label'].isin(['Chemical', 'Biological'])].copy()
        
        # Convert labels to binary (0=Chemical, 1=Biological)
        df_binary['label_binary'] = (df_binary['label'] == 'Biological').astype(int)
        
        print(f"Created features dataset with {len(df_binary)} images")
        print(f"Feature columns: {list(df_binary.columns)}")
        
        return df_binary
    
    def create_train_val_test_split(self, df):
        """Split the features dataset into train/val/test sets"""
        print("\nCreating train/validation/test splits...")
        
        # Ensure we have both classes
        chemical_df = df[df['label'] == 'Chemical']
        biological_df = df[df['label'] == 'Biological']
        
        print(f"Chemical samples: {len(chemical_df)}")
        print(f"Biological samples: {len(biological_df)}")
        
        if len(chemical_df) == 0 or len(biological_df) == 0:
            print("Warning: Missing one class in dataset!")
            return None
        
        # Split each class separately to maintain balance
        def split_class_data(class_df):
            train_split = self.config['data']['train_split']
            val_split = self.config['data']['val_split']
            test_split = self.config['data']['test_split']
            
            # First split: train+val vs test
            train_val, test = train_test_split(
                class_df, 
                test_size=test_split, 
                random_state=42
            )
            
            # Second split: train vs val
            train, val = train_test_split(
                train_val,
                test_size=val_split/(train_split + val_split),
                random_state=42
            )
            
            return train, val, test
        
        # Split both classes
        chem_train, chem_val, chem_test = split_class_data(chemical_df)
        bio_train, bio_val, bio_test = split_class_data(biological_df)
        
        # Combine splits
        train_df = pd.concat([chem_train, bio_train], ignore_index=True).sample(frac=1, random_state=42)
        val_df = pd.concat([chem_val, bio_val], ignore_index=True).sample(frac=1, random_state=42)
        test_df = pd.concat([chem_test, bio_test], ignore_index=True).sample(frac=1, random_state=42)
        
        print(f"Train set: {len(train_df)} images ({len(chem_train)} chem, {len(bio_train)} bio)")
        print(f"Validation set: {len(val_df)} images ({len(chem_val)} chem, {len(bio_val)} bio)")
        print(f"Test set: {len(test_df)} images ({len(chem_test)} chem, {len(bio_test)} bio)")
        
        # Save datasets
        output_dir = self.config['data']['processed_data_path']
        os.makedirs(output_dir, exist_ok=True)
        
        train_df.to_csv(os.path.join(output_dir, 'train_features.csv'), index=False)
        val_df.to_csv(os.path.join(output_dir, 'val_features.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test_features.csv'), index=False)
        
        print(f"Saved feature datasets to {output_dir}/")
        
        return train_df, val_df, test_df
    
    def visualize_object_analysis(self, object_counts, classification_counts, df):
        """Create visualizations of the object detection analysis"""
        print("\nCreating visualizations...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Object category frequency
        top_objects = dict(object_counts.most_common(12))
        objects = list(top_objects.keys())
        counts = list(top_objects.values())
        
        # Color code by object type
        colors = []
        for obj in objects:
            if obj in self.analyzer.strong_chemical_objects or obj in self.analyzer.moderate_chemical_objects:
                colors.append('red')
            elif obj in self.analyzer.strong_biological_objects or obj in self.analyzer.moderate_biological_objects:
                colors.append('blue')
            else:
                colors.append('gray')
        
        ax1.barh(objects, counts, color=colors, alpha=0.7)
        ax1.set_title('Object Detection Frequency')
        ax1.set_xlabel('Number of Detections')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Chemical'),
            Patch(facecolor='blue', alpha=0.7, label='Biological'),
            Patch(facecolor='gray', alpha=0.7, label='Neutral')
        ]
        ax1.legend(handles=legend_elements, loc='lower right')
        
        # 2. Image classification distribution
        labels = list(classification_counts.keys())
        sizes = list(classification_counts.values())
        colors_pie = ['red', 'blue', 'gray']
        
        ax2.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_pie[:len(labels)])
        ax2.set_title('Image Classification Distribution')
        
        # 3. Feature correlation heatmap
        feature_cols = [
            'chemical_weighted_score', 'biological_weighted_score',
            'chemical_objects', 'biological_objects',
            'avg_chemical_confidence', 'avg_biological_confidence'
        ]
        
        if all(col in df.columns for col in feature_cols):
            corr_matrix = df[feature_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax3)
            ax3.set_title('Feature Correlation Matrix')
        else:
            ax3.text(0.5, 0.5, 'Feature correlation\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Feature Analysis')
        
        # 4. Classification confidence distribution
        chemical_conf = df[df['label'] == 'Chemical']['confidence']
        biological_conf = df[df['label'] == 'Biological']['confidence']
        
        ax4.hist(chemical_conf, alpha=0.7, label='Chemical', color='red', bins=20)
        ax4.hist(biological_conf, alpha=0.7, label='Biological', color='blue', bins=20)
        ax4.set_title('Classification Confidence Distribution')
        ax4.set_xlabel('Confidence Score')
        ax4.set_ylabel('Number of Images')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('object_detection_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Additional analysis plot
        self._plot_detection_patterns(df)
    
    def _plot_detection_patterns(self, df):
        """Plot detection patterns analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Objects per image by class
        chemical_objects = df[df['label'] == 'Chemical']['total_objects']
        biological_objects = df[df['label'] == 'Biological']['total_objects']
        
        ax1.boxplot([chemical_objects, biological_objects], 
                   labels=['Chemical', 'Biological'])
        ax1.set_title('Objects per Image by Class')
        ax1.set_ylabel('Number of Objects')
        
        # 2. Weighted scores comparison
        ax2.scatter(df['chemical_weighted_score'], df['biological_weighted_score'], 
                   c=df['label_binary'], cmap='coolwarm', alpha=0.6)
        ax2.set_xlabel('Chemical Weighted Score')
        ax2.set_ylabel('Biological Weighted Score')
        ax2.set_title('Weighted Scores Comparison')
        ax2.plot([0, ax2.get_xlim()[1]], [0, ax2.get_ylim()[1]], 'k--', alpha=0.5)
        
        # 3. Object ratio analysis
        chemical_ratio = df['chemical_object_ratio']
        biological_ratio = df['biological_object_ratio']
        
        ax3.scatter(chemical_ratio, biological_ratio, 
                   c=df['label_binary'], cmap='coolwarm', alpha=0.6)
        ax3.set_xlabel('Chemical Object Ratio')
        ax3.set_ylabel('Biological Object Ratio')
        ax3.set_title('Object Type Ratios')
        
        # 4. Classification accuracy by number of objects
        bins = [0, 1, 3, 5, 10, float('inf')]
        bin_labels = ['0', '1-2', '3-4', '5-9', '10+']
        df['object_bins'] = pd.cut(df['total_objects'], bins=bins, labels=bin_labels, right=False)
        
        accuracy_by_objects = []
        for bin_label in bin_labels:
            bin_data = df[df['object_bins'] == bin_label]
            if len(bin_data) > 0:
                # Simple accuracy based on whether classification matches expected pattern
                correct = sum((bin_data['chemical_weighted_score'] < bin_data['biological_weighted_score']) == 
                            (bin_data['label'] == 'Biological'))
                accuracy = correct / len(bin_data)
                accuracy_by_objects.append(accuracy)
            else:
                accuracy_by_objects.append(0)
        
        ax4.bar(bin_labels, accuracy_by_objects, alpha=0.7, color='green')
        ax4.set_title('Classification Accuracy by Object Count')
        ax4.set_xlabel('Number of Objects in Image')
        ax4.set_ylabel('Accuracy')
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('detection_patterns_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def process_all_data(self, annotation_files, image_directories):
        """Process all COCO annotation files for object detection analysis"""
        all_images = {}
        all_detections = {}
        
        # Load all annotation files
        for i, (ann_file, img_dir) in enumerate(zip(annotation_files, image_directories)):
            print(f"\nProcessing file {i+1}/{len(annotation_files)}: {ann_file}")
            
            images, detections, categories = self.load_coco_data(ann_file)
            
            # Update mappings (ensuring unique image IDs across files)
            for img_id, filename in images.items():
                unique_id = f"{i}_{img_id}"
                all_images[unique_id] = filename
                if img_id in detections:
                    all_detections[unique_id] = detections[img_id]
        
        # Analyze from object detection perspective
        image_classifications, object_counts, classification_counts = self.analyze_detection_dataset(
            all_images, all_detections
        )
        
        # Create features dataset
        df = self.create_object_features_dataset(all_images, all_detections, image_classifications)
        
        # Create visualizations
        # NEW (comment it out):
        # self.visualize_object_analysis(object_counts, classification_counts, df)
        print("Skipping visualization due to matplotlib version issue")

        
        # Create train/val/test splits
        if df is not None and len(df) > 0:
            train_df, val_df, test_df = self.create_train_val_test_split(df)
            
            # Show sample analysis
            self._show_sample_analysis(all_detections, image_classifications)
            
            return train_df, val_df, test_df
        else:
            print("Error: Could not create feature dataset")
            return None, None, None
    
    def _show_sample_analysis(self, all_detections, image_classifications, num_samples=3):
        """Show detailed analysis for sample images"""
        print(f"\n=== Sample Image Analysis ===")
        
        # Get samples from each class
        chemical_samples = [img_id for img_id, info in image_classifications.items() 
                          if info['classification'] == 'Chemical'][:num_samples]
        biological_samples = [img_id for img_id, info in image_classifications.items() 
                            if info['classification'] == 'Biological'][:num_samples]
        
        for sample_type, samples in [('Chemical', chemical_samples), ('Biological', biological_samples)]:
            print(f"\n--- {sample_type} Samples ---")
            
            for img_id in samples:
                detections = all_detections[img_id]
                classification, confidence, analysis = self.analyzer.classify_image_advanced(detections)
                
                print(f"\nImage: {img_id}")
                print(f"Classification: {classification} (confidence: {confidence:.3f})")
                print(f"Objects detected: {len(detections)}")
                print(f"Dominant objects: {[obj['object'] for obj in analysis['dominant_objects'][:3]]}")
                print(f"Explanation: {analysis['confidence_explanation']}")

def main():
    """Main function to run the object detection preprocessing"""
    processor = COCOObjectProcessor()
    
    # List your annotation files and corresponding image directories
    annotation_files = [
        "data/raw/annotations/_annotations.coco.json",
        # Add more annotation files if you have them
    ]
    
    image_directories = [
        "data/raw/images/",
        # Add corresponding image directories
    ]
    
    # Process all data
    train_df, val_df, test_df = processor.process_all_data(annotation_files, image_directories)
    
    if train_df is not None:
        print(f"\nData preprocessing complete!")
        print(f"Next steps:")
        print(f"1. Review the object_detection_analysis.png and detection_patterns_analysis.png")
        print(f"2. Check the processed feature datasets in data/processed/")
        print(f"3. Run the training script to build your object-based classifier")
        print(f"4. Training will use {len(train_df)} samples for training")
    else:
        print("Error in data preprocessing. Please check your data and try again.")

if __name__ == "__main__":
    main()

"""
Data preprocessing script for Image Classifier SSE
Converts COCO format annotations to binary classification dataset
"""

import json
import os
import shutil
from collections import defaultdict, Counter
import yaml
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

class COCODataProcessor:
    def __init__(self, config_path="config.yaml"):
        """Initialize the data processor with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.biological_categories = set(self.config['categories']['biological'])
        self.chemical_categories = set(self.config['categories']['chemical'])
        self.neutral_categories = set(self.config['categories']['neutral'])
        
        # Create category ID mapping
        self.category_mapping = {}
        
    def load_coco_data(self, annotation_file):
        """Load and parse COCO annotation file"""
        print(f"Loading COCO data from {annotation_file}")
        
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create category ID to name mapping
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Create image ID to filename mapping
        images = {img['id']: img['file_name'] for img in coco_data['images']}
        
        # Process annotations
        image_annotations = defaultdict(list)
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            category_name = categories[ann['category_id']]
            image_annotations[image_id].append(category_name)
        
        return images, image_annotations, categories
    
    def classify_image(self, categories_in_image):
        """
        Classify an image as biological, chemical, or mixed based on detected objects
        
        Rules:
        1. If only biological objects -> biological
        2. If only chemical objects -> chemical  
        3. If mixed -> use majority vote
        4. If only neutral -> exclude or use context
        """
        bio_count = sum(1 for cat in categories_in_image if cat in self.biological_categories)
        chem_count = sum(1 for cat in categories_in_image if cat in self.chemical_categories)
        neutral_count = sum(1 for cat in categories_in_image if cat in self.neutral_categories)
        
        # If we have both bio and chem, use majority
        if bio_count > 0 and chem_count > 0:
            return 'biological' if bio_count >= chem_count else 'chemical'
        elif bio_count > 0:
            return 'biological'
        elif chem_count > 0:
            return 'chemical'
        else:
            # Only neutral categories - we'll exclude these for now
            return 'neutral'
    
    def analyze_dataset(self, images, image_annotations):
        """Analyze the dataset to understand class distribution"""
        print("Analyzing dataset...")
        
        # Count categories
        all_categories = []
        for categories in image_annotations.values():
            all_categories.extend(categories)
        
        category_counts = Counter(all_categories)
        
        # Classify images
        image_labels = {}
        label_counts = {'biological': 0, 'chemical': 0, 'neutral': 0}
        
        for image_id, categories in image_annotations.items():
            label = self.classify_image(categories)
            image_labels[image_id] = label
            label_counts[label] += 1
        
        # Print analysis
        print(f"\nDataset Analysis:")
        print(f"Total images: {len(images)}")
        print(f"Total annotations: {len(all_categories)}")
        print(f"\nClass distribution:")
        for label, count in label_counts.items():
            percentage = (count / len(images)) * 100
            print(f"  {label}: {count} images ({percentage:.1f}%)")
        
        print(f"\nCategory frequency:")
        for category, count in category_counts.most_common():
            print(f"  {category}: {count}")
        
        return image_labels, category_counts, label_counts
    
    def create_classification_dataset(self, images, image_labels, source_image_dir):
        """Create organized dataset for binary classification"""
        print("\nCreating classification dataset...")
        
        # Create output directories
        output_dir = self.config['data']['processed_data_path']
        os.makedirs(f"{output_dir}/biological", exist_ok=True)
        os.makedirs(f"{output_dir}/chemical", exist_ok=True)
        
        # Copy images to appropriate folders
        copied_count = {'biological': 0, 'chemical': 0}
        
        for image_id, label in image_labels.items():
            if label in ['biological', 'chemical']:  # Skip neutral for now
                source_path = os.path.join(source_image_dir, images[image_id])
                dest_path = os.path.join(output_dir, label, images[image_id])
                
                if os.path.exists(source_path):
                    shutil.copy2(source_path, dest_path)
                    copied_count[label] += 1
                else:
                    print(f"Warning: Image not found: {source_path}")
        
        print(f"Dataset created successfully!")
        print(f"Biological images: {copied_count['biological']}")
        print(f"Chemical images: {copied_count['chemical']}")
        
        return copied_count
    
    def create_train_val_test_split(self):
        """Split the processed dataset into train/val/test sets"""
        print("\nCreating train/validation/test splits...")
        
        processed_dir = self.config['data']['processed_data_path']
        
        # Get all images for each class
        bio_images = [f for f in os.listdir(f"{processed_dir}/biological") 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        chem_images = [f for f in os.listdir(f"{processed_dir}/chemical") 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Create splits for each class
        train_split = self.config['data']['train_split']
        val_split = self.config['data']['val_split']
        
        def split_class_data(images, class_name):
            # First split: train + val, test
            train_val, test = train_test_split(
                images, 
                test_size=self.config['data']['test_split'], 
                random_state=42
            )
            
            # Second split: train, val
            train, val = train_test_split(
                train_val, 
                test_size=val_split/(train_split + val_split), 
                random_state=42
            )
            
            return train, val, test
        
        # Split both classes
        bio_train, bio_val, bio_test = split_class_data(bio_images, 'biological')
        chem_train, chem_val, chem_test = split_class_data(chem_images, 'chemical')
        
        # Create split directories
        for split in ['train', 'val', 'test']:
            for class_name in ['biological', 'chemical']:
                os.makedirs(f"{processed_dir}/{split}/{class_name}", exist_ok=True)
        
        # Move files to appropriate splits
        def move_files(file_list, source_class, target_split):
            for filename in file_list:
                source = f"{processed_dir}/{source_class}/{filename}"
                dest = f"{processed_dir}/{target_split}/{source_class}/{filename}"
                shutil.move(source, dest)
        
        # Move biological images
        move_files(bio_train, 'biological', 'train')
        move_files(bio_val, 'biological', 'val')
        move_files(bio_test, 'biological', 'test')
        
        # Move chemical images  
        move_files(chem_train, 'chemical', 'train')
        move_files(chem_val, 'chemical', 'val')
        move_files(chem_test, 'chemical', 'test')
        
        # Remove empty class directories
        os.rmdir(f"{processed_dir}/biological")
        os.rmdir(f"{processed_dir}/chemical")
        
        print(f"Train set: {len(bio_train)} biological, {len(chem_train)} chemical")
        print(f"Validation set: {len(bio_val)} biological, {len(chem_val)} chemical")
        print(f"Test set: {len(bio_test)} biological, {len(chem_test)} chemical")
    
    def visualize_data_distribution(self, category_counts, label_counts):
        """Create visualizations of the data distribution"""
        print("\nCreating data visualizations...")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Category frequency
        categories = list(category_counts.keys())[:15]  # Top 15 categories
        counts = [category_counts[cat] for cat in categories]
        
        ax1.barh(categories, counts)
        ax1.set_title('Top 15 Category Frequencies')
        ax1.set_xlabel('Number of Annotations')
        
        # 2. Class distribution pie chart
        labels = [k for k, v in label_counts.items() if v > 0]
        sizes = [v for v in label_counts.values() if v > 0]
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        
        ax2.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors[:len(labels)])
        ax2.set_title('Class Distribution')
        
        # 3. Category type distribution
        bio_categories = [cat for cat in categories if cat in self.biological_categories]
        chem_categories = [cat for cat in categories if cat in self.chemical_categories]
        neutral_categories = [cat for cat in categories if cat in self.neutral_categories]
        
        type_counts = {
            'Biological': sum(category_counts[cat] for cat in bio_categories),
            'Chemical': sum(category_counts[cat] for cat in chem_categories),
            'Neutral': sum(category_counts[cat] for cat in neutral_categories)
        }
        
        ax3.bar(type_counts.keys(), type_counts.values(), 
                color=['lightblue', 'lightcoral', 'lightgreen'])
        ax3.set_title('Annotations by Category Type')
        ax3.set_ylabel('Number of Annotations')
        
        # 4. Images per class
        ax4.bar(['Biological', 'Chemical'], 
                [label_counts['biological'], label_counts['chemical']],
                color=['lightblue', 'lightcoral'])
        ax4.set_title('Images per Class (Excluding Neutral)')
        ax4.set_ylabel('Number of Images')
        
        plt.tight_layout()
        plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def process_all_data(self, annotation_files, image_directories):
        """Process all COCO annotation files"""
        all_images = {}
        all_annotations = {}
        
        # Load all annotation files
        for i, (ann_file, img_dir) in enumerate(zip(annotation_files, image_directories)):
            print(f"\nProcessing file {i+1}/{len(annotation_files)}: {ann_file}")
            
            images, annotations, categories = self.load_coco_data(ann_file)
            
            # Update mappings (ensuring unique image IDs across files)
            for img_id, filename in images.items():
                unique_id = f"{i}_{img_id}"
                all_images[unique_id] = filename
                if img_id in annotations:
                    all_annotations[unique_id] = annotations[img_id]
        
        # Analyze combined dataset
        image_labels, category_counts, label_counts = self.analyze_dataset(
            all_images, all_annotations
        )
        
        # Create visualizations
        self.visualize_data_distribution(category_counts, label_counts)
        
        # For now, use the first image directory for copying
        # In practice, you might need to modify this based on your file structure
        self.create_classification_dataset(all_images, image_labels, image_directories[0])
        
        # Create train/val/test splits
        self.create_train_val_test_split()
        
        return image_labels, category_counts, label_counts

def main():
    """Main function to run the data preprocessing"""
    processor = COCODataProcessor()
    
    # List your annotation files and corresponding image directories
    annotation_files = [
        "data/raw/annotations/_annotations.coco.json",
        # Add more annotation files if you have them
    ]
    
    image_directories = [
        "data/raw/images/",
        # Add corresponding image directories
    ]
    
    # Process all data
    processor.process_all_data(annotation_files, image_directories)
    
    print("\nData preprocessing complete!")
    print("Next steps:")
    print("1. Review the data_analysis.png visualization")
    print("2. Check the processed data in data/processed/")
    print("3. Run the training script to build your model")

if __name__ == "__main__":
    main()

import json
import os
import shutil
from collections import defaultdict, Counter
import yaml
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

class COCODataProcessor:
    def __init__(self, config_path="config.yaml"):
        """Initialize the data processor with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.biological_categories = set(self.config['categories']['biological'])
        self.chemical_categories = set(self.config['categories']['chemical'])
        self.neutral_categories = set(self.config['categories']['neutral'])
        
        # Create category ID mapping
        self.category_mapping = {}
        
    def load_coco_data(self, annotation_file):
        """Load and parse COCO annotation file"""
        print(f"Loading COCO data from {annotation_file}")
        
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create category ID to name mapping
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Create image ID to filename mapping
        images = {img['id']: img['file_name'] for img in coco_data['images']}
        
        # Process annotations
        image_annotations = defaultdict(list)
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            category_name = categories[ann['category_id']]
            image_annotations[image_id].append(category_name)
        
        return images, image_annotations, categories
    
    def classify_image(self, categories_in_image):
        """
        Classify an image as biological, chemical, or mixed based on detected objects
        
        Rules:
        1. If only biological objects -> biological
        2. If only chemical objects -> chemical  
        3. If mixed -> use majority vote
        4. If only neutral -> exclude or use context
        """
        bio_count = sum(1 for cat in categories_in_image if cat in self.biological_categories)
        chem_count = sum(1 for cat in categories_in_image if cat in self.chemical_categories)
        neutral_count = sum(1 for cat in categories_in_image if cat in self.neutral_categories)
        
        # If we have both bio and chem, use majority
        if bio_count > 0 and chem_count > 0:
            return 'biological' if bio_count >= chem_count else 'chemical'
        elif bio_count > 0:
            return 'biological'
        elif chem_count > 0:
            return 'chemical'
        else:
            # Only neutral categories - we'll exclude these for now
            return 'neutral'
    
    def analyze_dataset(self, images, image_annotations):
        """Analyze the dataset to understand class distribution"""
        print("Analyzing dataset...")
        
        # Count categories
        all_categories = []
        for categories in image_annotations.values():
            all_categories.extend(categories)
        
        category_counts = Counter(all_categories)
        
        # Classify images
        image_labels = {}
        label_counts = {'biological': 0, 'chemical': 0, 'neutral': 0}
        
        for image_id, categories in image_annotations.items():
            label = self.classify_image(categories)
            image_labels[image_id] = label
            label_counts[label] += 1
        
        # Print analysis
        print(f"\nDataset Analysis:")
        print(f"Total images: {len(images)}")
        print(f"Total annotations: {len(all_categories)}")
        print(f"\nClass distribution:")
        for label, count in label_counts.items():
            percentage = (count / len(images)) * 100
            print(f"  {label}: {count} images ({percentage:.1f}%)")
        
        print(f"\nCategory frequency:")
        for category, count in category_counts.most_common():
            print(f"  {category}: {count}")
        
        return image_labels, category_counts, label_counts
    
    def create_classification_dataset(self, images, image_labels, source_image_dir):
        """Create organized dataset for binary classification"""
        print("\nCreating classification dataset...")
        
        # Create output directories
        output_dir = self.config['data']['processed_data_path']
        os.makedirs(f"{output_dir}/biological", exist_ok=True)
        os.makedirs(f"{output_dir}/chemical", exist_ok=True)
        
        # Copy images to appropriate folders
        copied_count = {'biological': 0, 'chemical': 0}
        
        for image_id, label in image_labels.items():
            if label in ['biological', 'chemical']:  # Skip neutral for now
                source_path = os.path.join(source_image_dir, images[image_id])
                dest_path = os.path.join(output_dir, label, images[image_id])
                
                if os.path.exists(source_path):
                    shutil.copy2(source_path, dest_path)
                    copied_count[label] += 1
                else:
                    print(f"Warning: Image not found: {source_path}")
        
        print(f"Dataset created successfully!")
        print(f"Biological images: {copied_count['biological']}")
        print(f"Chemical images: {copied_count['chemical']}")
        
        return copied_count
    
    def create_train_val_test_split(self):
        """Split the processed dataset into train/val/test sets"""
        print("\nCreating train/validation/test splits...")
        
        processed_dir = self.config['data']['processed_data_path']
        
        # Get all images for each class
        bio_images = [f for f in os.listdir(f"{processed_dir}/biological") 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        chem_images = [f for f in os.listdir(f"{processed_dir}/chemical") 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Create splits for each class
        train_split = self.config['data']['train_split']
        val_split = self.config['data']['val_split']
        
        def split_class_data(images, class_name):
            # First split: train + val, test
            train_val, test = train_test_split(
                images, 
                test_size=self.config['data']['test_split'], 
                random_state=42
            )
            
            # Second split: train, val
            train, val = train_test_split(
                train_val, 
                test_size=val_split/(train_split + val_split), 
                random_state=42
            )
            
            return train, val, test
        
        # Split both classes
        bio_train, bio_val, bio_test = split_class_data(bio_images, 'biological')
        chem_train, chem_val, chem_test = split_class_data(chem_images, 'chemical')
        
        # Create split directories
        for split in ['train', 'val', 'test']:
            for class_name in ['biological', 'chemical']:
                os.makedirs(f"{processed_dir}/{split}/{class_name}", exist_ok=True)
        
        # Move files to appropriate splits
        def move_files(file_list, source_class, target_split):
            for filename in file_list:
                source = f"{processed_dir}/{source_class}/{filename}"
                dest = f"{processed_dir}/{target_split}/{source_class}/{filename}"
                shutil.move(source, dest)
        
        # Move biological images
        move_files(bio_train, 'biological', 'train')
        move_files(bio_val, 'biological', 'val')
        move_files(bio_test, 'biological', 'test')
        
        # Move chemical images  
        move_files(chem_train, 'chemical', 'train')
        move_files(chem_val, 'chemical', 'val')
        move_files(chem_test, 'chemical', 'test')
        
        # Remove empty class directories
        os.rmdir(f"{processed_dir}/biological")
        os.rmdir(f"{processed_dir}/chemical")
        
        print(f"Train set: {len(bio_train)} biological, {len(chem_train)} chemical")
        print(f"Validation set: {len(bio_val)} biological, {len(chem_val)} chemical")
        print(f"Test set: {len(bio_test)} biological, {len(chem_test)} chemical")
    
    def visualize_data_distribution(self, category_counts, label_counts):
        """Create visualizations of the data distribution"""
        print("\nCreating data visualizations...")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Category frequency
        categories = list(category_counts.keys())[:15]  # Top 15 categories
        counts = [category_counts[cat] for cat in categories]
        
        ax1.barh(categories, counts)
        ax1.set_title('Top 15 Category Frequencies')
        ax1.set_xlabel('Number of Annotations')
        
        # 2. Class distribution pie chart
        labels = [k for k, v in label_counts.items() if v > 0]
        sizes = [v for v in label_counts.values() if v > 0]
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        
        ax2.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors[:len(labels)])
        ax2.set_title('Class Distribution')
        
        # 3. Category type distribution
        bio_categories = [cat for cat in categories if cat in self.biological_categories]
        chem_categories = [cat for cat in categories if cat in self.chemical_categories]
        neutral_categories = [cat for cat in categories if cat in self.neutral_categories]
        
        type_counts = {
            'Biological': sum(category_counts[cat] for cat in bio_categories),
            'Chemical': sum(category_counts[cat] for cat in chem_categories),
            'Neutral': sum(category_counts[cat] for cat in neutral_categories)
        }
        
        ax3.bar(type_counts.keys(), type_counts.values(), 
                color=['lightblue', 'lightcoral', 'lightgreen'])
        ax3.set_title('Annotations by Category Type')
        ax3.set_ylabel('Number of Annotations')
        
        # 4. Images per class
        ax4.bar(['Biological', 'Chemical'], 
                [label_counts['biological'], label_counts['chemical']],
                color=['lightblue', 'lightcoral'])
        ax4.set_title('Images per Class (Excluding Neutral)')
        ax4.set_ylabel('Number of Images')
        
        plt.tight_layout()
        plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def process_all_data(self, annotation_files, image_directories):
        """Process all COCO annotation files"""
        all_images = {}
        all_annotations = {}
        
        # Load all annotation files
        for i, (ann_file, img_dir) in enumerate(zip(annotation_files, image_directories)):
            print(f"\nProcessing file {i+1}/{len(annotation_files)}: {ann_file}")
            
            images, annotations, categories = self.load_coco_data(ann_file)
            
            # Update mappings (ensuring unique image IDs across files)
            for img_id, filename in images.items():
                unique_id = f"{i}_{img_id}"
                all_images[unique_id] = filename
                if img_id in annotations:
                    all_annotations[unique_id] = annotations[img_id]
        
        # Analyze combined dataset
        image_labels, category_counts, label_counts = self.analyze_dataset(
            all_images, all_annotations
        )
        
        # Create visualizations
        self.visualize_data_distribution(category_counts, label_counts)
        
        # For now, use the first image directory for copying
        # In practice, you might need to modify this based on your file structure
        self.create_classification_dataset(all_images, image_labels, image_directories[0])
        
        # Create train/val/test splits
        self.create_train_val_test_split()
        
        return image_labels, category_counts, label_counts

def main():
    """Main function to run the data preprocessing"""
    processor = COCODataProcessor()
    
    # List your annotation files and corresponding image directories
    annotation_files = [
        "data/raw/annotations/_annotations.coco.json",
        # Add more annotation files if you have them
    ]
    
    image_directories = [
        "data/raw/images/",
        # Add corresponding image directories
    ]
    
    # Process all data
    processor.process_all_data(annotation_files, image_directories)
    
    print("\nData preprocessing complete!")
    print("Next steps:")
    print("1. Review the data_analysis.png visualization")
    print("2. Check the processed data in data/processed/")
    print("3. Run the training script to build your model")

if __name__ == "__main__":
    main()