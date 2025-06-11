# Quick script to analyze why images are getting "Unknown" predictions
import pandas as pd
import json
from collections import Counter

# Load the results
df = pd.read_csv('batch_predictions__annotations.coco.csv')

# Load COCO data to see what objects are in Unknown images
with open('data/raw/annotations/_annotations.coco.json', 'r') as f:
    coco_data = json.load(f)

# Create category mapping
categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

# Group annotations by image
image_objects = {}
for ann in coco_data['annotations']:
    image_id = ann['image_id']
    if image_id not in image_objects:
        image_objects[image_id] = []
    image_objects[image_id].append(categories[ann['category_id']])

# Analyze Unknown predictions
unknown_df = df[df['prediction'] == 'Unknown']
print(f"Analyzing {len(unknown_df)} Unknown predictions...")

# Count what objects appear in Unknown images
unknown_objects = []
for idx, row in unknown_df.iterrows():
    image_id = int(row['image_id'].split('_')[1])  # Extract original image ID
    if image_id in image_objects:
        unknown_objects.extend(image_objects[image_id])

object_counts = Counter(unknown_objects)
print(f"\nObjects in Unknown images:")
for obj, count in object_counts.most_common():
    print(f"  {obj}: {count}")

# Show some specific examples
print(f"\nSample Unknown images and their objects:")
for i, (idx, row) in enumerate(unknown_df.head(5).iterrows()):
    image_id = int(row['image_id'].split('_')[1])
    objects = image_objects.get(image_id, [])
    print(f"  Image {image_id}: {objects}")