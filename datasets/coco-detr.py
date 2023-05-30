"""
Code Copyright (c) guo mingyu
"""
import json
from pathlib import Path
from tqdm import tqdm

# COCO annotation file path
train_ann_file = Path('./path/to/coco/annotations/instances_train2017.json')
val_ann_file = Path('./path/to/coco/annotations/instances_val2017.json')

# Load COCO annotations
with open(train_ann_file, 'r') as f:
    train_ann = json.load(f)

with open(val_ann_file, 'r') as f:
    val_ann = json.load(f)

# Select 10 classes to train
selected_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Filter images and annotations based on selected classes
detr_train_ann = {'images': [], 'annotations': [], 'categories': []}
detr_val_ann = {'images': [], 'annotations': [], 'categories': []}

selected_train_images = []
selected_train_annotations = []
for ann in tqdm(train_ann['annotations'], desc="Processing train annotations"):
    if ann['category_id'] in selected_classes:
        selected_train_annotations.append(ann)

selected_val_images = []
selected_val_annotations = []
for ann in tqdm(val_ann['annotations'], desc="Processing validation annotations"):
    if ann['category_id'] in selected_classes:
        selected_val_annotations.append(ann)

for img in tqdm(train_ann['images'], desc="Processing train images"):
    if img['id'] in [ann['image_id'] for ann in selected_train_annotations]:
        selected_train_images.append(img)

for img in tqdm(val_ann['images'], desc="Processing validation images"):
    if img['id'] in [ann['image_id'] for ann in selected_val_annotations]:
        selected_val_images.append(img)

# Update categories and annotations with new IDs
for i, cat_id in enumerate(selected_classes):
    cat = train_ann['categories'][cat_id - 1]
    cat['id'] = i + 1
    detr_train_ann['categories'].append(cat)
    detr_val_ann['categories'].append(cat)

for ann in tqdm(selected_train_annotations, desc="Updating train annotations"):
    ann['category_id'] = selected_classes.index(ann['category_id']) + 1
    detr_train_ann['annotations'].append(ann)

for ann in tqdm(selected_val_annotations, desc="Updating validation annotations"):
    ann['category_id'] = selected_classes.index(ann['category_id']) + 1
    detr_val_ann['annotations'].append(ann)

# Update images with new IDs
for img in selected_train_images:
    detr_train_ann['images'].append(img)

for img in selected_val_images:
    detr_val_ann['images'].append(img)

# Save DETR annotations as JSON files
detr_train_path = 'detr_train.json'
detr_val_path = 'detr_val.json'

with open(detr_train_path, 'w') as f:
    json.dump(detr_train_ann, f)

with open(detr_val_path, 'w') as f:
    json.dump(detr_val_ann, f)
