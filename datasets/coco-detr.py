"""
Code Copyright (c) guo mingyu
"""
import json
from pathlib import Path

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

# Convert COCO annotations to DETR format
detr_train_ann = {'images': [], 'annotations': [], 'categories': []}
detr_val_ann = {'images': [], 'annotations': [], 'categories': []}

for i, cat in enumerate(train_ann['categories']):
    if cat['id'] in selected_classes:
        cat['id'] = i
        detr_train_ann['categories'].append(cat)

for i, cat in enumerate(val_ann['categories']):
    if cat['id'] in selected_classes:
        cat['id'] = i
        detr_val_ann['categories'].append(cat)

for img in train_ann['images']:
    new_img = {'file_name': img['file_name'], 'id': img['id'], 'height': img['height'], 'width': img['width']}
    detr_train_ann['images'].append(new_img)

for img in val_ann['images']:
    new_img = {'file_name': img['file_name'], 'id': img['id'], 'height': img['height'], 'width': img['width']}
    detr_val_ann['images'].append(new_img)

ann_id = 0
for ann in train_ann['annotations']:
    if ann['category_id'] in selected_classes:
        new_ann = {'bbox': ann['bbox'], 'image_id': ann['image_id'], 'category_id': selected_classes.index(ann['category_id']), 'id': ann_id, 'iscrowd': 0, 'area': ann['area']}
        detr_train_ann['annotations'].append(new_ann)
        ann_id += 1

ann_id = 0
for ann in val_ann['annotations']:
    if ann['category_id'] in selected_classes:
        new_ann = {'bbox': ann['bbox'], 'image_id': ann['image_id'], 'category_id': selected_classes.index(ann['category_id']), 'id': ann_id, 'iscrowd': 0, 'area': ann['area']}
        detr_val_ann['annotations'].append(new_ann)
        ann_id += 1

# Save DETR annotations as JSON files
detr_train_path = 'detr_train.json'
detr_val_path = 'detr_val.json'

with open(detr_train_path, 'w') as f:
    json.dump(detr_train_ann, f)

with open(detr_val_path, 'w') as f:
    json.dump(detr_val_ann, f)

