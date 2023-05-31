import json
import os
from tqdm import tqdm


def filter_coco_dataset(coco_path, categories, output_path):
    # Load COCO annotations
    with open(coco_path, 'r') as f:
        coco = json.load(f)

    # Filter categories
    filtered_annotations = []
    filtered_categories = []
    image_ids = set()

    # Find image IDs of the filtered categories
    for annotation in tqdm(coco['annotations'], desc='Finding image IDs'):
        if annotation['category_id'] in categories:
            image_ids.add(annotation['image_id'])

    # Filter images and annotations
    filtered_images = []
    for image in tqdm(coco['images'], desc='Filtering images'):
        if image['id'] in image_ids:
            filtered_images.append(image)

    for category in tqdm(coco['categories'], desc='Filtering categories'):
        if category['id'] in categories:
            filtered_categories.append(category)

    for annotation in tqdm(coco['annotations'], desc='Filtering annotations'):
        if annotation['category_id'] in categories and annotation['image_id'] in image_ids:
            filtered_annotations.append(annotation)

    # Create new COCO dataset
    filtered_coco = {
        'info': coco['info'],
        'licenses': coco['licenses'],
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': filtered_categories
    }

    # Save filtered COCO dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(filtered_coco, f)


# List of category IDs to keep
category_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Path to the original COCO dataset
coco_train_path = 'path/to/coco/annotations/instances_train2017.json'
coco_val_path = 'path/to/coco/annotations/instances_val2017.json'

# Path to save the filtered COCO datasets
filtered_coco_train_path = 'path/to/filtered/coco/instances_train2017_filtered.json'
filtered_coco_val_path = 'path/to/filtered/coco/instances_val2017_filtered.json'

# Filter the COCO datasets
filter_coco_dataset(coco_train_path, category_ids, filtered_coco_train_path)
filter_coco_dataset(coco_val_path, category_ids, filtered_coco_val_path)
