import json
import os
from tqdm import tqdm

selected_categories = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light']
selected_category_ids = []

with open("./path/to/coco/annotations/detr_train.json", "r") as f:
    train_data = json.load(f)
    categories = train_data["categories"]
    for category in categories:
        if category["name"] in selected_categories:
            selected_category_ids.append(category["id"])

    images = []
    annotations = []
    for image in tqdm(train_data["images"], desc="Processing train data"):
        image_id = image["id"]
        image_annotations = []
        for annotation in train_data["annotations"]:
            if annotation["image_id"] == image_id and annotation["category_id"] in selected_category_ids:
                image_annotations.append(annotation)
        if len(image_annotations) > 0:
            images.append(image)
            annotations.extend(image_annotations)

    train_data["images"] = images
    train_data["annotations"] = annotations

    with open("./path/to/coco/annotations/selected_train.json", "w") as f:
        json.dump(train_data, f)

with open("./path/to/coco/annotations/detr_val.json", "r") as f:
    val_data = json.load(f)
    categories = val_data["categories"]
    for category in categories:
        if category["name"] in selected_categories:
            selected_category_ids.append(category["id"])

    images = []
    annotations = []
    for image in tqdm(val_data["images"], desc="Processing validation data"):
        image_id = image["id"]
        image_annotations = []
        for annotation in val_data["annotations"]:
            if annotation["image_id"] == image_id and annotation["category_id"] in selected_category_ids:
                image_annotations.append(annotation)
        if len(image_annotations) > 0:
            images.append(image)
            annotations.extend(image_annotations)

    val_data["images"] = images
    val_data["annotations"] = annotations

    with open("./path/to/coco/annotations/selected_val.json", "w") as f:
        json.dump(val_data, f)
