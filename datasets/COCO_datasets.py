import torch
from PIL import Image
import numpy as np
import json
import os
from tqdm import tqdm
import argparse
import concurrent.futures
import sys

# Create the argument parser
parser = argparse.ArgumentParser(description="Script to load COCO data")

# Add the path arguments with default values
parser.add_argument("--train_data_path", default=r'./path/to/coco', type=str,
                    help="Path to the training data")
parser.add_argument("--val_data_path", default=r'./path/to/coco', type=str,
                    help="Path to the validation data")

# Parse the arguments
args = parser.parse_args()

# If the user specified a different path, use it instead of the default
if args.train_data_path != r'./path/to/coco':
    train_data_path = args.train_data_path
if args.val_data_path != r'./path/to/coco':
    val_data_path = args.val_data_path

# Load the COCO categories
with open(args.train_data_path + "/annotations/coco_categories.json", "r") as f:
    categories = json.load(f)

# Load the selected images and annotations
with open(args.train_data_path + "/annotations/selected_train.json", "r") as f:
    train_data = json.load(f)
with open(args.val_data_path + "/annotations/selected_val.json", "r") as f:
    val_data = json.load(f)

# Create a dictionary to map category names to category IDs
category_id_map = {}
for category in categories:
    category_id_map[category["name"]] = category["id"]

# Define the function to process the images and targets
def process_image(image):
    try:
        # Load the image
        image_path = "./path/to/coco/train2017/" + image["file_name"]
        img = Image.open(image_path).convert("RGB")
        img_width, img_height = img.size

        # Create the list of boxes and labels for the image
        boxes = []
        labels = []
        for annotation in train_data["annotations"]:
            if annotation["image_id"] == image["id"]:
                x, y, w, h = annotation["bbox"]
                x_center = x + w / 2
                y_center = y + h / 2
                x_center /= img_width
                y_center /= img_height
                w /= img_width
                h /= img_height
                boxes.append([x_center, y_center, w, h])
                labels.append(category_id_map[categories[annotation["category_id"] - 1]["name"]])

        # Return the image and targets
        result = (np.array(img), {
            "boxes": torch.tensor(boxes, dtype=torch.float),
            "labels": torch.tensor(labels, dtype=torch.long)
        })
        # Free up memory
        del img
        return result
    except Exception as e:
        print(f"Error processing image '{image['file_name']}': {e}")
        sys.exit(1)

# Create the list of images and targets for training
train_images = []
train_targets = []

total_images = len(train_data["images"])

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(tqdm(executor.map(process_image, train_data["images"]), desc="Processing training images", unit=" images", ncols=80, total=total_images))
    for result in results:
        if result is not None:
            train_images.append(result[0])
            train_targets.append(result[1])
        executor.update(1)

# Define the function to process the validation images and targets
def process_val_image(image):
    try:
        # Load the image
        image_path = "./path/to/coco/val2017/" + image["file_name"]
        img = Image.open(image_path).convert("RGB")
        img_width, img_height = img.size

        # Create the list of boxes and labels for the image
        boxes = []
        labels = []
        for annotation in val_data["annotations"]:
            if annotation["image_id"] == image["id"]:
                x, y, w, h = annotation["bbox"]
                x_center = x + w / 2
                y_center = y + h / 2
                x_center /= img_width
                y_center /= img_height
                w /= img_width
                h /= img_height
                boxes.append([x_center, y_center, w, h])
                labels.append(category_id_map[categories[annotation["category_id"] - 1]["name"]])

        # Return the image and targets
        result = (np.array(img), {
            "boxes": torch.tensor(boxes, dtype=torch.float),
            "labels": torch.tensor(labels, dtype=torch.long)
        })
        # Free up memory
        del img
        return result
    except Exception as e:
        print(f"Error processing image {image['file_name']}: {str(e)}")
        sys.exit(1)

# Create the list of images and targets for validation
val_images = []
val_targets = []

total_images = len(val_data["images"])

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(tqdm(executor.map(process_val_image, val_data["images"]), desc="Processing validation images", unit=" images", ncols=80, total=total_images))
    for result in results:
        if result is not None:
            val_images.append(result[0])
            val_targets.append(result[1])
        executor.update(1)

# Save the data
torch.save((train_images, train_targets), "train_data.pt")
torch.save((val_images, val_targets), "val_data.pt")
print("Data processing complete.")
