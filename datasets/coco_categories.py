import json
import os
from tqdm import tqdm

# Set the file paths
instances_path = "./path/to/coco/annotations/instances_train2017.json"
output_path = "./path/to/coco/annotations/coco_categories.json"

# Get the file size
file_size = os.path.getsize(instances_path)

# Load the COCO instances file
try:
    with open(instances_path, "r") as f:
        instances = json.load(f)
except FileNotFoundError:
    print(f"Error: File {instances_path} not found.")
    exit(1)
except json.JSONDecodeError:
    print(f"Error: Unable to decode JSON from file {instances_path}.")
    exit(1)

# Extract the categories
categories = instances["categories"]

# Save the categories to a new file
try:
    with open(output_path, "w") as f:
        for i, category in tqdm(enumerate(categories), total=len(categories), unit=" categories", desc="Extracting categories", ncols=80):
            json.dump(category, f)
            if i < len(categories) - 1:
                f.write("\n")
except FileNotFoundError:
    print(f"Error: Unable to open file {output_path} for writing.")
    exit(1)

print("Successfully extracted and saved the COCO categories to 'coco_categories.json'!")
