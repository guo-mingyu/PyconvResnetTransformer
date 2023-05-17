"""
Code Copyright (c) guo mingyu
"""
import os
import sys
import tarfile
import urllib.request
from tqdm import tqdm

data_dir = "coco"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# List of all the files to download
files = [
    ("http://images.cocodataset.org/zips/train2017.zip", "train2017.zip"),
    ("http://images.cocodataset.org/zips/val2017.zip", "val2017.zip"),
    ("http://images.cocodataset.org/zips/test2017.zip", "test2017.zip"),
    ("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", "annotations_trainval2017.zip"),
    ("http://images.cocodataset.org/annotations/image_info_test2017.zip", "image_info_test2017.zip")
]

# Download and extract files
for url, filename in tqdm(files, desc="Downloading COCO 2017", ncols=100):
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        with urllib.request.urlopen(url) as u, open(filepath, 'wb') as f:
            meta = u.info()
            file_size = int(meta.get_all("Content-Length")[0])
            pbar = tqdm(total=file_size, desc=filename, unit="B", unit_scale=True, unit_divisor=1024, ncols=100)
            file_size_dl = 0
            block_sz = 8192
            while True:
                buffer = u.read(block_sz)
                if not buffer:
                    break
                file_size_dl += len(buffer)
                pbar.update(len(buffer))
                f.write(buffer)
            pbar.close()
    if filename.endswith(".zip"):
        with tarfile.open(filepath) as tar:
            tar.extractall(data_dir)
        os.remove(filepath)

print("Download complete.")
