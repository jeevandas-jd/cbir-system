import pickle
import os
import numpy as np
from PIL import Image

# Path to your dataset
DATA_PATH = "/home/jd/Downloads/cifar-10-python/cifar-10-batches-py"
SAVE_PATH = "data/raw"

os.makedirs(SAVE_PATH, exist_ok=True)

# Label names
with open(os.path.join(DATA_PATH, "batches.meta"), "rb") as f:
    meta = pickle.load(f, encoding="bytes")
    label_names = [label.decode("utf-8") for label in meta[b"label_names"]]

def load_batch(file):
    with open(file, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    return batch

img_count = 0

for i in range(1, 6):
    batch = load_batch(os.path.join(DATA_PATH, f"data_batch_{i}"))

    images = batch[b"data"]
    labels = batch[b"labels"]

    for j in range(len(images)):
        img = images[j].reshape(3, 32, 32)
        img = np.transpose(img, (1, 2, 0))

        label = label_names[labels[j]]
        label_dir = os.path.join(SAVE_PATH, label)
        os.makedirs(label_dir, exist_ok=True)

        img_path = os.path.join(label_dir, f"{img_count}.png")
        Image.fromarray(img).save(img_path)

        img_count += 1

print(f"✅ Converted {img_count} images")