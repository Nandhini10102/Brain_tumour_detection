# =============================
# Step 1: Understand the Dataset
# =============================

import os
import cv2
import random
import matplotlib.pyplot as plt

# Path to TRAIN dataset
dataset_path = "data/Tumour/train"  # <- make sure your dataset is inside data/Tumour/train

# 1. Review categories
categories = [c for c in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, c))]
print("Tumor Categories:", categories)

# Helper: get valid image files
def get_image_files(category):
    valid_exts = (".jpg", ".jpeg", ".png", ".tif", ".bmp")
    return [f for f in os.listdir(os.path.join(dataset_path, category)) if f.lower().endswith(valid_exts)]

# --- Show one sample image per category ---
fig, axs = plt.subplots(1, len(categories), figsize=(15,5))
if len(categories) == 1: axs = [axs]

for i, category in enumerate(categories):
    files = get_image_files(category)
    if len(files) > 0:
        img_path = os.path.join(dataset_path, category, random.choice(files))
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[i].imshow(img)
        axs[i].set_title(category)
        axs[i].axis("off")
    else:
        axs[i].text(0.5, 0.5, "No Images", ha="center", va="center")
        axs[i].set_title(category)
        axs[i].axis("off")
plt.show()

# 2. Check for class imbalance
counts = [len(get_image_files(c)) for c in categories]
print("Image counts per class:", dict(zip(categories, counts)))

plt.bar(categories, counts)
plt.title("Class Distribution (Train Set)")
plt.xlabel("Tumor Type")
plt.ylabel("Number of Images")
plt.show()

# 3. Check image resolution consistency
sample_shapes = []
for category in categories:
    files = get_image_files(category)
    if len(files) > 0:
        img_path = os.path.join(dataset_path, category, random.choice(files))
        img = cv2.imread(img_path)
        sample_shapes.append(img.shape)
    else:
        sample_shapes.append(("Empty Folder",))

print("\nSample image resolutions from each class:")
for cat, shape in zip(categories, sample_shapes):
    print(f"{cat}: {shape}")

# 4. Explore image distributions visually (10 random images overall)
fig, axs = plt.subplots(2, 5, figsize=(15, 6))
axs = axs.ravel()

for i in range(10):
    category = random.choice(categories)
    files = get_image_files(category)
    if len(files) > 0:
        img_path = os.path.join(dataset_path, category, random.choice(files))
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[i].imshow(img)
        axs[i].set_title(category)
        axs[i].axis("off")
    else:
        axs[i].text(0.5, 0.5, "No Images", ha="center", va="center")
        axs[i].set_title(category)
        axs[i].axis("off")

plt.tight_layout()
plt.show()
