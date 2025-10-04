import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

# Paths
train_path = "data/Tumour/train"
valid_path = "data/Tumour/valid"
test_path  = "data/Tumour/test"

# Categories from train folder
categories = [c for c in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, c))]
print("Categories:", categories)

IMG_SIZE = 224

def load_data(base_path, categories):
    X, y = [], []
    for idx, category in enumerate(categories):
        files = [f for f in os.listdir(os.path.join(base_path, category)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for file in files:
            img_path = os.path.join(base_path, category, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # resize to 224x224
            img = img / 255.0                           # normalize
            X.append(img)
            y.append(idx)
    return np.array(X), np.array(y)

# Load datasets
X_train, y_train = load_data(train_path, categories)
X_valid, y_valid = load_data(valid_path, categories)
X_test,  y_test  = load_data(test_path, categories)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=len(categories))
y_valid = to_categorical(y_valid, num_classes=len(categories))
y_test  = to_categorical(y_test,  num_classes=len(categories))

# Shapes
print("Train set:", X_train.shape, y_train.shape)
print("Valid set:", X_valid.shape, y_valid.shape)
print("Test set:",  X_test.shape,  y_test.shape)
