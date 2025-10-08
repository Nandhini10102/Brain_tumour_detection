# 3DataAugmentation.py
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset path
base_path = r"C:\Users\nandhinimani\OneDrive\Desktop\Brain_Tumour\data\Tumour"

train_path = os.path.join(base_path, "train")
valid_path = os.path.join(base_path, "valid")
test_path  = os.path.join(base_path, "test")


# Image size & batch
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ---------------------------
# Data Augmentation
# ---------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,            # rotate images by up to 20°
    width_shift_range=0.25,       # horizontal shift
    height_shift_range=0.25,      # vertical shift
    shear_range=0.2,   
    zoom_range=0.3,               # zoom in/out
    horizontal_flip=True,         # flip horizontally
    vertical_flip=True,           # flip vertically
    brightness_range=[0.5, 1.5],  # brightness adjustment
    fill_mode="nearest"           # fill pixels after rotation/shift
)

valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

# ---------------------------
# Data Generators
# ---------------------------
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

valid_generator = valid_datagen.flow_from_directory(
    valid_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# ---------------------------
# Preview Augmented Images
# ---------------------------
x_batch, y_batch = next(train_generator)

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_batch[i])   # already scaled 0–1
    plt.axis("off")

plt.suptitle("Augmented Images (Random Transformations)", fontsize=16)
plt.show()
