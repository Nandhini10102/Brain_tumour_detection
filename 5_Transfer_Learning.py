import zipfile
import os

# Path to your zip file
zip_path = "Tumour-20250915T133457Z-1-001.zip"
extract_path = "data"   # where to extract

# Unzip only if not already extracted
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"✅ Extracted to {extract_path}")
else:
    print("✅ Dataset already extracted")




import os
import numpy as np
from tensorflow.keras.applications import ResNet50, MobileNetV2, InceptionV3, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# -------------------------
# Paths
# -------------------------
train_path = "data/Tumour/train"
valid_path = "data/Tumour/valid"
test_path  = "data/Tumour/test"

IMG_SIZE = (160, 160)   # 🔥 smaller size for faster training
BATCH_SIZE = 16
EPOCHS = 10
MODEL_SAVE_PATH = "models/best_transfer.keras"

# -------------------------
# Data Generators
# -------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest"
)

valid_datagen = ImageDataGenerator(rescale=1./255)

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

num_classes = train_generator.num_classes

# -------------------------
# Function to build Transfer Model
# -------------------------
def build_transfer_model(base_model, num_classes):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    preds = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=preds)
    return model

# -------------------------
# Choose Pretrained Model
# -------------------------
def get_model(model_name, input_shape=(160,160,3), num_classes=4):
    if model_name == "resnet50":
        base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    elif model_name == "mobilenet":
        base = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    elif model_name == "inception":
        base = InceptionV3(weights="imagenet", include_top=False, input_shape=input_shape)
    elif model_name == "efficientnet":
        base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Unsupported model name")

    return build_transfer_model(base, num_classes)

# -------------------------
# Train Model
# -------------------------
if __name__ == "__main__":
    model_name = "resnet50"  # 🔥 change this to "mobilenet" / "inception" / "efficientnet"
    model = get_model(model_name, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=num_classes)

    model.compile(optimizer=Adam(1e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
    ]

    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    print(f"✅ Training complete. Best model saved at {MODEL_SAVE_PATH}")
