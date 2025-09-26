import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# -----------------------------
# Paths
# -----------------------------
zip_path = "Tumour-20250915T133457Z-1-001.zip"
extract_path = "data"
train_path = os.path.join(extract_path, "Tumour/train")
valid_path = os.path.join(extract_path, "Tumour/valid")
test_path  = os.path.join(extract_path, "Tumour/test")
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# -----------------------------
# Unzip dataset (only once)
# -----------------------------
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✅ Dataset extracted")
else:
    print("✅ Dataset already extracted")

# -----------------------------
# Data Generators
# -----------------------------
IMG_SIZE = (160, 160)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)
valid_generator = valid_datagen.flow_from_directory(
    valid_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)
test_generator = test_datagen.flow_from_directory(
    test_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
)

num_classes = train_generator.num_classes
class_labels = list(train_generator.class_indices.keys())

# -----------------------------
# Callbacks
# -----------------------------
cb_early = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
cb_rlrop = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)

# -----------------------------
# Helper: Plot training history
# -----------------------------
def plot_history(history, title):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history["accuracy"], label="train acc")
    plt.plot(history.history["val_accuracy"], label="val acc")
    plt.title(title + " - Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.title(title + " - Loss")
    plt.legend()
    plt.show()

# -----------------------------
# Model 1: Custom CNN
# -----------------------------
cnn = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(160,160,3)),
    MaxPooling2D((2,2)),
    BatchNormalization(),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    BatchNormalization(),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    BatchNormalization(),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

cnn.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
chkpt_cnn = ModelCheckpoint(os.path.join(models_dir, "custom_cnn.keras"), monitor="val_loss", save_best_only=True, verbose=1)

print("🚀 Training Custom CNN...")
history_cnn = cnn.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=20,
    callbacks=[cb_early, cb_rlrop, chkpt_cnn],
    verbose=1
)

plot_history(history_cnn, "Custom CNN")

# -----------------------------
# Model 2: Transfer Learning (ResNet50)
# -----------------------------
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(160,160,3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
preds = Dense(num_classes, activation="softmax")(x)

resnet_model = Model(inputs=base_model.input, outputs=preds)
resnet_model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
chkpt_resnet = ModelCheckpoint(os.path.join(models_dir, "resnet50_transfer.keras"), monitor="val_loss", save_best_only=True, verbose=1)

print("🚀 Training ResNet50 Transfer Learning...")
history_resnet = resnet_model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=10,
    callbacks=[cb_early, cb_rlrop, chkpt_resnet],
    verbose=1
)

plot_history(history_resnet, "ResNet50 Transfer Learning")

# -----------------------------
# Evaluation Helper
# -----------------------------
def evaluate_model(model, name):
    print(f"\n🔎 Evaluating {name}...")
    preds = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_generator.classes
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f"{name} - Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

# -----------------------------
# Load best models & Evaluate
# -----------------------------
best_cnn = load_model(os.path.join(models_dir, "custom_cnn.keras"))
best_resnet = load_model(os.path.join(models_dir, "resnet50_transfer.keras"))

evaluate_model(best_cnn, "Custom CNN")
evaluate_model(best_resnet, "ResNet50 Transfer Learning")

print("✅ Training + Evaluation complete. Models saved in /models/")

