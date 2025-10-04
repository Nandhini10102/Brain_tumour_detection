import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from PIL import Image

# ==========================
# Paths
# ==========================
train_path = r"C:\Users\nandhinimani\OneDrive\Desktop\Brain_Tumour\data\Tumour\train"
valid_path = r"C:\Users\nandhinimani\OneDrive\Desktop\Brain_Tumour\data\Tumour\valid"
test_path  = r"C:\Users\nandhinimani\OneDrive\Desktop\Brain_Tumour\data\Tumour\test"

# ==========================
# Convert grayscale images to RGB
# ==========================
for folder in [train_path, valid_path, test_path]:
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        for file in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path, file)
            img = Image.open(img_path).convert('RGB')  # Convert to RGB
            img.save(img_path)

# ==========================
# Data generators
# ==========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224,224),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    valid_path,
    target_size=(224,224),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical'
)

# ==========================
# Custom CNN
# ==========================
custom_cnn = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

custom_cnn.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# ==========================
# EfficientNetB0 Transfer Learning
# ==========================
base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',   # pretrained
    input_shape=(224,224,3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(train_generator.num_classes, activation='softmax')(x)

efficientnet_model = Model(inputs=base_model.input, outputs=outputs)

# Freeze base layers initially
for layer in base_model.layers:
    layer.trainable = False

efficientnet_model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# ==========================
# Callbacks
# ==========================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
]

# ==========================
# Training Custom CNN
# ==========================
custom_cnn.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=callbacks
)

# ==========================
# Training EfficientNetB0
# ==========================
efficientnet_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=callbacks
)

# ==========================
# Optional: Fine-tuning EfficientNetB0
# ==========================
# Unfreeze some top layers for fine-tuning
for layer in base_model.layers[-50:]:
    layer.trainable = True

efficientnet_model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

efficientnet_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=callbacks
)








