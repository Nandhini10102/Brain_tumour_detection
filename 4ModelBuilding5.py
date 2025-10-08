# 5_Transfer_Learning.py
# -----------------------
# Step 5: Transfer Learning + Fine-tuning Model

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from DataAugmentation3 import train_generator, valid_generator, test_generator  # ✅ Import from Step 3

# ------------------------------
# Model Setup
# ------------------------------
num_classes = len(train_generator.class_indices)
use_transfer_learning = True  # False = custom CNN

if use_transfer_learning:
    # Transfer Learning: ResNet50
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base_model.trainable = False  # Freeze base layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.4)(x)

    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)

else:
    # Custom CNN
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

# ------------------------------
# Compile Model
# ------------------------------
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ------------------------------
# Callbacks
# ------------------------------
checkpoint = ModelCheckpoint("best_model.keras", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# ------------------------------
# Train Model
# ------------------------------
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=25,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# ------------------------------
# Evaluate Model
# ------------------------------
test_loss, test_acc = model.evaluate(test_generator)
print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")

y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys()))

# ------------------------------
# Fine-Tuning (optional)
# ------------------------------
if use_transfer_learning:
    # Unfreeze top layers for fine-tuning
    for layer in base_model.layers[-10:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n🔁 Fine-tuning ResNet50 top layers...\n")
    fine_tune_history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=10,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    print("\n✅ Fine-tuning Complete!")
