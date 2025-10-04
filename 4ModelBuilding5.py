import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ------------------------------
# Paths
# ------------------------------
train_path = "data/Tumour/train"
valid_path = "data/Tumour/valid"
test_path = "data/Tumour/test"

# ------------------------------
# Data Generators
# ------------------------------
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

valid_generator = valid_datagen.flow_from_directory(
    valid_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

num_classes = len(train_generator.class_indices)

# ------------------------------
# Model Choice
# ------------------------------
use_transfer_learning = True  # Set False for custom CNN

if use_transfer_learning:
    # Transfer Learning with ResNet50
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base_model.trainable = False  # Freeze base initially

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
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
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ------------------------------
# Callbacks
# ------------------------------
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# ------------------------------
# Train Model
# ------------------------------
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=20,
    callbacks=[checkpoint, early_stop]
)

# ------------------------------
# Evaluate on Test Set
# ------------------------------
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Predictions
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# Confusion Matrix & Classification Report
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))

print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys()))

# ------------------------------
# Fine-Tuning ResNet50
# ------------------------------
if use_transfer_learning:
    # Unfreeze the top 10 layers of ResNet50
    for layer in base_model.layers[-10:]:
        layer.trainable = True

    # Compile model with a lower learning rate
    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nStarting fine-tuning of top ResNet50 layers...\n")
    
    # Fine-tune
    fine_tune_history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=10,  # Usually fewer epochs for fine-tuning
        callbacks=[checkpoint, early_stop]
    )
    # ------------------------------
# Fine-Tuning ResNet50
# ------------------------------
if use_transfer_learning:
    # Unfreeze the top 10 layers of ResNet50
    for layer in base_model.layers[-10:]:
        layer.trainable = True

    # Compile model with a lower learning rate
    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nStarting fine-tuning of top ResNet50 layers...\n")
    
    # Fine-tune
    fine_tune_history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=10,  # Usually fewer epochs for fine-tuning
        callbacks=[checkpoint, early_stop]
    )

# ------------------------------
# Fine-Tuning ResNet50
# ------------------------------
if use_transfer_learning:
    # Unfreeze the top 10 layers of ResNet50
    for layer in base_model.layers[-10:]:
        layer.trainable = True

    # Compile model with a lower learning rate
    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nStarting fine-tuning of top ResNet50 layers...\n")
    
    # Fine-tune
    fine_tune_history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=10,  # Usually fewer epochs for fine-tuning
        callbacks=[checkpoint, early_stop]
    )
# ------------------------------
# Fine-Tuning ResNet50
# ------------------------------
if use_transfer_learning:
    # Unfreeze the top 10 layers of ResNet50
    for layer in base_model.layers[-10:]:
        layer.trainable = True

    # Compile model with a lower learning rate
    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nStarting fine-tuning of top ResNet50 layers...\n")
    
    # Fine-tune
    fine_tune_history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=10,  # Usually fewer epochs for fine-tuning
        callbacks=[checkpoint, early_stop]
    )
# ------------------------------
# Fine-Tuning ResNet50
# ------------------------------
if use_transfer_learning:
    # Unfreeze the top 10 layers of ResNet50
    for layer in base_model.layers[-10:]:
        layer.trainable = True

    # Compile model with a lower learning rate
    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nStarting fine-tuning of top ResNet50 layers...\n")
    
    # Fine-tune
    fine_tune_history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=10,  # Usually fewer epochs for fine-tuning
        callbacks=[checkpoint, early_stop]
    )
# ------------------------------
# Fine-Tuning ResNet50
# ------------------------------
if use_transfer_learning:
    # Unfreeze the top 10 layers of ResNet50
    for layer in base_model.layers[-10:]:
        layer.trainable = True

    # Compile model with a lower learning rate
    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nStarting fine-tuning of top ResNet50 layers...\n")
    
    # Fine-tune
    fine_tune_history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=10,  # Usually fewer epochs for fine-tuning
        callbacks=[checkpoint, early_stop]
    )
# ------------------------------
# Fine-Tuning ResNet50
# ------------------------------
if use_transfer_learning:
    # Unfreeze the top 10 layers of ResNet50
    for layer in base_model.layers[-10:]:
        layer.trainable = True

    # Compile model with a lower learning rate
    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nStarting fine-tuning of top ResNet50 layers...\n")
    
    # Fine-tune
    fine_tune_history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=10,  # Usually fewer epochs for fine-tuning
        callbacks=[checkpoint, early_stop]
    )


# ------------------------------
# Fine-Tuning ResNet50
# ------------------------------
if use_transfer_learning:
    # Unfreeze the top 10 layers of ResNet50
    for layer in base_model.layers[-10:]:
        layer.trainable = True

    # Compile model with a lower learning rate
    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nStarting fine-tuning of top ResNet50 layers...\n")
    
    # Fine-tune
    fine_tune_history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=10,  # Usually fewer epochs for fine-tuning
        callbacks=[checkpoint, early_stop]
    )
