import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# Paths
# -----------------------------
test_path = "data/Tumour/test"
models_dir = "models"

# -----------------------------
# Data Generator
# -----------------------------
IMG_SIZE = (160, 160)
BATCH_SIZE = 32

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

class_labels = list(test_generator.class_indices.keys())

# -----------------------------
# Load Best Models
# -----------------------------
cnn_model = load_model(os.path.join(models_dir, "custom_cnn.keras"))
resnet_model = load_model(os.path.join(models_dir, "resnet50_transfer.keras"))

# -----------------------------
# Helper Function for Metrics
# -----------------------------
def get_metrics(model, name):
    preds = model.predict(test_generator, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_generator.classes
    
    acc  = np.mean(y_pred == y_true)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n🔎 {name} - Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_labels))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f"{name} - Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

    return acc, prec, rec, f1

# -----------------------------
# Evaluate Models
# -----------------------------
cnn_acc, cnn_prec, cnn_rec, cnn_f1 = get_metrics(cnn_model, "Custom CNN")
res_acc, res_prec, res_rec, res_f1 = get_metrics(resnet_model, "ResNet50 Transfer Learning")

# -----------------------------
# Compare Results (Bar Chart)
# -----------------------------
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
cnn_scores = [cnn_acc, cnn_prec, cnn_rec, cnn_f1]
res_scores = [res_acc, res_prec, res_rec, res_f1]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8,5))
plt.bar(x - width/2, cnn_scores, width, label="Custom CNN")
plt.bar(x + width/2, res_scores, width, label="ResNet50 TL")

plt.xticks(x, metrics)
plt.ylim(0,1)
plt.ylabel("Score")
plt.title("Model Comparison")
plt.legend()
plt.show()

# -----------------------------
# Final Recommendation
# -----------------------------
if res_acc > cnn_acc:
    print("\n✅ ResNet50 Transfer Learning is more accurate → Recommended for deployment 🚀")
else:
    print("\n✅ Custom CNN is more efficient → Recommended if speed & lightweight model is needed ⚡")
