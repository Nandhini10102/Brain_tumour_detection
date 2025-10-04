import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import os

# ----------------------------
# Config
# ----------------------------
test_data_dir = r"C:\Users\nandhinimani\OneDrive\Desktop\Brain_Tumour\data\Tumour\valid"
classes = sorted(os.listdir(test_data_dir))
models_paths = {
    "Custom CNN": r"C:\Users\nandhinimani\OneDrive\Desktop\Brain_Tumour\models\custom_cnn.keras",
    "EfficientNetB0": r"C:\Users\nandhinimani\OneDrive\Desktop\Brain_Tumour\models\best_transfer.keras",
    "ResNet50 Transfer": r"C:\Users\nandhinimani\OneDrive\Desktop\Brain_Tumour\models\resnet50_transfer.keras"
}

# ----------------------------
# Load test data for a model
# ----------------------------
def load_test_data_for_model(model):
    input_shape = model.input_shape[1:3]  # (height, width)
    X, y = [], []
    
    for idx, cls in enumerate(classes):
        cls_path = os.path.join(test_data_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for img_file in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_file)
            if os.path.isfile(img_path) and img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = image.load_img(img_path, target_size=input_shape)
                img_array = image.img_to_array(img)
                X.append(img_array)
                y.append(idx)
                
    X = np.array(X, dtype="float32") / 255.0
    y = to_categorical(y, num_classes=len(classes))
    return X, y

# ----------------------------
# Evaluate and compare models
# ----------------------------
comparison_results = {}

for model_name, model_path in models_paths.items():
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        continue
    
    print(f"\nLoading {model_name}...")
    model = load_model(model_path)
    
    X_test, y_test = load_test_data_for_model(model)
    
    # Measure inference time
    start_time = time.time()
    y_pred_probs = model.predict(X_test)
    end_time = time.time()
    
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    
    avg_time_per_image = (end_time - start_time) / len(X_test)
    
    comparison_results[model_name] = {
        "accuracy": acc,
        "precision": np.mean([report[c]['precision'] for c in classes]),
        "recall": np.mean([report[c]['recall'] for c in classes]),
        "f1_score": np.mean([report[c]['f1-score'] for c in classes]),
        "avg_inference_time": avg_time_per_image
    }

# ----------------------------
# Print comparison table
# ----------------------------
print("\nModel Comparison Summary:")
for model, metrics in comparison_results.items():
    print(f"\n{model}:")
    for k, v in metrics.items():
        if k == "avg_inference_time":
            print(f"  {k}: {v:.5f} sec/image")
        else:
            print(f"  {k}: {v:.4f}")

# ----------------------------
# Optional: Visualize accuracy & F1-score
# ----------------------------
model_names = list(comparison_results.keys())
accuracy_vals = [comparison_results[m]['accuracy'] for m in model_names]
f1_vals = [comparison_results[m]['f1_score'] for m in model_names]

plt.figure(figsize=(8,5))
plt.bar(model_names, accuracy_vals, color='skyblue', label='Accuracy')
plt.bar(model_names, f1_vals, color='orange', alpha=0.6, label='F1-score')
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.show()

# ----------------------------
# Determine the best model
# ----------------------------
# Normalize inference times to [0,1] (smaller is better)
times = np.array([comparison_results[m]['avg_inference_time'] for m in model_names])
time_scores = 1 - (times / times.max())  # higher = better

best_score = -1
best_model = None

for idx, model in enumerate(model_names):
    score = (
        0.5 * comparison_results[model]['accuracy'] +
        0.4 * comparison_results[model]['f1_score'] +
        0.1 * time_scores[idx]
    )
    comparison_results[model]['overall_score'] = score
    if score > best_score:
        best_score = score
        best_model = model

# ----------------------------
# Print best model
# ----------------------------
print(f"\n🏆 Best Model for Deployment: {best_model}")
print(f"Overall Score: {best_score:.4f}")
