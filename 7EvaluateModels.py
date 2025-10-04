import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ----------------------------
# Config
# ----------------------------
test_data_dir = r"C:\Users\nandhinimani\OneDrive\Desktop\Brain_Tumour\data\Tumour\valid"
classes = sorted(os.listdir(test_data_dir))  # Ensure classes are sorted
models_paths = {
    "Custom CNN": r"C:\Users\nandhinimani\OneDrive\Desktop\Brain_Tumour\models\custom_cnn.keras",
    "EfficientNetB0": r"C:\Users\nandhinimani\OneDrive\Desktop\Brain_Tumour\models\best_transfer.keras",
    "ResNet50 Transfer": r"C:\Users\nandhinimani\OneDrive\Desktop\Brain_Tumour\models\resnet50_transfer.keras"
}

# ----------------------------
# Function to load test images
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
# Function to evaluate model
# ----------------------------
def evaluate_model(model, X_test, y_test, model_name):
    print(f"\nEvaluating {model_name}...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ----------------------------
# Main evaluation loop
# ----------------------------
for model_name, model_path in models_paths.items():
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        continue
    
    print(f"\nLoading {model_name} from {model_path}")
    model = load_model(model_path)
    
    # Load test data matching model input shape
    X_test, y_test = load_test_data_for_model(model)
    
    # Evaluate
    evaluate_model(model, X_test, y_test, model_name)




