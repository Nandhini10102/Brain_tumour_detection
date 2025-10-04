📄 Brain Tumor MRI Image Classification

🧠 Project Overview

This project leverages Deep Learning techniques to classify brain MRI images into multiple categories of tumors.
It combines both a Custom CNN model and Transfer Learning (ResNet50, etc.), and provides a Streamlit web application for real-time predictions.

🚀 Features

✅ Preprocessing & Augmentation of MRI images

✅ Custom CNN built from scratch

✅ Transfer Learning with ResNet50

✅ Model comparison: Accuracy, Precision, Recall, F1-score

✅ Confusion Matrix visualization

✅ Streamlit app for interactive predictions

✅ User-friendly interface with model selection & confidence scores

📂 Project Structure

BRAIN_TUMOUR/
│
├── data/
│   └── Tumour/
│       ├── test/
│       │   ├── glioma/
│       │   ├── meningioma/
│       │   ├── no_tumor/
│       │   └── pituitary/
│       └── _classes.csv
│
├── train/
├── valid/
│
├── models/
│
├── README.dataset.txt
│
├── 1Dataset.py
├── 2Preprocessing.py
├── 3DataAugmentation.py
├── 4ModelBuilding5.py
├── 6TrainModels.py
├── 7EvaluateModels.py
├── 8ModelComparision.py
├── app.py
├── Brain_Tumor.ipynb
└── Tumour-20250915T... (some dataset or zip file)

🧩 Tech Stack

  # Python 3.x

  # TensorFlow / Keras

  # OpenCV

  # scikit-learn

  # Matplotlib / Seaborn

  # Streamlit

📊 Workflow

  1.Dataset Preparation

  # Unzipped & organized into train/valid/test

  # Normalized pixel values, resized to (160, 160)

  # Data Augmentation (rotation, flip, zoom)

  2.Model Development

  # Custom CNN with Conv2D, Pooling, BatchNorm, Dropout

  # Transfer Learning using ResNet50 (ImageNet weights)

  3.Training & Evaluation

  # EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

  # Metrics: Accuracy, Precision, Recall, F1-score

  # Confusion Matrix visualization

  4.Model Comparison

  # Custom CNN vs ResNet50

  # Performance plotted with bar charts

  # Best model chosen for deployment

  5.Deployment

  ## Streamlit app with:

  # MRI upload option

  # Prediction + confidence percentage

  # Confidence scores bar chart

  # Sidebar with sample MRI references

📈 Results

  Model	Accuracy	Precision	Recall	F1-score
  Custom CNN	XX%	XX%	XX%	XX%
  ResNet50 TL	XX%	XX%	XX%	XX%

  # ✅ ResNet50 Transfer Learning performed best and is deployed in the app.

🔮 Future Improvements

  # Add more pretrained models (EfficientNet, InceptionV3)

  # Deploy on Streamlit Cloud / HuggingFace Spaces

  # Integrate explainable AI (Grad-CAM heatmaps)

  # Optimize for mobile/edge devices

🙌 Acknowledgements

  # Dataset: Brain MRI Images for Brain Tumor Detection (Kaggle / standard dataset)

  # TensorFlow/Keras for deep learning framework

  # Streamlit for app deployment

👨‍💻 Author

S.Nandhini
📧 nnandhinisundhar@gmail.com
🌐 GitHub : https://github.com/Nandhini10102
