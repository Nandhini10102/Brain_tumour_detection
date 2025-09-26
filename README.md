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

Brain_Tumour_Classification/
│── data/                   # Dataset (not uploaded to GitHub if large)
│   ├── Tumour/
│       ├── train/
│       ├── valid/
│       ├── test/
│── models/                 # Saved trained models (.keras)
│   ├── custom_cnn.keras
│   ├── resnet50_transfer.keras
│── notebooks/              # Jupyter notebooks for exploration
│   ├── Brain_Tumor.ipynb
│── scripts/                # All Python scripts
│   ├── 1_data_preprocessing.py
│   ├── 2_data_augmentation.py
│   ├── 3_custom_cnn.py
│   ├── 4_transfer_learning.py
│   ├── 5_train_and_evaluate.py
│   ├── 6_model_comparison.py
│── app.py                  # Streamlit application
│── requirements.txt        # Dependencies
│── README.md               # Project documentation

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
🌐 [Your Portfolio / LinkedIn / GitHub]
