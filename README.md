# 🐶 Dog Vision - Dog Breed Classification using TensorFlow

This project implements a deep learning model to classify dog breeds using images, leveraging TensorFlow and a pre-trained MobileNetV2 model from TensorFlow Hub. It was developed and trained on Google Colab with a subset of the [Kaggle Dog Breed Identification dataset](https://www.kaggle.com/competitions/dog-breed-identification).

## 📁 Dataset

- The dataset consists of images of dogs, each labeled with its breed.
- Files are stored in:
  - `train/` - Training images
  - `test/` - Test images (for final predictions)
  - `labels.csv` - Contains mapping from image ID to breed

## 🧠 Model

- **Architecture**: Transfer Learning using MobileNetV2 as a base, with a custom Dense output layer.
- **Input Size**: 224x224 RGB images
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Evaluation Metric**: Accuracy

## 📦 Key Features

- Image preprocessing using `tf.image` utilities
- Efficient `tf.data.Dataset` batching for training and evaluation
- Train/validation split using `scikit-learn`
- Custom training callbacks:
  - TensorBoard logging
  - EarlyStopping on validation accuracy
- Full model training and saving
- Re-loading trained models for predictions
- Exports predictions on test set to CSV

## 🔍 File Structure

Dog Vision/ ├── train/ # Training images ├── test/ # Test images ├── labels.csv # Training labels ├── models/ # Trained model exports ├── logs/ # TensorBoard logs └── dog_vision_colab_notebook.ipynb # Main project notebook

## 🚀 How to Run

1. **Open the project notebook** in Google Colab.
2. **Mount Google Drive** for file access:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
Set dataset paths and run cells to:

Load & preprocess data

Train the model

Save model checkpoints

Generate predictions for the test set

📈 Training & Validation
Uses train_test_split to split the first 1000 images into training and validation (80-20).

Trained using 100 epochs (can be changed).

TensorBoard logs are saved to Google Drive for visualization.

🗂️ Model Management
Saved in .h5 format with timestamps and custom suffixes.

Reloadable using a helper function with custom_objects={"KerasLayer": hub.KerasLayer}.

📊 Output
Trained model predictions on test set saved to:

bash
Copy
Edit
drive/MyDrive/Dog Vision/test_predictions.csv
🔧 Dependencies
TensorFlow 2.x

TensorFlow Hub

Pandas, NumPy, scikit-learn

Google Colab (for runtime and Drive integration)

✨ Credits
Created by Anirudh on Google Colab as part of a deep learning project exploring transfer learning in computer vision.

Feel free to contribute or fork the repo
