import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
import datetime

# Corrected TensorFlow Hub Model URL (Make sure this is the right URL)
MODEL_URL = "https://www.kaggle.com/models/google/mobilenet-v2/TensorFlow2/035-128-classification/2"

# Load dataset
labels_csv = pd.read_csv("drive/MyDrive/Dog Vision/labels.csv")

# Prepare filenames
filenames = ["drive/MyDrive/Dog Vision/train/" + fname + ".jpg" for fname in labels_csv["id"]]

# Convert labels to numerical format
labels = labels_csv["breed"].values
unique_breeds = np.unique(labels)

# Convert labels into boolean arrays
boolean_labels = np.array([[label == breed for breed in unique_breeds] for label in labels])

# Split data into training and validation sets
NUM_IMAGES = 1000  # Adjust as needed
X_train, X_val, y_train, y_val = train_test_split(
    filenames[:NUM_IMAGES],
    boolean_labels[:NUM_IMAGES],
    test_size=0.2,
    random_state=42
)

# Image preprocessing function
IMG_SIZE = 224

def process_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    return image

# Function to get image and label
def get_image_label(image_path, label):
    image = process_image(image_path)
    return image, label

# Function to create data batches
BATCH_SIZE = 32

def create_data_batches(X, y=None, batch_size=BATCH_SIZE, test_data=False, valid_data=False):
    if test_data:
        data = tf.data.Dataset.from_tensor_slices(tf.constant(X))
        data_batch = data.map(process_image).batch(batch_size)
        return data_batch
    elif valid_data:
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        data_batch = data.map(get_image_label).batch(batch_size)
        return data_batch
    else:
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        data = data.shuffle(buffer_size=len(X))
        data_batch = data.map(get_image_label).batch(batch_size)
        return data_batch

# Create training and validation data batches
train_data = create_data_batches(X_train, y_train)
val_data = create_data_batches(X_val, y_val, valid_data=True)

# Define model parameters
INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3]
OUTPUT_SHAPE = len(unique_breeds)

# Function to create the model
def create_model(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url=MODEL_URL):
    print("Building model with:", model_url)
    model = tf.keras.Sequential([
        hub.KerasLayer(model_url, trainable=False),
        tf.keras.layers.Dense(units=output_shape, activation="softmax")
    ])
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )
    model.build(input_shape)
    return model

# TensorBoard callback function
def create_tensorboard_callback():
    logdir = os.path.join("drive/MyDrive/Dog Vision/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    return tf.keras.callbacks.TensorBoard(logdir)

# Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3)

# Training function
NUM_EPOCHS = 100  # Adjust as needed

def train_model():
    model = create_model()
    tensorboard_callback = create_tensorboard_callback()
    model.fit(
        x=train_data,
        epochs=NUM_EPOCHS,
        validation_data=val_data,
        validation_freq=1,
        callbacks=[tensorboard_callback, early_stopping]
    )
    return model

# Function to unbatch dataset
def unbatchify(data):
    images, labels = [], []
    for image, label in data.unbatch().as_numpy_iterator():
        images.append(image)
        labels.append(label)
    return images, labels

# Save and load model functions
def save_model(model, suffix=None):
    model_dir = os.path.join("drive/MyDrive/Dog Vision/models", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    model_path = model_dir + "-" + suffix + ".h5"
    model.save(model_path)
    return model_path

def load_model(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={"KerasLayer": hub.KerasLayer})

# Training on full dataset
full_data = create_data_batches(filenames, boolean_labels)
full_model = create_model()
full_model_tensorboard = create_tensorboard_callback()
full_model_early_stopping = tf.keras.callbacks.EarlyStopping(monitor="accuracy", patience=3)

full_model.fit(
    x=full_data,
    epochs=NUM_EPOCHS,
    callbacks=[full_model_tensorboard, full_model_early_stopping]
)

# Save and reload the trained model
model_path = save_model(full_model, suffix="full-image-set-mobilenetv2-Adam")
loaded_full_model = load_model(model_path)

# Making predictions on test dataset
test_path = "drive/MyDrive/Dog Vision/test/"
test_filenames = [os.path.join(test_path, fname) for fname in os.listdir(test_path)]
test_data = create_data_batches(test_filenames, test_data=True)

test_predictions = loaded_full_model.predict(test_data, verbose=1)
np.savetxt("drive/MyDrive/Dog Vision/test_predictions.csv", test_predictions, delimiter=',')
