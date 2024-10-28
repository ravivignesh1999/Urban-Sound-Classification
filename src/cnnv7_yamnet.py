import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import warnings
import librosa
from tensorflow.keras.callbacks import History, CSVLogger

warnings.filterwarnings("ignore", message=".*n_fft.*too large.*")

# Load the dataset
data_dir = "C:\\Users\\Hemant Sarmukaddam\\Desktop\\Research\\UrbanSound8K"
metadata_file = os.path.join(data_dir, "metadata", "UrbanSound8K.csv")
audio_dir = os.path.join(data_dir, "audio")
metadata = pd.read_csv(metadata_file)

# Load the YAMNet model
model = hub.load('https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1')

def extract_features(file_path, max_length=40):
    signal, sr = librosa.load(file_path, sr=22050)
    audio = tf.convert_to_tensor(signal, dtype=tf.float32)
    scores, embeddings, log_mel_spectrogram = model(audio)
    # Pad or truncate the embeddings to ensure consistent length
    if len(embeddings) < max_length:
        embeddings = tf.pad(embeddings, [[0, max_length - len(embeddings)], [0, 0]])
    else:
        embeddings = embeddings[:max_length]
    
    return embeddings


feature_file = "yamnet_features.npz"

if os.path.exists(feature_file):
    print("Loading features from file...")
    data = np.load(feature_file)
    X = data["X"]
    y = data["y"]
else:
    print("Extracting features and saving to file...")
    X = []
    y = []
    for index, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Loading data"):
        fold_dir = os.path.join(audio_dir, f"fold{row['fold']}")
        file_path = os.path.join(fold_dir, row['slice_file_name'])
        if os.path.isfile(file_path):
            try:
                features = extract_features(file_path)
                X.append(features)
                y.append(row['classID'])
            except Exception as e:
                print(f"Error encountered while processing {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")
    
    X = np.array(X)
    y = to_categorical(y, num_classes=len(np.unique(y)))
    np.savez(feature_file, X=X, y=y)

# Split data into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# Build the custom classifier
input_shape = X_train.shape[1:]
input_layer = tf.keras.Input(shape=input_shape)
x = tf.keras.layers.Dense(128, activation='relu')(input_layer)
x = tf.keras.layers.Dropout(0.5)(x)
global_average_layer = tf.keras.layers.GlobalAveragePooling1D()(x)
output_layer = tf.keras.layers.Dense(10, activation='softmax')(global_average_layer)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
csv_logger = CSVLogger('training_history_cnn_yamnet.csv')
# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks = [csv_logger])

# Evaluate the model
train_loss, train_accuracy = model.evaluate(X_train, y_train)
print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Save the model
model.save("audio_classification_model_yamnet.h5")