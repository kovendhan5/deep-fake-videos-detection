import numpy as np
import cv2
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array

# Preprocessing function for video frames
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))  # Resize frame to match ResNet50 input size
    frame = frame.astype("float") / 255.0  # Normalize pixel values to [0, 1]
    return img_to_array(frame)

# Load dataset (assuming frames are stored in two directories: 'real' and 'fake')
def load_dataset(real_dir, fake_dir):
    data = []
    labels = []

    # Load real frames
    for img_name in os.listdir(real_dir):
        img_path = os.path.join(real_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            processed_img = preprocess_frame(img)
            data.append(processed_img)
            labels.append(0)  # 0 for real

    # Load fake frames
    for img_name in os.listdir(fake_dir):
        img_path = os.path.join(fake_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            processed_img = preprocess_frame(img)
            data.append(processed_img)
            labels.append(1)  # 1 for deepfake

    return np.array(data), np.array(labels)

# Build and compile the model
def build_deepfake_model():
    # Load the pre-trained ResNet50 model without the top classification layers
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification layers
    head_model = base_model.output
    head_model = Flatten()(head_model)
    head_model = Dense(512, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)  # Dropout for regularization
    head_model = Dense(1, activation="sigmoid")(head_model)  # Binary classification (real or fake)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=head_model)

    # Compile the model with Adam optimizer
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"])

    return model

# Train the model
def train_deepfake_model(real_dir, fake_dir, batch_size=32, epochs=20):
    # Load the dataset
    data, labels = load_dataset(real_dir, fake_dir)

    # Split into training and testing sets
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Create data augmentation generators for training
    train_aug = ImageDataGenerator(rotation_range=15, fill_mode="nearest")

    # Build the model
    model = build_deepfake_model()

    # Train the model
    history = model.fit(train_aug.flow(trainX, trainY, batch_size=batch_size),
                        validation_data=(testX, testY),
                        steps_per_epoch=len(trainX) // batch_size,
                        epochs=epochs)

    # Save the trained model
    model.save("deepfake_detection_model.h5")

    return model

# Use this function to make predictions with the trained model
def detect_deepfake(model, frame):
    processed_frame = preprocess_frame(frame)
    processed_frame = np.expand_dims(processed_frame, axis=0)  # Add batch dimension
    prediction = model.predict(processed_frame)[0][0]

    return prediction > 0.5  # Return True if prediction is higher than 0.5 (indicating deepfake)
