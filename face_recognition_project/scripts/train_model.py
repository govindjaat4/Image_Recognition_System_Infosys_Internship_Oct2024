import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Directory paths
# train_dir = 'dataset/train/'

train_dir = 'dataset/shorted'

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,  # Increase the rotation range
    width_shift_range=0.3,  # Increase shifts
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=[0.8, 1.5],  # More zoom 
    brightness_range=[0.8, 1.2],  # Brightness variation
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical'
)

# Get person names
person_names = list(train_generator.class_indices.keys())

# Model architecture
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(person_names), activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(train_generator, epochs=30)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(train_generator, epochs=150, callbacks=[early_stopping])

# Save the trained model
model.save('model/face_recognition_model.h5')
