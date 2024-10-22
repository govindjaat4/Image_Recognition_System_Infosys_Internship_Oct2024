from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

# Load the trained model
model = load_model('model/face_recognition_model.h5')

# Directory where training images are stored
# dataset_dir = 'dataset/train'

dataset_dir = 'dataset/shorted'
# Get person names (ensure this matches the number of classes the model was trained on)
person_names = os.listdir(dataset_dir)
print(f"Person Names: {person_names}")