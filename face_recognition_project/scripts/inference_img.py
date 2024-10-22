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
print(f"Number of Classes: {len(person_names)}")

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set a confidence threshold (e.g., 50%)
confidence_threshold = 50.0

# Load the input image
image_path = 'dataset/shorted/Zhu_Rongji/newtest.jpg'  # Replace with your image path
image = cv2.imread(image_path)

if image is None:
    print("Error loading image")
else:
    # Convert image to grayscale (required for face detection)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through the faces and process each one
    for (x, y, w, h) in faces:
        # Draw a bounding box around the face
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extract the face region from the image
        face = image[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (200, 200))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)  # Add batch dimension

        # Make predictions for the face
        predictions = model.predict(face)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class] * 100  # Get confidence in percentage

        # Debugging: Print the predicted class index, class label, and confidence
        print(f"Predicted Class Index: {predicted_class}")
        print(f"Predicted Confidence: {confidence:.2f}%")

        # Check if the predicted class is within the range of person_names
        if predicted_class < len(person_names):
            label = person_names[predicted_class]
            confidence_text = f"{label} ({confidence:.2f}%)"
        else:
            label = "Unknown"
            confidence_text = f"Unknown ({confidence:.2f}%)"

        # Display the confidence on the bounding box
        cv2.putText(image, confidence_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the image with bounding boxes and labels
    cv2.imshow("Face Recognition", image)

    # Wait for the user to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
