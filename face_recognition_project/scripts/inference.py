from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

# Load the trained model
model = load_model('model/face_recognition_model.h5')

dataset_dir = 'dataset/shorted'
# Get person names (ensure this matches the number of classes the model was trained on)
person_names = os.listdir(dataset_dir)
print(f"Person Names: {person_names}")
print(f"Number of Classes: {len(person_names)}")

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set a confidence threshold (e.g., 75%)
confidence_threshold = 75.0

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale (required for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))

    # Loop through the faces and process each one
    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face = frame[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (200, 200))
        face = face / 255.0
        face = np.expand_dims(face, axis=0) # Add batch dimension

        # Make predictions for the face
        predictions = model.predict(face)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class] * 100 # Get confidence in percentage

        # Check if the confidence is higher than the threshold
        if confidence > confidence_threshold:
            if predicted_class < len(person_names):
                label = person_names[predicted_class]
            else:
                label = "Unknown"
            box_color = (0, 255, 0) # Green for high confidence
        else:
            label = "Unknown"
            box_color = (0, 0, 255) # Red for low confidence

        # Display the confidence on the bounding box
        cv2.putText(frame, f"{label} ({confidence:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

    # Show the frame with bounding boxes and labels
    cv2.imshow("Face Recognition", frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
