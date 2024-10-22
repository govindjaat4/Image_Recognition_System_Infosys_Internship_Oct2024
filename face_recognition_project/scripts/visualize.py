import cv2
import face_recognition
import os
image_folder=''
def visualize_faces(image_folder):
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_folder, filename)
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)

            # Show the image
            image_with_faces = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(image_with_faces, (left, top), (right, bottom), (0, 255, 0), 2)

            cv2.imshow(f"Face in {filename}", image_with_faces)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# Call the function
visualize_faces("captured_faces")
