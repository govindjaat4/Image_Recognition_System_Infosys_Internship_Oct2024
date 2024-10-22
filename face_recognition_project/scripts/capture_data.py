import cv2
import os

# Name of the person to label the data (you can change this for each person)


person_name = input("Enter Person Name : ")  # Change this for each person you are capturing
save_dir = f'dataset/shorted/{person_name}'

# Create a directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for the built-in webcam or 1 for an external one

# Start capturing images
img_count = 0
while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break

    # Display the frame
    cv2.imshow("Webcam", frame)

    # Save every 10th frame to the dataset
    if img_count % 10 == 0:
        img_path = os.path.join(save_dir, f'{person_name}_{img_count}.jpg')
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")

    img_count += 1

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
