import cv2
import os

"""
def detect_faces(image_path):
    # Load the pre-trained Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the image
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=6, minSize=(30, 30))

    # Return the number of detected faces and their bounding boxes
    return len(faces), faces


def analyze_images_in_years(artist_directory):
    total_faces = 0

    for year_folder in os.listdir(artist_directory):
        year_path = os.path.join(artist_directory, year_folder)
        if os.path.isdir(year_path):
            print(f"Analyzing images in {year_folder}:")
            for filename in os.listdir(year_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(year_path, filename)
                    num_faces, _ = detect_faces(image_path)
                    total_faces += num_faces
                    print(f"   {filename}: {num_faces} faces detected")

    return total_faces


# Replace this directory with the path to the folder containing artist's subfolders
artist_directory = '../img/edgar-degas'

# Analyze images in the specified directory
total_faces_detected = analyze_images_in_years(artist_directory)

# Print the total number of faces detected
print("\nTotal number of faces detected in all images:", total_faces_detected)
"""

def detect_faces(image_path):
    # Load the pre-trained Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the image
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=6, minSize=(30, 30))

    # Draw rectangles around the detected faces and save the image
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save the image with detected faces
    cv2.imwrite('detected_faces4.jpg', image)

    # Return the number of detected faces and their bounding boxes
    return len(faces), faces


# Replace 'image_path' with the path to the image you want to analyze
image_path = '../img/edgar-degas/1867/213605.jpg'

# Detect faces in the image
num_faces, faces = detect_faces(image_path)

# Print the number of detected faces
print("Number of faces detected:", num_faces)

