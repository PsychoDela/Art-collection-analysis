import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def color_analysis(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Flatten the image to make it easier to calculate statistics
    pixels = image_hsv.reshape(-1, 3)

    # Calculate average brightness and saturation
    brightness = np.mean(pixels[:, 2])
    saturation = np.mean(pixels[:, 1])

    return brightness, saturation


def analyze_images_in_years(artist_directory):
    brightness_data = []
    saturation_data = []

    for year_folder in os.listdir(artist_directory):
        year_path = os.path.join(artist_directory, year_folder)
        if os.path.isdir(year_path):
            print(f"Analyzing images in {year_folder}:")
            for filename in os.listdir(year_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(year_path, filename)
                    brightness, saturation = color_analysis(image_path)
                    brightness_data.append(brightness)
                    saturation_data.append(saturation)

    return brightness_data, saturation_data


artist_directory = '../img/claude-monet'  # Replace with the directory containing artist's folders
brightness_data, saturation_data = analyze_images_in_years(artist_directory)

# Plot histograms
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(brightness_data, bins=40, color='blue', alpha=0.7)
plt.title('Average Brightness Histogram')
plt.xlabel('Average Brightness')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(saturation_data, bins=40, color='green', alpha=0.7)
plt.title('Average Saturation Histogram')
plt.xlabel('Average Saturation')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
