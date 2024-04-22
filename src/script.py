import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def color_analysis(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate average brightness and saturation
    brightness = np.mean(image_hsv[:, :, 2])
    saturation = np.mean(image_hsv[:, :, 1])

    # Calculate histograms for RGB channels
    hist_rgb = []
    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        hist_rgb.append(hist)

    # Normalize histograms
    hist_rgb = [hist / np.sum(hist) for hist in hist_rgb]

    return brightness, saturation, hist_rgb


def analyze_images_in_years(artist_directory):
    brightness_data = []
    saturation_data = []
    hist_rgb_data = []

    for year_folder in os.listdir(artist_directory):
        year_path = os.path.join(artist_directory, year_folder)
        if os.path.isdir(year_path):
            print(f"Analyzing images in {year_folder}:")
            for filename in os.listdir(year_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(year_path, filename)
                    brightness, saturation, hist_rgb = color_analysis(image_path)
                    brightness_data.append(brightness)
                    saturation_data.append(saturation)
                    hist_rgb_data.append(hist_rgb)

    return brightness_data, saturation_data, hist_rgb_data

# Replace this directory with the path to the folder containing artist's subfolders
artist_directory = '../img/claude-monet'

brightness_data, saturation_data, hist_rgb_data = analyze_images_in_years(artist_directory)

# Plot histograms for brightness and saturation
plt.figure(figsize=(18, 6))

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

# Plot histograms for RGB
plt.figure(figsize=(18, 12))

for i in range(3):
    plt.subplot(2, 3, i+1)
    plt.plot(hist_rgb_data[0][i], color=['red', 'green', 'blue'][i], alpha=0.7)  # Plot normalized histogram
    plt.title(f'RGB Channel {i+1} Histogram')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


