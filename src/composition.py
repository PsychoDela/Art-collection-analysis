import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def analyze_texture(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Canny edge detection
    edges = cv2.Canny(image, 100, 200)

    # Compute the histogram of edge orientations
    hist, _ = np.histogram(edges, bins=np.arange(0, 361, 10))

    return hist

def analyze_images(artist_directory):
    texture_histograms = []

    for year_folder in os.listdir(artist_directory):
        year_path = os.path.join(artist_directory, year_folder)
        if os.path.isdir(year_path):
            print(f"Analyzing images in {year_folder}:")
            for filename in os.listdir(year_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(year_path, filename)
                    hist = analyze_texture(image_path)
                    texture_histograms.append(hist)

    # Calculate the average histogram
    avg_hist = np.mean(texture_histograms, axis=0)

    # Plot the average histogram
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0, 360, 10), avg_hist, color='blue')
    plt.title('Average Edge Orientation Histogram')
    plt.xlabel('Edge Orientation (degrees)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


analyze_images('../img/claude-monet')
analyze_images('../img/edgar-degas')

