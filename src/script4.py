import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the path to the directory containing your images
image_dir = "../img/edgar-degas"


# Function to calculate average channel intensity for an image
def calculate_average_intensity(image):
    return np.mean(image, axis=(0, 1))


# Lists to store average intensities and corresponding years
avg_intensities_r = []
avg_intensities_g = []
avg_intensities_b = []
years = []

# Iterate through each folder (year) in the image directory
for year_folder in os.listdir(image_dir):
    year_path = os.path.join(image_dir, year_folder)
    if os.path.isdir(year_path):
        # Collect all image files within the year folder
        image_files = [f for f in os.listdir(year_path) if os.path.isfile(os.path.join(year_path, f))]
        if image_files:
            # Load the first image to get dimensions
            sample_image = cv2.imread(os.path.join(year_path, image_files[0]))
            for image_file in image_files:
                # Load each image
                image = cv2.imread(os.path.join(year_path, image_file))
                # Calculate average channel intensities
                avg_intensity = calculate_average_intensity(image)
                # Append average intensities and corresponding year
                avg_intensities_r.append(avg_intensity[0])
                avg_intensities_g.append(avg_intensity[1])
                avg_intensities_b.append(avg_intensity[2])
                years.append(int(year_folder))

# Plotting
plt.figure(figsize=(10, 6))

# Red channel
plt.subplot(3, 1, 1)
plt.scatter(years, avg_intensities_r, color='red')
plt.title('Red Channel Intensity Over Years')
plt.xlabel('Year')
plt.ylabel('Average Intensity')

# Green channel
plt.subplot(3, 1, 2)
plt.scatter(years, avg_intensities_g, color='green')
plt.title('Green Channel Intensity Over Years')
plt.xlabel('Year')
plt.ylabel('Average Intensity')

# Blue channel
plt.subplot(3, 1, 3)
plt.scatter(years, avg_intensities_b, color='blue')
plt.title('Blue Channel Intensity Over Years')
plt.xlabel('Year')
plt.ylabel('Average Intensity')

# Adjust layout to prevent overlapping
plt.tight_layout()

# Save the plot to a file
plt.savefig('channel_intensity_over_years-degas.png')

# Show the plot
plt.show()
