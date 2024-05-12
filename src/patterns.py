import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# Function to calculate brightness standard deviation from an image
def calculate_brightness_std(image_path):
    img = Image.open(image_path)
    img = img.convert('L')  # Convert to grayscale
    brightness_std = np.std(np.array(img))
    return brightness_std


# Function to calculate brightness standard deviation for each year
def calculate_brightness_std_by_year(dataset_path):
    years = []
    brightness_std_values = []

    for year_folder in sorted(os.listdir(dataset_path)):
        year_path = os.path.join(dataset_path, year_folder)
        if os.path.isdir(year_path):
            years.append(year_folder)
            brightness_std_values_year = []
            for image_file in os.listdir(year_path):
                image_path = os.path.join(year_path, image_file)
                brightness_std = calculate_brightness_std(image_path)
                brightness_std_values_year.append(brightness_std)
            brightness_std_year = np.mean(brightness_std_values_year)
            brightness_std_values.append(brightness_std_year)

    return years, brightness_std_values


# Function to plot the graph
def plot_style_changes(years, brightness_std_values):
    plt.figure(figsize=(12, 6))  # Set the size of the plot
    plt.plot(years, brightness_std_values, marker='o', linestyle='-')
    plt.xlabel('Year')
    plt.ylabel('Brightness Standard Deviation')
    plt.title('Style Evolution: Brightness Standard Deviation over Time')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Path to your dataset folder
dataset_path = '../img/edgar-degas'

# Calculate brightness standard deviation for each year
years, brightness_std_values = calculate_brightness_std_by_year(dataset_path)

# Plot the graph
plot_style_changes(years, brightness_std_values)
