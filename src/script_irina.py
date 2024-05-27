import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image


def brightness_mean(files):
    #returns list of brightness means (for each picture)
    bm=[]
    for file in files:
        img = cv2.imread(file)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness_mean = img_gray.mean()
        bm.append(brightness_mean)

    return bm

def saturation_mean(files):
    sm=[]
    for file in files:
        img = cv2.imread(file)
        
        # Convert the image from BGR to HSV
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Extract the saturation channel
        saturation = img_hsv[:,:,1]
        saturation_mean = np.mean(saturation)
        sm.append(saturation_mean)
        
    return sm

def brightness_median(files):
    #returns list of brightness means (for each picture)
    bm=[]
    for file in files:
        img = cv2.imread(file)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness_median = np.median(img_gray)
        bm.append(brightness_median)

    return bm

def saturation_median(files):
    sm=[]
    for file in files:
        img = cv2.imread(file)
        
        # Convert the image from BGR to HSV
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Extract the saturation channel
        saturation = img_hsv[:,:,1]
        saturation_median = np.median(saturation)
        sm.append(saturation_median)
        
    return sm

def brightness_standard_deviation(files):
    stddev=[]
    for file in files:
        img = cv2.imread(file)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Compute the brightness (pixel intensity) for each pixel
        brightness_values = img_gray.flatten()
        
        # Calculate the standard deviation of brightness values
        brightness_std = np.std(brightness_values)
        stddev.append(brightness_std)
    
    return stddev

def entropy(files):
    entropies=[]
    for file in files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        
        # Compute histogram of pixel intensities
        hist = cv2.calcHist([img],[0],None,[256],[0,256])
        
        # Normalize histogram to get probability distribution
        hist_norm = hist.ravel() / hist.sum()
        
        # Compute entropy
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-8))  # Add a small value to prevent log(0)

        entropies.append(entropy)

    return entropies


"""
X-axis: brightness mean
Y-axis: saturation mean

For Degas and Monet paintings

Possible features: average hue((a median average of colors
of every pixel represented on 0-255 scale)
"""

"""
X-axis: standard deviation
Y-axis: entropy

both measured using pixels brightness values

plot using points to show density
plot both features in 1D histograms
"""

"""
X-axis = brightness median.
Y-axis = saturation median.
"""

"""
VISUALIZING AN IMAGE SET IN RELATION TO A SPACE OF ALL POSSIBLE IMAGES
Footprint of image set in relation to all possible images
X-axis = brightness mean. Min = 0; Max = 255.
Y-axis = brightness standard deviation. Min = 0; Max = 126.7.
"""

"""
VISUALIZING PARTS OF AN IMAGE SET IN RELATION TO THE WHOLE SET
X-axis = brightness mean;
Y-aixs = brightness standard deviation:
"""

"""
X-axis: years
Y-axis: brightness meadian/saturation median
"""

def plot(files, X, Y):
    #list of files, list of x axis, list of y axis
    fig, ax = plt.subplots()
    ax.scatter(X, Y) 
    
    for x0, y0, path in zip(X, Y,files):
        try:
            # Resize the image to reduce memory usage
            with Image.open(path) as img:
                img.thumbnail((200, 200))  # Adjust the size as needed
                img_array = np.array(img)

            # Convert the image to OffsetImage and add to plot
            ab = AnnotationBbox(OffsetImage(img_array, zoom=0.1), (x0, y0), frameon=False)
            ax.add_artist(ab)
        except Exception as e:
            print(f"Error loading image {path}: {e}")

    #plt.savefig('../graphs/monet/brightness_saturation_mean.svg', format='svg', bbox_inches='tight')
    
    plt.show()

def plot_points(X,Y):
    plt.style.use('dark_background')
    plt.scatter(X, Y, color='white', alpha=0.6)
    plt.title('Monet Density')
    plt.show()

def plot_series(files,X,Y):
    for file_path, x, y in zip(files, X, Y):
        # Extract the year from the file name
        year_str = file_path.split('/')[3]
        
        if year_str == 'portreti':
            plt.scatter(x, y, color="red", label=year_str)  # Use specific color for the specific year
        else:
            plt.scatter(x, y, color='blue', alpha=0.5)  # Use a default color for other years

    # Show the plot
    plt.show()

def plot_time(files,Y):
    
    years = []
    for file_path in files:
        year_str = file_path.split('/')[3]
        try:
            year = int(year_str)
            years.append(year)
        except ValueError:
            print(f"Warning: Unable to extract year from file path '{file_path}'")
            continue
    
    fig, ax = plt.subplots()
    ax.scatter(years, Y) 
    
    for x0, y0, path in zip(years, Y,files):
        try:
            # Resize the image to reduce memory usage
            with Image.open(path) as img:
                img.thumbnail((200, 200))  # Adjust the size as needed
                img_array = np.array(img)

            # Convert the image to OffsetImage and add to plot
            ab = AnnotationBbox(OffsetImage(img_array, zoom=0.1), (x0, y0), frameon=False)
            ax.add_artist(ab)
        except Exception as e:
            print(f"Error loading image {path}: {e}")

    plt.show()
    """
    plt.style.use('dark_background')
    plt.scatter(years, Y, color='white', alpha=0.6)
    plt.title('Monet through time')
    plt.show()
    """

def unpack_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


folder_path = '../img/claude-monet'
files=unpack_files(folder_path)
#print(files)

"""
bm=brightness_mean(files)
sm=saturation_mean(files)

stddev=brightness_standard_deviation(files)
ent=entropy(files)

with open('monet-brightness-mean.txt', 'r') as file:
    bm = [float(line.strip()) for line in file]

with open('monet-saturation-mean.txt', 'r') as file:
    sm = [float(line.strip()) for line in file]
    """
bm=brightness_median(files)
#sm=saturation_median(files)
#plot(files,bm,sm)
#plot(files,stddev,ent)
#plot_time(files,bm)
#plot_points(stddev,ent)

