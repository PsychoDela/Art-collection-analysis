import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

#for each image calculate histogram for each color channel
#normalize
#calculate entropy
#combine channel entropies

#higher entropy-> more diversity in colors

def calculate_entropy(image):
    entropies = []
    # Calculate entropy for each color channel
    for channel in range(image.shape[2]):
        # Calculate histogram for the channel
        hist = cv2.calcHist([image], [channel], None, [256], [0, 256])
        
        # Normalize histogram
        hist_norm = hist.ravel() / hist.sum()
        
        # Calculate entropy for the channel
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))  # Adding small value to avoid log(0)
        
        # Add entropy to the list
        entropies.append(entropy)
    
    # Combine entropies from all channels (you can average or sum them)
    combined_entropy = np.mean(entropies)  # You can change np.mean to np.sum if you want to sum them
    
    max_entropy = np.log2(image.size)  # Maximum entropy for the image size
    normalized_entropy = combined_entropy / max_entropy
    
    return normalized_entropy

def process_folder(files):
    total_entropy = 0
    num_images = 0
    entropy_data = []
    # Iterate over each image in the folder
    for file in files:
        image = cv2.imread(file)
        
        # Calculate entropy
        entropy = calculate_entropy(image)
        
        # Accumulate total entropy
        total_entropy += entropy
        num_images += 1

        entropy_data.append((file, entropy))
    
    average_entropy = total_entropy / num_images
    print(f"Average Entropy across {num_images} images: {average_entropy}")

    entropy_data.sort(key=lambda x: x[1])
    
    # Display images with most and least entropy
    most_entropy_image_path, most_entropy = entropy_data[-1]
    least_entropy_image_path, least_entropy = entropy_data[0]
    
    # Load images for display
    most_entropy_image = cv2.imread(most_entropy_image_path)
    least_entropy_image = cv2.imread(least_entropy_image_path)
    
    # Plot images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(least_entropy_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Least Entropy: {least_entropy:.2f}")
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(most_entropy_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Most Entropy: {most_entropy:.2f}")
    axes[1].axis('off')
    plt.show()

def unpack_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


folder_path = '../img/edgar-degas'
files=unpack_files(folder_path)
process_folder(files)

folder_path2 = '../img/claude-monet'
files2=unpack_files(folder_path2)
process_folder(files2)




