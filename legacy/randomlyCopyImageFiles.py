import os
import random
import shutil

# Define paths
source_folder = "PCB-cropped/all"  # Path to the folder containing 10,000 images
destination_folder = "PCB-cropped-combined"  # Path to the folder where 200 images will be copied

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Get a list of all image files in the source folder
image_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

# Check if the source folder contains enough images
if len(image_files) < 5000:
    print("Not enough images in the source folder to select 200.")
else:
    # Randomly select 5000 image files
    selected_images = random.sample(image_files, 5000)

    # Copy selected images to the destination folder
    for image in selected_images:
        src_path = os.path.join(source_folder, image)
        dest_path = os.path.join(destination_folder, image)
        shutil.copy(src_path, dest_path)

    print(f"Successfully copied 5000 images to {destination_folder}")
