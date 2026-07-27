import os
import random
from PIL import Image
from torchvision import transforms

# Paths
input_folder = "/home/edward-yapp/Downloads/transistor/train/good"  # Folder with original images
output_folder = "PCB-cropped-combined"  # Folder to save augmented images
target_image_count = 5000  # Total augmented images needed

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Define augmentations
augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
    transforms.RandomVerticalFlip(p=0.5),  # Random vertical flip
    transforms.RandomRotation(degrees=(-2, 2)),  # Random rotation in a small range
    transforms.Resize(292),  # Resize to 292
    transforms.RandomCrop(282),  # Crop to 282
    transforms.CenterCrop(256),  # Center crop to 256
])

# Get all image files in the input folder
image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
if len(image_files) == 0:
    raise ValueError("No images found in the input folder.")

# Initialize counter
current_count = 0

# Perform augmentations until the target count is reached
while current_count < target_image_count:
    # Randomly select an original image
    image_file = random.choice(image_files)
    src_path = os.path.join(input_folder, image_file)

    # Load the image and ensure it's in RGB format
    img = Image.open(src_path).convert("RGB")

    # Apply augmentations
    augmented_img = augmentations(img)

    # Save the augmented image
    output_path = os.path.join(output_folder, f"{current_count:05d}_aug_{image_file}")
    augmented_img.save(output_path)
    current_count += 1

print(f"Successfully generated {target_image_count} augmented images in {output_folder}.")
