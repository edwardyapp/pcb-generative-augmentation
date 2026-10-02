import os
import random
from PIL import Image

# Paths
input_folder = "PCB/all"  # Folder containing 10 high-resolution images
output_folder = "/home/edward-yapp/PycharmProjects/vq-vae-2-pytorch-ad/PCB-good-cropped-256 (Copy)/chip01/train/good"  # Folder to save augmented images
# output_folder = "/home/edward-yapp/PycharmProjects/vq-vae-2-pytorch-ad/PCB-good-cropped-256/chip01/test/good"  # Folder to save augmented images
# output_folder = "/home/edward-yapp/PycharmProjects/vq-vae-2-pytorch-ad/PCB-good-cropped-512/pcb/train/good"  # Folder to save augmented images
# output_folder = "/home/edward-yapp/PycharmProjects/vq-vae-2-pytorch-ad/PCB-good-cropped-512/pcb/test/good"  # Folder to save augmented images
# output_folder = "/home/edward-yapp/PycharmProjects/vq-vae-2-pytorch-ad/PCB-good-cropped-1024/pcb/train/good"  # Folder to save augmented images
# output_folder = "/home/edward-yapp/PycharmProjects/vq-vae-2-pytorch-ad/PCB-good-cropped-1024/pcb/test/good"  # Folder to save augmented images
os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

# Parameters
crop_size = 256  # Size of the crop
rotations = [0, 90, 180, 270]  # Possible rotation angles
augmentations_per_image = 200  # Number of augmentations per image

# Get all image files in the input folder
image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
if len(image_files) != 10:
    raise ValueError(f"The input folder must contain exactly 10 images. Found: {len(image_files)}")

# Ask user which images to process
print("Available images:")
for i, image in enumerate(image_files):
    print(f"{i + 1}. {image}")

choice = input("Enter the numbers of the images to process (comma-separated, or 'all' to process all images): ").strip()
if choice.lower() == 'all':
    selected_images = image_files
else:
    selected_indices = [int(x) - 1 for x in choice.split(',') if x.isdigit() and 0 < int(x) <= len(image_files)]
    selected_images = [image_files[i] for i in selected_indices]

if not selected_images:
    raise ValueError("No valid images selected.")

# Augment selected images
for idx, image_file in enumerate(selected_images):
    image_path = os.path.join(input_folder, image_file)

    try:
        # Open the image
        img = Image.open(image_path).convert("RGB")  # Ensure RGB format
        width, height = img.size

        # Check if the image is large enough for cropping
        if width < crop_size or height < crop_size:
            raise ValueError(f"Image {image_file} is smaller than the crop size ({crop_size}x{crop_size}).")

        # Perform augmentations
        for augmentation_idx in range(augmentations_per_image):
            # Randomly select the top-left corner for the crop within safe bounds
            left = random.randint(0, width - crop_size)
            top = random.randint(0, height - crop_size)
            right = left + crop_size
            bottom = top + crop_size

            # Perform the crop
            cropped_img = img.crop((left, top, right, bottom))

            # Randomly rotate the image
            rotation_angle = random.choice(rotations)
            rotated_img = cropped_img.rotate(rotation_angle)

            # Randomly flip the image horizontally or vertically
            if random.choice([True, False]):
                rotated_img = rotated_img.transpose(Image.FLIP_LEFT_RIGHT)
            if random.choice([True, False]):
                rotated_img = rotated_img.transpose(Image.FLIP_TOP_BOTTOM)

            # Save the augmented image with a unique name
            output_path = os.path.join(
                output_folder,
                f"{os.path.splitext(image_file)[0]}_{augmentation_idx:04d}_{rotation_angle}.jpg"
            )
            rotated_img.save(output_path)

    except Exception as e:
        print(f"Error processing {image_file}: {e}")

# Output the results
print(f"Successfully generated {augmentations_per_image * len(selected_images)} images in {output_folder}.")
