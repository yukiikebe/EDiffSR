import os
from PIL import Image
from tqdm import tqdm

def downsample_images(input_dir, output_dir, scale=0.25):
    """
    Downsample images in each directory to a specified scale.

    Parameters:
        input_dir (str): Path to the dataset's root directory.
        output_dir (str): Path to save the downsampled images.
        scale (float): Scaling factor for the images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through each category in the dataset
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        if not os.path.isdir(category_path):
            continue  # Skip non-directory files

        output_category_path = os.path.join(output_dir, category)
        os.makedirs(output_category_path, exist_ok=True)

        # Process each image in the category
        for img_name in tqdm(os.listdir(category_path), desc=f"Processing {category}"):
            img_path = os.path.join(category_path, img_name)
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue  # Skip non-image files

            try:
                with Image.open(img_path) as img:
                    # Downsample the image
                    new_size = (int(img.width * scale), int(img.height * scale))
                    img_resized = img.resize(new_size, Image.ANTIALIAS)

                    # Save the downsampled image
                    output_img_path = os.path.join(output_category_path, img_name)
                    img_resized.save(output_img_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

# Define input and output directories
input_dir = "./AID_sample/HR"
output_dir = "./AID_sample/LR"

# Downsample images to 1/4 of their original size
downsample_images(input_dir, output_dir, scale=0.25)
