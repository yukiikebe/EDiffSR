import os
import random
import shutil
from tqdm import tqdm

def split_dataset(input_dir, output_test_dir, num_images):
    """
    Split a dataset by randomly selecting a fixed number of files from each category.

    Parameters:
        input_dir (str): Path to the dataset's root directory.
        output_test_dir (str): Path to save the test dataset.
        num_images (int): Number of files to use for the test set from each category.
    """
    if not os.path.exists(output_test_dir):
        os.makedirs(output_test_dir)

    # Iterate through each category in the dataset
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        if not os.path.isdir(category_path):
            continue  # Skip non-directory files

        output_category_test_path = os.path.join(output_test_dir, category)
        os.makedirs(output_category_test_path, exist_ok=True)

        # Get all image files in the category
        image_files = [
            f for f in os.listdir(category_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        # Ensure the number of selected files does not exceed available files
        num_test_files = min(num_images, len(image_files))
        test_files = random.sample(image_files, num_test_files)

        # Move the selected files to the test directory
        for img_name in tqdm(test_files, desc=f"Processing {category}"):
            src_path = os.path.join(category_path, img_name)
            dest_path = os.path.join(output_category_test_path, img_name)
            shutil.move(src_path, dest_path)

# Define input and output directories
input_dir = "./AID/data"
output_test_dir = "./AID_sample/HR"

# Specify the number of images to select for the test set
num_images = 1  # Example: 50 images per category

# Split the dataset by selecting a fixed number of images
split_dataset(input_dir, output_test_dir, num_images)
