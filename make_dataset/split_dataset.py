import os
import random
import shutil
from tqdm import tqdm

def split_dataset_by_matching_files(root_dir, output_dir, train_size, val_size, test_size):
    """
    Split datasets into train, validation, and test sets while ensuring the same files
    are selected across corresponding subdirectories (e.g., HR, LR, GT), handling prefixes like "patch_".

    Parameters:
        root_dir (str): Path to the dataset's root directory containing subdirectories like 'GT', 'HR', and 'LR'.
        output_dir (str): Path to save the split datasets.
        train_size (int): Number of images per category for the train set.
        val_size (int): Number of images per category for the validation set.
        test_size (int): Number of images per category for the test set.
    """
    total_required = train_size + val_size + test_size

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each category in the dataset
    # for category in os.listdir(os.path.join(root_dir, "GT")):  # Assume 'GT' contains the full category list
    for category in os.listdir(os.path.join(root_dir, "HR")):  # Assume 'HR' contains the full category list
        print(f"Processing category: {category}")

        # Collect matching files across HR, LR, and GT
        matching_files = set()  # Set of matching standardized names (without prefix)
        # file_mapping = {"HR": {}, "LR": {}, "GT": {}}  # Map standardized names to full paths
        file_mapping = {"HR": {}, "LR": {}}  # Map standardized names to full paths

        # for sub_dir in ["HR", "LR", "GT"]:
        for sub_dir in ["HR", "LR"]:
            category_path = os.path.join(root_dir, sub_dir, category)
            if not os.path.isdir(category_path):
                raise ValueError(f"Category {category} not found in subdirectory {sub_dir}. Check your directory structure.")

            # Process files in the category
            files = os.listdir(category_path)
            for file_name in files:
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Standardize file name (remove "patch_" prefix if present)
                    standardized_name = file_name[len("patch_"):] if file_name.startswith("patch_") else file_name
                    file_mapping[sub_dir][standardized_name] = os.path.join(category_path, file_name)

            # Add standardized names to matching set
            if not matching_files:
                matching_files = set(file_mapping[sub_dir].keys())
            else:
                matching_files = matching_files.intersection(file_mapping[sub_dir].keys())

        # Check if there are enough matching files
        matching_files = list(matching_files)
        if len(matching_files) < total_required:
            raise ValueError(f"Not enough matching files in category '{category}' (found {len(matching_files)}, needed {total_required}).")

        # Shuffle the matching files
        random.shuffle(matching_files)

        # Split files into train, validation, and test sets
        train_files = matching_files[:train_size]
        val_files = matching_files[train_size:train_size + val_size]
        test_files = matching_files[train_size + val_size:train_size + val_size + test_size]

        # Copy files into train, val, and test directories
        # for sub_dir in ["HR", "LR", "GT"]:
        for sub_dir in ["HR", "LR"]:
            for subset, subset_files in zip(
                ["train", "val", "test"],
                [train_files, val_files, test_files]
            ):
                subset_dir = os.path.join(output_dir, sub_dir, subset, category)
                os.makedirs(subset_dir, exist_ok=True)

                for standardized_name in tqdm(subset_files, desc=f"Copying {category} - {subset} ({sub_dir})"):
                    src_path = file_mapping[sub_dir][standardized_name]
                    dest_path = os.path.join(subset_dir, os.path.basename(src_path))
                    shutil.copy(src_path, dest_path)

def split_dataset_by_matching_files_ratio(root_dir, output_dir, train_ratio, val_ratio, test_ratio):
    """
    Split datasets into train, validation, and test sets by ratio, ensuring the same files
    are selected across corresponding subdirectories (e.g., HR, LR).

    Parameters:
        root_dir (str): Path to the dataset's root directory containing subdirectories like 'HR' and 'LR'.
        output_dir (str): Path to save the split datasets.
        train_ratio (float): Ratio of the train set (e.g., 0.8 for 80%).
        val_ratio (float): Ratio of the validation set (e.g., 0.1 for 10%).
        test_ratio (float): Ratio of the test set (e.g., 0.1 for 10%).
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not abs(total_ratio - 1.0) < 1e-6:
        raise ValueError("Ratios must sum to 1.0.")
    
    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each category in the dataset
    for category in os.listdir(os.path.join(root_dir, "HR")):
        print(f"Processing category: {category}")
        
        # Collect matching files across HR and LR
        matching_files = set()  # Set of matching standardized names (without prefix)
        file_mapping = {"HR": {}, "LR": {}}
        
        for sub_dir in ["HR", "LR"]:
            category_path = os.path.join(root_dir, sub_dir, category)
            if not os.path.isdir(category_path):
                raise ValueError(f"Category {category} not found in subdirectory {sub_dir}. Check your directory structure.")
            
            # Process files in the category
            files = os.listdir(category_path)
            for file_name in files:
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                    # Standardize file name (remove "patch_" prefix if present)
                    standardized_name = file_name
                    file_mapping[sub_dir][standardized_name] = os.path.join(category_path, file_name)
            
            # Add standardized names to matching set
            if not matching_files:
                matching_files = set(file_mapping[sub_dir].keys())
            else:
                matching_files = matching_files.intersection(file_mapping[sub_dir].keys())
            
        # Check if there are enough matching files
        matching_files = list(matching_files)
        total_files = len(matching_files)
        train_size = int(train_ratio * total_files)
        val_size = int(val_ratio * total_files)
        test_size = total_files - train_size - val_size
        
        if total_files < train_size + val_size + test_size:
            raise ValueError(f"Not enough matching files in category '{category}' (found {total_files}, needed {train_size + val_size + test_size}).")
        
        # Shuffle the matching files
        random.shuffle(matching_files)
        
        # Split files into train, validation, and test sets
        train_files = matching_files[:train_size]
        val_files = matching_files[train_size:train_size + val_size]
        test_files = matching_files[train_size + val_size:]
        
        # Copy files into train, val, and test directories
        for sub_dir in ["HR", "LR"]:
            for subset, subset_files in zip(
                ["train", "val", "test"],
                [train_files, val_files, test_files]
            ):
                subset_dir = os.path.join(output_dir, sub_dir, subset, category)
                os.makedirs(subset_dir, exist_ok=True)
                
                for standardized_name in tqdm(subset_files, desc=f"Copying {category} - {subset} ({sub_dir})"):
                    src_path = file_mapping[sub_dir][standardized_name]
                    dest_path = os.path.join(subset_dir, os.path.basename(src_path))
                    shutil.copy(src_path, dest_path)
                    
    
# Define input and output directories
input_dir = "../Orthophotos_patches_png_scale16"
output_dir = "../Orthophotos_patches_png_scale16_split_ratio"

# Specify the number of images per category for each set
# train_size = 100
# val_size = 10
# test_size = 10

# # Split the dataset
# split_dataset_by_matching_files(input_dir, output_dir, train_size, val_size, test_size)

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
split_dataset_by_matching_files_ratio(input_dir, output_dir, train_ratio, val_ratio, test_ratio)
