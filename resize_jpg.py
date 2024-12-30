import os
from PIL import Image

def resize_images_in_dir(input_dir, output_dir, size=(640, 640)):
    """
    Resize all JPG images in a directory and save them to a new directory.

    :param input_dir: Path to the directory containing input images
    :param output_dir: Path to the directory to save resized images
    :param size: Tuple (width, height) for the new size
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Loop through all files in the input directory
        for file_name in os.listdir(input_dir):
            input_path = os.path.join(input_dir, file_name)

            # Process only JPG images
            if file_name.lower().endswith(('.jpg', '.jpeg')):
                try:
                    with Image.open(input_path) as img:
                        # Resize the image
                        resized_img = img.resize(size, Image.ANTIALIAS)

                        # Save the resized image to the output directory
                        output_path = os.path.join(output_dir, file_name)
                        resized_img.save(output_path, format='JPEG')
                        print(f"Resized: {file_name} -> {output_path}")
                except Exception as img_error:
                    print(f"Failed to process {file_name}: {img_error}")
    except Exception as e:
        print(f"Error resizing images in directory: {e}")

# Example usage
input_directory = "Sentinel/GT/Fire"  # Replace with your input directory path
output_directory = "resize_Sentinel/GT/Fire"  # Replace with your output directory path
resize_images_in_dir(input_directory, output_directory)
