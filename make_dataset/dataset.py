import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
from random import randrange
from tqdm import tqdm
import rasterio
import cv2
from rasterio.enums import Resampling

def adjust_brightness(image, alpha=1.0, beta=10):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def add_blur(image, ksize=(3, 3)): # ksize shoould be odd number
    return cv2.GaussianBlur(image, ksize, 0)

def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    noisy_image = np.copy(image)
    total_pixels = image.shape[0] * image.shape[1]
    
    # Salt noise
    num_salt = int(total_pixels * salt_prob)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy_image[salt_coords[0], salt_coords[1]] = 255
    
    # Pepper noise
    num_pepper = int(total_pixels * pepper_prob)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0
    
    return noisy_image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    # y, _, _ = img.split()
    return img


def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

def rescale_img_tiff(image, scale):
    """Rescale multi-band image while preserving range and dtype."""
    num_bands = image.shape[-1]
    resized = []
    
    for band in range(num_bands):
        resized[:, :, band] = resize(image[:, :, band], (resized.shape[0], resized.shape[1]), anti_aliasing=True, preserve_range=True)
    
    return resized

def get_patch(img_in, img_tar, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    #(th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))
    #img_bic = img_bic.crop((ty, tx, ty + tp, tx + tp))

    #info_patch = {
    #    'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar


def augment(img_in, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        #img_bic = ImageOps.flip(img_bic)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            #img_bic = ImageOps.mirror(img_bic)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            #img_bic = img_bic.rotate(180)
            info_aug['trans'] = True

    return img_in, img_tar, info_aug

def create_resized_patches(input_dir, hr_dir, lr_dir, patch_size=256, scale=4):
    """
    Create HR (256x256) and LR (56x56) patches from AID dataset.

    Args:
        input_dir (str): Path to original AID dataset.
        hr_dir (str): Output directory for HR patches.
        lr_dir (str): Output directory for LR patches.
        patch_size (int): Size of HR patches (default: 256).
        scale (int): Downscaling factor for LR patches (default: 4).
    """
    if not os.path.exists(hr_dir):
        os.makedirs(hr_dir)
    if not os.path.exists(lr_dir):
        os.makedirs(lr_dir)

    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        if not os.path.isdir(category_path):
            continue

        output_hr_category_path = os.path.join(hr_dir, category)
        os.makedirs(output_hr_category_path, exist_ok=True)

        output_lr_category_path = os.path.join(lr_dir, category)
        os.makedirs(output_lr_category_path, exist_ok=True)

        for file_name in tqdm(os.listdir(category_path), desc=f"Processing {category}"):
            img_path = os.path.join(category_path, file_name)
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Load the original image
                img = Image.open(img_path).convert('RGB')

                # Extract HR patch
                img_width, img_height = img.size
                if img_width < patch_size or img_height < patch_size:
                    continue  

                hr_patch, _ = get_patch(img, img, patch_size, scale)

                # Downscale HR patch to create LR patch
                lr_patch = rescale_img(hr_patch, scale=1/scale)

                # Save HR patch
                output_hr = os.path.join(output_hr_category_path, f"patch_{file_name}")
                hr_patch.save(output_hr, format='JPEG')

                # Save LR patch
                output_lr = os.path.join(output_lr_category_path, f"patch_{file_name}")
                lr_patch.save(output_lr, format='JPEG')

def create_resized_patches_from_tiff(input_dir, output_dir, option=2, patch_size=256, scale=4, overlap=0.5, black_threshold=50, max_black_ratio=0.5):
    """
    Create HR (256x256) and LR (56x56) patches from Orthophotos (Tiff images) dataset.
    It allows a 50% overlap between patches, but exclude patches if more than 50% of the patch is black.  

    Args:
        input_dir (str): Path to original Orthophotos (Tiff images) dataset.
        Option: 1 for tiff output, 2 for png output
        hr_dir (str): Output directory for HR patches.
        lr_dir (str): Output directory for LR patches.
        patch_size (int): Size of HR patches (default: 256).
        scale (int): Downscaling factor for LR patches (default: 4).
    """
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
        
    hr_dir = os.path.join(output_dir, "HR")
    if not os.path.exists(hr_dir):
        os.makedirs(hr_dir)
        os.makedirs(os.path.join(hr_dir, "Farmland"))
    lr_dir = os.path.join(output_dir, "LR")
    if not os.path.exists(lr_dir):
        os.makedirs(lr_dir)
        os.makedirs(os.path.join(lr_dir, "Farmland"))
    
    hr_dir_farmland = os.path.join(hr_dir, "Farmland")
    lr_dir_farmland = os.path.join(lr_dir, "Farmland")
    
    patch_count = 0
    
    for file_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, file_name)
        print(f"Processing {file_name}")
        if file_name.lower().endswith(('.tif', '.tiff')):
            # Load the original image
            try:
                with rasterio.open(img_path) as src:
                    img_array = src.read([1, 2, 3])
                    img_array = np.moveaxis(img_array, 0, -1)  # Move channels to the last dimension
        
                    # Normalize the data to 0-255 if needed
                    if img_array.dtype != np.uint8:
                        img_min = img_array.min()
                        img_max = img_array.max()
                        img_array = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            except Exception as e:
                error_msg = f"Error loading {file_name}: {e}"
                print(error_msg)
            
            print(f"Processing {file_name} with size {img_array.shape}")
            
            height, width, _ = img_array.shape
            step = int(patch_size * (1 - overlap))
            
            for y in range(0, height - patch_size + 1, step):
                for x in range(0, width - patch_size + 1, step):
                    # Extract the patch
                    patch = img_array[y:y + patch_size, x:x + patch_size, :]

                    # Calculate the black pixel ratio
                    black_pixels = np.sum(np.all(patch <= black_threshold, axis=-1))
                    total_pixels = patch_size * patch_size
                    black_ratio = black_pixels / total_pixels

                    # Exclude patches with more than 50% black background
                    if black_ratio <= max_black_ratio:
                        patch_filename = f"patch_{patch_count:04d}.tif"
                        patch_path = os.path.join(hr_dir_farmland, patch_filename)

                        # Save the patch using rasterio
                        if option == 1:
                            with rasterio.open(
                                patch_path,
                                'w',
                                driver='GTiff',
                                height=patch.shape[0],
                                width=patch.shape[1],
                                count=3,
                                dtype=patch.dtype
                            ) as dst:
                                for band in range(3):
                                    dst.write(patch[:, :, band], band + 1)
                                #LR patch
                                lr_patch = rescale_img(patch, scale=1/scale)
                                lr_patch_filename = f"{patch_count:04d}.tif"
                                lr_patch_path = os.path.join(lr_dir_farmland, lr_patch_filename)
                                with rasterio.open(
                                    lr_patch_path,
                                    'w',
                                    driver='GTiff',
                                    height=lr_patch.shape[0],
                                    width=lr_patch.shape[1],
                                    count=3,
                                    dtype=lr_patch.dtype
                                ) as dst:
                                    for band in range(3):
                                        dst.write(lr_patch[:, :, band], band + 1)
                        else:
                            png_path = os.path.join(hr_dir_farmland, f"{patch_count:04d}.png")
                            patch_img = Image.fromarray(patch.astype('uint8'), 'RGB')  # Convert to PIL image
                            patch_img.save(png_path)
                            noisy_patch = add_salt_and_pepper_noise(patch, salt_prob=0.005, pepper_prob=0.005)
                            noisy_patch_img = Image.fromarray(noisy_patch.astype('uint8'), 'RGB')
                            noisy_patch_img.save(os.path.join(hr_dir_farmland, f"{patch_count:04d}_noise.png"))
                            bright_patch = adjust_brightness(patch, alpha=1.1, beta=0)
                            bright_patch_img = Image.fromarray(bright_patch.astype('uint8'), 'RGB')
                            bright_patch_img.save(os.path.join(hr_dir_farmland, f"{patch_count:04d}_bright.png"))
                            
                            lr_patch = rescale_img(patch_img, scale=1/scale)
                            # Save LR patch
                            output_lr = os.path.join(lr_dir_farmland, f"{patch_count:04d}.png")
                            lr_patch.save(output_lr, format='PNG')
                            noisy_lr_patch = rescale_img(noisy_patch_img, scale=1/scale)
                            noisy_lr_patch.save(os.path.join(lr_dir_farmland, f"{patch_count:04d}_noise.png"), format='PNG')
                            bright_lr_patch = rescale_img(bright_patch_img, scale=1/scale)
                            bright_lr_patch.save(os.path.join(lr_dir_farmland, f"{patch_count:04d}_bright.png"), format='PNG')

                        patch_count += 1

    print(f"Extracted {patch_count} patches to {output_dir}")

def create_resized_tiff_patches_from_tiff(input_dir, output_dir, patch_size=256, scale=4, overlap=0.5, black_threshold=50, max_black_ratio=0.5):
    """
    Create HR (256x256) and LR (56x56) patches from Orthophotos (Tiff images) dataset.
    It allows a 50% overlap between patches, but exclude patches if more than 50% of the patch is black.  

    Args:
        input_dir (str): Path to original Orthophotos (Tiff images) dataset.
        Option: 1 for tiff output, 2 for png output
        hr_dir (str): Output directory for HR patches.
        lr_dir (str): Output directory for LR patches.
        patch_size (int): Size of HR patches (default: 256).
        scale (int): Downscaling factor for LR patches (default: 4).
    """
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
        
    hr_dir = os.path.join(output_dir, "HR")
    if not os.path.exists(hr_dir):
        os.makedirs(hr_dir)
        os.makedirs(os.path.join(hr_dir, "Farmland"))
    lr_dir = os.path.join(output_dir, "LR")
    if not os.path.exists(lr_dir):
        os.makedirs(lr_dir)
        os.makedirs(os.path.join(lr_dir, "Farmland"))
    
    hr_dir_farmland = os.path.join(hr_dir, "Farmland")
    lr_dir_farmland = os.path.join(lr_dir, "Farmland")
    
    patch_count = 0
    
    for file_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, file_name)
        # print(f"Processing {file_name}")
        if file_name.lower().endswith(('.tif', '.tiff')):
            # Load the original image
            try:
                with rasterio.open(img_path) as src:
                    img_array = src.read()
                    img_array = np.moveaxis(img_array, 0, -1)
                    num_bands = img_array.shape[-1]
                    
                    profile = src.profile  # Copy metadata
                    profile.update(
                        driver="GTiff",
                        height=patch_size,
                        width=patch_size,
                        count=num_bands,
                        dtype=img_array.dtype
                    )
        
            except Exception as e:
                error_msg = f"Error loading {file_name}: {e}"
                print(error_msg)
            
            print(f"Processing {file_name} with size {img_array.shape}")
            
            height, width, _ = img_array.shape
            step = int(patch_size * (1 - overlap))
            
            for y in range(0, height - patch_size + 1, step):
                for x in range(0, width - patch_size + 1, step):
                    # Extract the patch
                    patch = img_array[y:y + patch_size, x:x + patch_size, :]

                    # Calculate the black pixel ratio
                    black_pixels = np.sum(np.all(patch <= black_threshold, axis=-1))
                    total_pixels = patch_size * patch_size
                    black_ratio = black_pixels / total_pixels

                    # Exclude patches with more than 50% black background
                    if black_ratio <= max_black_ratio:
                        patch_filename = f"{patch_count:04d}.tif"
                        patch_path = os.path.join(hr_dir_farmland, patch_filename)

                        # Save the patch using rasterio
                        with rasterio.open(
                            patch_path,
                            'w',
                            ** profile
                        ) as dst:
                            for band in range(num_bands):
                                dst.write(patch[:, :, band], band + 1)
                        
                        with rasterio.open(
                            patch_path,
                            'r',
                            ** profile
                        ) as dst:
                            #LR patch
                            lr_patch_filename = f"{patch_count:04d}.tif"
                            lr_patch_path = os.path.join(lr_dir_farmland, lr_patch_filename)
                            #rescale to make LR patch
                            new_height = int(dst.height / scale)
                            new_width = int(dst.width / scale)

                            # Resample the image
                            data = dst.read(
                                out_shape=(dst.count, new_height, new_width), 
                                resampling=Resampling.cubic
                            )

                            # Adjust the transform to match the new resolution
                            new_transform = dst.transform * dst.transform.scale(
                                (dst.width / new_width),
                                (dst.height / new_height)
                            )

                            # Save the rescaled image
                            with rasterio.open(
                                lr_patch_path, 'w',
                                height=new_height, width=new_width,
                                count=dst.count, dtype=data.dtype,
                                crs=dst.crs, transform=new_transform
                            ) as dst_lr:
                                dst_lr.write(data)
                                    
                            patch_count += 1
        # break
    print(f"Extracted {patch_count} patches to {output_dir}")

def merge_patches(hr_dir, output_tiff_path, output_png_path, patch_size=256, overlap=0.5):
    """
    Merge high-resolution (HR) patches back into a single image and save as TIFF and PNG.

    Args:
        hr_dir (str): Path to the directory containing HR patches.
        output_tiff_path (str): Output file path for the merged image in TIFF format.
        output_png_path (str): Output file path for debugging in PNG format.
        patch_size (int): Size of each patch (default: 256).
        overlap (float): Overlap percentage (default: 0.5).
    """
    
    patch_files = sorted([f for f in os.listdir(hr_dir) if f.endswith(".tif")])
    num_patches = len(patch_files)
    
    if num_patches == 0:
        print("No patches found in the directory.")
        return

    # Read the first patch to get dimensions and metadata
    first_patch_path = os.path.join(hr_dir, patch_files[0])
    with rasterio.open(first_patch_path) as first_patch:
        num_bands = first_patch.count
        patch_profile = first_patch.profile  # Metadata
        patch_dtype = first_patch.dtypes[0]  # Data type of images

    # Determine the number of patches per row and column
    step = int(patch_size * (1 - overlap))
    width_patches = int(np.sqrt(num_patches))  # Assuming square grid
    height_patches = width_patches

    img_width = (width_patches - 1) * step + patch_size
    img_height = (height_patches - 1) * step + patch_size

    # Initialize an empty array for the merged image
    merged_image = np.zeros((img_height, img_width, num_bands), dtype=patch_dtype)

    # Place patches in correct locations
    patch_index = 0
    for i in range(height_patches):
        for j in range(width_patches):
            if patch_index >= num_patches:
                break

            patch_path = os.path.join(hr_dir, patch_files[patch_index])
            with rasterio.open(patch_path) as patch:
                patch_data = patch.read()
                patch_data = np.moveaxis(patch_data, 0, -1)  # Convert from (bands, H, W) to (H, W, bands)

            y, x = i * step, j * step  # Top-left corner position of the patch
            merged_image[y:y + patch_size, x:x + patch_size, :] = patch_data
            patch_index += 1

    # Save the merged image as TIFF
    patch_profile.update(
        height=img_height, 
        width=img_width, 
        transform=None  # Update this if geospatial data is needed
    )

    with rasterio.open(output_tiff_path, 'w', **patch_profile) as dst:
        for band in range(num_bands):
            dst.write(merged_image[:, :, band], band + 1)

    print(f"Merged image saved at {output_tiff_path}")

    # Convert to PNG for debugging
    if num_bands >= 3:
        png_image = cv2.cvtColor(merged_image[:, :, :3], cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
    else:
        png_image = merged_image[:, :, 0]  # If single channel, use grayscale

    cv2.imwrite(output_png_path, png_image)
    print(f"Debug PNG image saved at {output_png_path}")
    
class DatasetFromFolder(data.Dataset):
    def __init__(self, HR_dir, LR_dir, patch_size, upscale_factor, data_augmentation, transform=None):
        super(DatasetFromFolder, self).__init__()
        self.hr_image_filenames = [join(HR_dir, x) for x in listdir(HR_dir) if is_image_file(x)]
        self.lr_image_filenames = [join(LR_dir, x) for x in listdir(LR_dir) if is_image_file(x)]
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):

        target = load_img(self.hr_image_filenames[index])
        name = self.hr_image_filenames[index]
        lr_name = name.replace('GT', 'LR')

        input = load_img(lr_name)

        input, target, = get_patch(input, target, self.patch_size, self.upscale_factor)

        if self.data_augmentation:
            input, target, _ = augment(input, target)

        if self.transform:
            input = self.transform(input)
            target = self.transform(target)

        return input, target

    def __len__(self):
        return len(self.hr_image_filenames)


class DatasetFromFolderEval(data.Dataset):
    def __init__(self, HR_dir, LR_dir, upscale_factor, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        self.hr_image_filenames = [join(HR_dir, x) for x in listdir(HR_dir) if is_image_file(x)]
        self.lr_image_filenames = [join(LR_dir, x) for x in listdir(LR_dir) if is_image_file(x)]
        self.upscale_factor = upscale_factor
        self.transform = transform

    def __getitem__(self, index):
        target = load_img(self.hr_image_filenames[index])
        name = self.hr_image_filenames[index]
        lr_name = name.replace('GT', 'LR')
        input = load_img(lr_name)

        if self.transform:
            input = self.transform(input)
            target = self.transform(target)

        return input, target, name

    def __len__(self):
        return len(self.hr_image_filenames)

def main():
    # input_dir = "../AID/GT"
    # hr_dir = "../AID/HR"
    # if not os.path.exists(hr_dir):
    #     os.makedirs(hr_dir, exist_ok=True)
    # lr_dir = "../AID/LR"
    # if not os.path.exists(lr_dir):
    #     os.makedirs(lr_dir, exist_ok=True)
    # create_resized_patches(input_dir, hr_dir, lr_dir)
    
    input_dir = "../Orthophotos"
    # input_dir = "../results/sisr/Test-AID-farm-multiband-fliprot-iter500000-wholeimage20/AID"
    # output_dir = "../Orthophotos_patches_tiff_scale16_20_wholeimage"
    output_dir = "../Orthophotos_patches_png_scale16_noise"
    # output_dir = "../Orthophotos_patches_tiff_scale16_combine_20"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir, exist_ok=True)
    create_resized_patches_from_tiff(input_dir, output_dir, scale=16) # make png images
    # create_resized_tiff_patches_from_tiff(input_dir, output_dir ,scale=16) # make tif images
    #no overlap and doesn't exclude black patches
    # create_resized_tiff_patches_from_tiff(input_dir, output_dir, scale=16, overlap=0, black_threshold=0, max_black_ratio=1.0)
    # merge_patches(input_dir, output_dir + "/merged_farmland.tif", output_dir + "/merged_farmland.png")
    
main()