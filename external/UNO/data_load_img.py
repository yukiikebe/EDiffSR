from glob import glob
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import kornia.filters
import glob

class EdgeDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        """
        Args:
            lr_dir (str): Path to Low-Resolution (64x64) images. --> (256x256)
            hr_dir (str): Path to High-Resolution (256x256) images.
        """
        self.lr_files = sorted(glob.glob(os.path.join(lr_dir, '*', '*.jpg')))
        self.hr_files = sorted(glob.glob(os.path.join(hr_dir, '*', '*.jpg')))

        assert len(self.lr_files) == len(self.hr_files), "Mismatch between number of LR and HR images!"

        self.to_tensor = transforms.ToTensor()
        self.to_grayscale = transforms.Grayscale(num_output_channels=1)

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        # Load images
        lr_img = Image.open(self.lr_files[idx]).convert("RGB")
        hr_img = Image.open(self.hr_files[idx]).convert("RGB")

        # Resize LR image to 256x256
        lr_img = lr_img.resize((256, 256), resample=Image.BICUBIC)
        # Grayscale and ToTensor
        # lr_img = self.to_tensor(self.to_grayscale(lr_img))  
        # hr_img = self.to_tensor(self.to_grayscale(hr_img))  
        
        lr_img = self.to_tensor(lr_img)
        hr_img = self.to_tensor(hr_img)

        # Apply Sobel filter
        # edge_lr = kornia.filters.sobel(lr_img.unsqueeze(0)).squeeze(0)  
        # edge_hr = kornia.filters.sobel(hr_img.unsqueeze(0)).squeeze(0)  

        # Save the first 5 images for debugging
        if idx < 5:
            save_dir = "./debug_images"
            os.makedirs(save_dir, exist_ok=True)
            lr_save_path = os.path.join(save_dir, f"lr_edge_{idx}.png")
            hr_save_path = os.path.join(save_dir, f"hr_edge_{idx}.png")
            transforms.ToPILImage()(lr_img).save(lr_save_path)
            transforms.ToPILImage()(hr_img).save(hr_save_path)
            
        return lr_img, hr_img
