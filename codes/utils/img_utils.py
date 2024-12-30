import math
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import make_grid
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
from DISTS_pytorch import DISTS
from torch.nn.functional import adaptive_avg_pool2d

try:
    import accimage
except ImportError:
    accimage = None
    
import lpips
# from ignite.metrics import FID
# from ignite.metrics.gan.fid import fid_score
# from torchmetrics.image.fid import FrechetInceptionDistance


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def to_pil_image(pic, mode=None):

    if not (_is_numpy_image(pic) or _is_tensor_image(pic)):
        raise TypeError("pic should be Tensor or ndarray. Got {}.".format(type(pic)))

    npimg = pic
    if isinstance(pic, torch.FloatTensor):
        pic = pic.mul(255).byte()
    if torch.is_tensor(pic):
        npimg = np.transpose(pic.numpy(), (1, 2, 0))

    if not isinstance(npimg, np.ndarray):
        raise TypeError(
            "Input pic must be a torch.Tensor or NumPy ndarray, "
            + "not {}".format(type(npimg))
        )

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = "L"
        if npimg.dtype == np.int16:
            expected_mode = "I;16"
        if npimg.dtype == np.int32:
            expected_mode = "I"
        elif npimg.dtype == np.float32:
            expected_mode = "F"
        if mode is not None and mode != expected_mode:
            raise ValueError(
                "Incorrect mode ({}) supplied for input type {}. Should be {}".format(
                    mode, np.dtype, expected_mode
                )
            )
        mode = expected_mode

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ["RGBA", "CMYK"]
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError(
                "Only modes {} are supported for 4D inputs".format(
                    permitted_4_channel_modes
                )
            )

        if mode is None and npimg.dtype == np.uint8:
            mode = "RGBA"
    else:
        permitted_3_channel_modes = ["RGB", "YCbCr", "HSV"]
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError(
                "Only modes {} are supported for 3D inputs".format(
                    permitted_3_channel_modes
                )
            )
        if mode is None and npimg.dtype == np.uint8:
            mode = "RGB"

    if mode is None:
        raise TypeError("Input type {} is not supported".format(npimg.dtype))

    return Image.fromarray(npimg, mode=mode)


def to_tensor(pic):
    if not (_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError("pic should be PIL Image or ndarray. Got {}".format(type(pic)))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        return img.float().div(255)

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == "I":
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == "I;16":
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == "YCbCr":
        nchannel = 3
    elif pic.mode == "I;16":
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    """
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    """
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            "Only support 4D, 3D and 2D tensor. But received with dimension: {:d}".format(
                n_dim
            )
        )
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode="RGB"):
    cv2.imwrite(img_path, img)


def img2tensor(img):
    """
    # BGR to RGB, HWC to CHW, numpy to tensor
    Input: img(H, W, C), [0,255], np.uint8 (default)
    Output: 3D(C,H,W), RGB order, float tensor
    """
    img = img.astype(np.float32) / 255.0
    img = img[:, :, [2, 1, 0]]
    img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
    return img


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    # Validate slice dimensions to avoid empty slices
    if img1.shape[0] <= 10 or img1.shape[1] <= 10:
        raise ValueError("Input images are too small for SSIM calculation.")

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    # ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
    #     (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    # )
    
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

    return ssim_map.mean()


def calculate_ssim(img1, img2):
    """calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    """
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")

# def calculate_fid(real_images, generated_images, device="cuda"):
#     """
#     Calculate the Fréchet Inception Distance (FID) between real and generated images.

#     Args:
#         real_images (np.ndarray or torch.Tensor): Real images with shape (N, H, W, C) or (H, W, C).
#         generated_images (np.ndarray or torch.Tensor): Generated images with shape (N, H, W, C) or (H, W, C).
#         device (str): Device to run the computation ("cpu" or "cuda").

#     Returns:
#         float: The FID score.
#     """
#     def preprocess_images(images):
#         """Ensure images are in the correct shape and format."""
#         if images.ndim == 3:  # Single image: (H, W, C)
#             images = np.expand_dims(images, axis=0)  # Add batch dimension
#         if isinstance(images, np.ndarray):
#             images = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0  # Convert to (N, C, H, W)
#         return images

#     def get_activations(images, model, device):
#         """Extract activations from the penultimate layer of InceptionV3."""
#         with torch.no_grad():
#             if images.shape[1] != 3:
#                 raise ValueError("Input images must have 3 channels (RGB).")

#             # Resize input images to (299, 299)
#             images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)

#             # Pass through the InceptionV3 model
#             features = model(images.to(device)).detach().cpu().numpy()  # Forward pass to get activations
#         return features

#     # Preprocess input images
#     real_images = preprocess_images(real_images)
#     generated_images = preprocess_images(generated_images)

#     # Load pre-trained InceptionV3 model
#     inception = inception_v3(pretrained=True, transform_input=False).to(device)
#     inception.eval()

#     # Calculate activations
#     real_activations = get_activations(real_images, inception, device)
#     generated_activations = get_activations(generated_images, inception, device)

#     # Calculate mean and covariance
#     eps = 1e-6
#     mu_real, sigma_real = np.mean(real_activations, axis=0), np.cov(real_activations, rowvar=False) + eps * np.eye(real_activations.shape[1])
#     mu_generated, sigma_generated = np.mean(generated_activations, axis=0), np.cov(generated_activations, rowvar=False) + eps * np.eye(generated_activations.shape[1])

#     # Compute FID
#     diff = mu_real - mu_generated
#     covmean, _ = sqrtm(sigma_real.dot(sigma_generated), disp=False)
#     if np.iscomplexobj(covmean):
#         covmean = covmean.real
#     fid = diff.dot(diff) + np.trace(sigma_real + sigma_generated - 2 * covmean)

#     return fid

def calculate_lpips(real_images, generated_images, device="cuda"):
    """
    Calculate the Learned Perceptual Image Patch Similarity (LPIPS) between real and generated images.

    Args:
        real_images (torch.Tensor): A batch of real images with shape (N, 3, H, W).
        generated_images (torch.Tensor): A batch of generated images with shape (N, 3, H, W).
        device (str): Device to run the computation ("cpu" or "cuda").

    Returns:
        float: The LPIPS score.
    """
    
    # Add batch dimension if input is a single image
    if real_images.ndim == 3:
        real_images = np.expand_dims(real_images, axis=0)  # Shape (1, H, W, C)
    if generated_images.ndim == 3:
        generated_images = np.expand_dims(generated_images, axis=0)
        
    # Convert NumPy arrays to PyTorch tensors if needed
    if isinstance(real_images, np.ndarray):
        real_images = torch.from_numpy(real_images).permute(0, 3, 1, 2).float() / 255.0
    if isinstance(generated_images, np.ndarray):
        generated_images = torch.from_numpy(generated_images).permute(0, 3, 1, 2).float() / 255.0

    # Load pre-trained LPIPS model
    lpips_model = lpips.LPIPS(net="vgg").to(device)
    lpips_model.eval()

    # Calculate LPIPS
    with torch.no_grad():
        lpips_score = lpips_model(real_images.to(device), generated_images.to(device)).mean().item()
    return lpips_score

def calculate_dists(ref_images, deg_images, device="cuda"):
    """
    Calculate the DISTS (Deep Image Structure and Texture Similarity) metric.

    Args:
        ref_images (torch.Tensor): A batch of reference images with shape (N, 3, H, W).
        deg_images (torch.Tensor): A batch of degraded images with shape (N, 3, H, W).
        device (str): Device to run the computation ("cpu" or "cuda").

    Returns:
        torch.Tensor: The DISTS score for each image pair.
    """
    def preprocess_images(images):
        """Ensure images are in the correct shape (N, C, H, W) as PyTorch tensors."""
        if isinstance(images, np.ndarray):
            if images.ndim == 3:  # Single image: (H, W, C)
                images = np.expand_dims(images, axis=0)  # Add batch dimension: (1, H, W, C)
            images = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0  # Convert to (N, C, H, W)
        elif isinstance(images, torch.Tensor):
            if images.ndim == 3:  # Single image: (C, H, W)
                images = images.unsqueeze(0)  # Add batch dimension: (1, C, H, W)
        else:
            raise TypeError("Input images must be a NumPy array or a PyTorch tensor.")
        return images

    # Preprocess input images
    ref_images = preprocess_images(ref_images)
    deg_images = preprocess_images(deg_images)

    # Initialize DISTS metric
    dists_model = DISTS().to(device)
    dists_model.eval()

    # Compute DISTS
    with torch.no_grad():
        scores = dists_model(ref_images.to(device), deg_images.to(device))
    return scores
