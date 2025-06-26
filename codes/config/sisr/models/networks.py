import logging

import torch

from models import modules as M
import torch.nn as nn

logger = logging.getLogger("base")

import sys
sys.path.append('/home/yuki/research/EDiffSR_combine_DifGaussian/external/UNO')
from navier_stokes_uno2d import Unet2D_FNO

# Generator
def define_G(opt):
    opt_net = opt["network_G"]
    which_model = opt_net["which_model_G"]
    setting = opt_net["setting"]
    if which_model == "ConditionalNAFNet":
        uno = None
        if opt_net["use_uno"]:
            uno_config = opt_net["uno_setting"]
            uno = Unet2D_FNO(in_channels=3, 
                            num_channels=[64, 128, 256, 512, 1024], 
                            target_size=[256, 256], 
                            trunc_mode_stages=uno_config["trunc_mode_stages"], 
                            use_sobel_stages=uno_config["use_sobel_stages"], 
                            patch_based_stages=uno_config["patch_based_stages"],
                            patch_size_stages=uno_config["patch_size_stages"], 
                            factorize_mode_stages=uno_config["factorize_mode_stages"],
                            use_attn_stages=uno_config["use_attn_stages"],
                            simple_propagate_stages=uno_config["simple_propagate_stages"],
                            window_size_stages=uno_config["window_size_stages"],
                            use_swin_stages=uno_config["use_swin_stages"],
                            include_mid=True if uno_config["use_fno_only"] else False,
                            include_up=uno_config["use_fno_only"], # Enable upsampling only when FNO is standalone)
                            skip_type=uno_config["skip_type"],
                            type_grid=uno_config["type_grid"])  
            
        netG = getattr(M, which_model)(**setting, uno=uno)
    else:
        netG = getattr(M, which_model)(**setting)

    return netG


# Discriminator
def define_D(opt):
    opt_net = opt["network_D"]
    setting = opt_net["setting"]
    netD = getattr(M, which_model)(**setting)
    return netD


# Perceptual loss
def define_F(opt, use_bn=False):
    gpu_ids = opt["gpu_ids"]
    device = torch.device("cuda" if gpu_ids else "cpu")
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = M.VGGFeatureExtractor(
        feature_layer=feature_layer, use_bn=use_bn, use_input_norm=True, device=device
    )
    netF.eval()  # No need to train
    return netF
