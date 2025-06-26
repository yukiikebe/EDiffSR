import logging

import torch

from models import modules as M
import torch.nn as nn

logger = logging.getLogger("base")

import sys
sys.path.append('/home/yuki/research/EDiffSR/external/UNO')
from navier_stokes_uno2d import UNO, UNO_HiLoc

sys.path.append('/home/yuki/research/EDiffSR/external/galerkin_transformer/libs')

# Generator
def define_G(opt):
    opt_net = opt["network_G"]
    which_model = opt_net["which_model_G"]
    setting = opt_net["setting"]
    if which_model == "ConditionalNAFNet":
        uno = None
        if opt_net["use_uno"]:
            if opt_net["use_uno_hiloc"]:
                galerkin_config = opt_net.get("galerkin_transformer_setting", {})
                uno = UNO_HiLoc(
                    in_width=opt_net["uno_in_width"],
                    width=opt_net["uno_width"],
                    galerkin_config = galerkin_config
                )
            else:
                uno = UNO(
                    in_width=opt_net["uno_in_width"],
                    width=opt_net["uno_width"]
                )
            
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
