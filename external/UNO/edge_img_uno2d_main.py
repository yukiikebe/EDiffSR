import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from data_load_img import *
import matplotlib.pyplot as plt
from navier_stokes_uno2d_pretrain import UNO, UNO_P, UNO_HiLoc
import operator
import random
from functools import reduce
from functools import partial
from edge_train_2d import train_model
from timeit import default_timer
from utilities3 import *
from Adam import Adam
import gc
import math
import os

# train_lr_path = os.path.expanduser("~/EDiffSR/research/Orthophotos_patches_png_scale16_split_ratio/LR/train/Farmland")
# train_hr_path = os.path.expanduser("~/EDiffSR/research/Orthophotos_patches_png_scale16_split_ratio/HR/train/Farmland")

train_lr_path = os.path.expanduser("~/research/EDiffSR/AID_split_matched/LR/train")
train_hr_path = os.path.expanduser("~/research/EDiffSR/AID_split_matched/HR/train")

# hyper Parameters
# Learning rate = 0.001
# Weight decay = 1e-5


S = 256  # resolution SxS
ntrain = 4000  # number of training instances
ntest = 500  # number of test instances
nval = 500  # number of validation instances
batch_size = 16
width = 48  # lifting dimension
inwidth = 3  # dimension of UNO input ( 10 time step +  position embedding )
epochs = 105
# epochs = 1

# Following code load data from two separate files containing Navier-Stokes equation simulation
dataset = EdgeDataset(lr_dir=train_lr_path, hr_dir=train_hr_path)

n_total_samples = len(dataset)

train_size = int(0.7 * n_total_samples)
val_size = int(0.15 * n_total_samples)
test_size = n_total_samples - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

gc.collect()

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = UNO_HiLoc(inwidth, width)

train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    ntrain,
    nval,
    ntest,
    weight_path="UNO_latest.pt"
)
