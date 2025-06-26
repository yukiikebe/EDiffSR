import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
import logging
import os.path
import sys
import time
from collections import OrderedDict
import torchvision.utils as tvutils

import numpy as np
import torch
from IPython import embed
##import lpips

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.util import bgr2ycbcr
from DISTS_pytorch import DISTS
from pytorch_fid import fid_score
from PIL import Image
import shutil
from torchvision.transforms.functional import normalize
import lpips
import tifffile
import rasterio
import cv2
import matplotlib.pyplot as plt

def save_preprocessed_image(image, save_path):
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a NumPy array.")

    if len(image.shape) < 2:
        raise ValueError(f"Invalid image shape: {image.shape}")

    if image.dtype in [np.float32, np.float64]:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    if len(image.shape) == 3 and image.shape[2] == 1:  
        image = image[:, :, 0]
    
    # Convert NumPy array to PIL image
    img = Image.fromarray(image)

    img.save(save_path)

    
#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, default="options/test/aid.yml", help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt)

#### mkdir and logger
util.mkdirs(
    (
        path
        for key, path in opt["path"].items()
        if not key == "experiments_root"
        and "pretrain_model" not in key
        and "resume" not in key
    )
)

os.system("rm ./result")
os.symlink(os.path.join(opt["path"]["results_root"], ".."), "./result")

util.setup_logger(
    "base",
    opt["path"]["log"],
    "test_" + opt["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("base")
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    )
    logger.info(f"Number of iterations for testing in [{dataset_opt['name']}]: {len(test_loader)}")
    
    
    test_loaders.append(test_loader)

# load pretrained model by default
model = create_model(opt)
device = model.device
print(f"Device: {device}")
sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)

sde.set_model(model.model)

scale = opt['degradation']['scale']

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt["name"]  # path opt['']
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_times = []
    for i, test_data in enumerate(test_loader):
        need_GT = False if test_loader.dataset.opt["dataroot_GT"] is None else True
        img_path = test_data["GT_path"][0] if need_GT else test_data["LQ_path"][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        #### input dataset_LQ
        LQ, GT = test_data["LQ"], test_data["GT"]
        
        LQ = util.upscale(LQ, scale)
        noisy_state = sde.noise_state(LQ)

        model.feed_data(noisy_state, LQ, GT)
        tic = time.time()
        model.test(sde, save_states=True)
        toc = time.time()
        test_times.append(toc - tic)

        visuals = model.get_current_visuals()
    
        suffix = opt["suffix"]
 
        if visuals["Output"].shape[0] == 1 or visuals["Output"].shape[0] > 3:
            output = util.tensor2img(visuals["Output"].squeeze(), out_type=np.float32)
            gt_img = util.tensor2img(visuals["GT"].squeeze(), out_type=np.float32)
        else:
            output = util.tensor2img(visuals["Output"].squeeze())
            gt_img = util.tensor2img(visuals["GT"].squeeze())
        print("dtype3", output.dtype)
        print("shape3", output.shape)
        
        # psnr = util.calculate_psnr(output, gt_img)
        # print(f"PSNR: {psnr}")

        #save as tiff file(all band) and png file(rgb) using rasterio
        save_tif_image_path = os.path.join(dataset_dir, img_name + ".tif")
        save_png_image_path = os.path.join(dataset_dir, img_name + ".png")
        
        if len(output.shape) == 2:
            output = output[:, :, np.newaxis]
            print("shape after", output.shape)
            
        if output.shape[2] > 3:
            with rasterio.open(
                save_tif_image_path,
                'w',
                height=output.shape[0],
                width=output.shape[1],
                count=output.shape[2],
                dtype=output.dtype
                ) as dst:
                for band in range(output.shape[2]):
                    dst.write(output[:, :, band], band + 1)
        
            with rasterio.open(save_tif_image_path) as src:
                output = src.read([1,2,3]) #check if the bands are correct
                # if output[i].max() - output[i].min() < 5:
                #     output = np.full_like(output, fill_value=0, dtype=np.uint8)
                output = np.moveaxis(output, 0, -1)
                if output.dtype != np.uint8:
                    img_min, img_max = output.min(), output.max()
                    if img_max > img_min:
                        output = ((output - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                    else:
                        print("output min and max are equal")
                        output = np.zeros_like(output, dtype=np.uint8)
                
                img_rgb = Image.fromarray(output, 'RGB')
                img_rgb.save(save_png_image_path)
        elif output.shape[2] == 3:
            # output = np.moveaxis(output, 0, -1)
            if output.shape[2] != 3:
                raise ValueError(f"Invalid number of channels: {output.shape[2]}. Expected 3 channels.")
            if output.dtype != np.uint8:
                img_min, img_max = output.min(), output.max()
                if img_max > img_min:
                    output = ((output - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    print("output min and max are equal")
                    output = np.zeros_like(output, dtype=np.uint8)
            # output = (output * 255).round().clip(0, 255).astype(np.uint8)
            util.save_img(output, save_png_image_path)
        elif output.shape[2] == 1:
            output = output.squeeze()
            print("shape after", output.shape)
            print("dtype after", output.dtype)  
            with rasterio.open(
                save_tif_image_path,
                'w',
                height=output.shape[0],
                width=output.shape[1],
                count=1,
                dtype=output.dtype
            ) as dst:
                dst.write(output, 1)        
        elif output.shape[2] == 2:
            with rasterio.open(
                save_tif_image_path,
                'w',
                height=output.shape[0],
                width=output.shape[1],
                count=output.shape[2],
                dtype=output.dtype
            ) as dst:
                for band in range(output.shape[2]):
                    dst.write(output[:, :, band], band + 1)
        else:
            print("output shape is not valid")
    
    print(f"average test time: {np.mean(test_times):.4f}")
