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
import math

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
    
def canny_edge_detection(img):
    print("img shape", img.shape)
    if img.ndim >= 3:
        output = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        output = img
    v = np.median(output)  
    sigma = 0.33    
    lower = max(10, (1.0-sigma) * v)    
    upper = max(lower + 1, min(255, (1.0+sigma) * v))   
    edge_rgb_img = cv2.Canny(output, lower, upper) 
    return edge_rgb_img

def sobel_edge_detection(img):
    if img.ndim >= 3:
        output = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        output = img
    sobel_x = cv2.Sobel(output, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(output, cv2.CV_32F, 0, 1, ksize=3)
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)
    sobel_edges = np.uint8(np.clip(sobel_edges, 0, 255))
    _, sobel_binary = cv2.threshold(sobel_edges, 50, 255, cv2.THRESH_BINARY)
    return sobel_binary

def process_and_save_edge_masks(canny_img, canny_gt, sobel_img, sobel_gt, dataset_dir, img_name, threshold = 0.5, file_name = "RGB"):
    # Process and save Canny edge masks
    threshold = threshold
    canny_edge_mask = (canny_img > threshold).astype(np.float32)
    canny_edge_gt_mask = canny_gt * canny_edge_mask
    pred_magnitude_mask = canny_img * canny_edge_mask
    canny_edge_gt_mask = Image.fromarray(canny_edge_gt_mask).convert("L")
    canny_pred_edge_rgb_img_mask = Image.fromarray(pred_magnitude_mask).convert("L")
    canny_edge_gt_mask.save(os.path.join(dataset_dir, img_name + f"_GT_{file_name}_canny_edge_mask.png"))
    canny_pred_edge_rgb_img_mask.save(os.path.join(dataset_dir, img_name + f"_SR_{file_name}_canny_edge_mask.png"))

    # Process and save Sobel edge masks
    sobel_edge_mask = (sobel_gt > threshold).astype(np.float32)
    sobel_gt_mask = sobel_gt * sobel_edge_mask
    pred_magnitude_mask = sobel_img * sobel_edge_mask
    sobel_gt_mask = Image.fromarray(sobel_gt_mask).convert("L")
    sobel_pred_edge_rgb_img_mask = Image.fromarray(pred_magnitude_mask).convert("L")
    sobel_gt_mask.save(os.path.join(dataset_dir, img_name + f"_GT_{file_name}_sobel_edge_mask.png"))
    sobel_pred_edge_rgb_img_mask.save(os.path.join(dataset_dir, img_name + f"_SR_{file_name}_sobel_edge_mask.png"))

    # Calculate L2 loss for mask images
    canny_edge_gt_mask = np.array(canny_edge_gt_mask)
    canny_pred_edge_rgb_img_mask = np.array(canny_pred_edge_rgb_img_mask)
    sobel_gt_mask = np.array(sobel_gt_mask)
    sobel_pred_edge_rgb_img_mask = np.array(sobel_pred_edge_rgb_img_mask)
    canny_edge_rgb_l2 = np.mean((canny_edge_gt_mask - canny_pred_edge_rgb_img_mask) ** 2)
    sobel_edge_rgb_l2 = np.mean((sobel_gt_mask - sobel_pred_edge_rgb_img_mask) ** 2)

    return canny_edge_rgb_l2, sobel_edge_rgb_l2

    
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
    test_results["psnr"] = []
    test_results["ssim"] = []
    test_results["psnr_y"] = []
    test_results["ssim_y"] = []
    test_results["lpips"] = [] 
    test_results["DISTS"] = []
    test_times = []
    all_channels_avg_psnr = 0.0
    each_channels_psnrs_avg = [[] for _ in range(opt["datasets"]["test 1"]["img_channel"])]
    all_NVDI_psnr_values_from_RNIR = []
    all_NVDI_psnr_values_from_NDVI_channel = []
    # all_canny_edge_rgb_psnr = []
    # all_canny_edge_r_psnr = []
    # all_canny_edge_nir_psnr = []
    # all_sobel_edge_rgb_psnr = []
    # all_sobel_edge_r_psnr = []
    # all_sobel_edge_nir_psnr = []
    
    all_canny_edge_rgb_l2 = []
    all_canny_edge_r_l2 = []
    all_canny_edge_nir_l2 = []
    all_sobel_edge_rgb_l2 = []
    all_sobel_edge_r_l2 = []
    all_sobel_edge_nir_l2 = []
    all_canny_edge_rgb_mask_l2 = []
    all_sobel_edge_rgb_mask_l2 = []
    all_canny_edge_r_mask_l2 = []
    all_sobel_edge_r_mask_l2 = []
    all_canny_edge_nir_mask_l2 = []
    all_sobel_edge_nir_mask_l2 = []
    
    for i, test_data in enumerate(test_loader):
        single_img_psnr = []
        single_img_ssim = []
        single_img_psnr_y = []
        single_img_ssim_y = []
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
 
        
        output = util.tensor2img(visuals["Output"].squeeze(), out_type=np.float32)
        gt_img = util.tensor2img(visuals["GT"].squeeze(), out_type=np.float32)
        
        print("dtype3", output.dtype)
        print("shape3", output.shape)
        print("dtype_gt_img", gt_img.dtype)
        print("shape_gt_img", gt_img.shape)
        print("output min/max:", output.min(), output.max())
        print("gt_img min/max:", gt_img.min(), gt_img.max())

        #save as tiff file(all band) and png file(rgb) using rasterio
        save_tif_image_path = os.path.join(dataset_dir, img_name + ".tif")
        save_png_image_path = os.path.join(dataset_dir, img_name + ".png")
        
        if len(output.shape) == 2:
            output = output[:, :, np.newaxis]
            gt_img = gt_img[:, :, np.newaxis]
            print("shape after", output.shape)
            
        if output.shape[2] > 3:
            avg_tmp_psnr, each_channels_psnrs = util.calculate_psnr(output, gt_img)
            all_channels_avg_psnr += avg_tmp_psnr
            for i in range(len(each_channels_psnrs)):
                each_channels_psnrs_avg[i].append(each_channels_psnrs[i])
            #calculate NDVI
            if output.shape[2] == 8:
                sr_nir = output[:, :, 3]
                sr_red = output[:, :, 0]
                input_nir = gt_img[:, :, 3]
                input_red = gt_img[:, :, 0]
                sr_ndvi = (sr_nir - sr_red) / (sr_nir + sr_red)
                sr_ndvi = sr_ndvi = (sr_ndvi - np.min(sr_ndvi)) / (np.max(sr_ndvi) - np.min(sr_ndvi)) # normalize to 0-1
                input_ndvi = (input_nir - input_red) / (input_nir + input_red)
                input_ndvi = (input_ndvi - np.min(input_ndvi)) / (np.max(input_ndvi) - np.min(input_ndvi)) # normalize to 0-1
                print("input_nir min/max:", input_nir.min(), input_nir.max())
                print("input_red min/max:", input_red.min(), input_red.max())
                print("sr_ndvi min/max:", sr_ndvi.min(), sr_ndvi.max())
                print("input_ndvi min/max:", input_ndvi.min(), input_ndvi.max())
                sr_ndvi_from_ndvi_channel = output[:, :, 7]
                sr_ndvi_from_ndvi_channel = (sr_ndvi_from_ndvi_channel - np.min(sr_ndvi_from_ndvi_channel)) / (np.max(sr_ndvi_from_ndvi_channel) - np.min(sr_ndvi_from_ndvi_channel)) # normalize to 0-1
                print("sr_ndvi_from_ndvi_channel min/max:", sr_ndvi_from_ndvi_channel.min(), sr_ndvi_from_ndvi_channel.max())
                psnr = util.calculate_psnr(sr_ndvi_from_ndvi_channel, input_ndvi)
                if math.isnan(psnr):
                    print("PSNR is nan")
                else:
                    print("psnr", psnr)
                    all_NVDI_psnr_values_from_NDVI_channel.append(psnr)
            else:
                sr_nir = output[:, :, 3]
                sr_red = output[:, :, 0]
                input_nir = gt_img[:, :, 3]
                input_red = gt_img[:, :, 0]
                sr_ndvi = (sr_nir - sr_red) / (sr_nir + sr_red)
                sr_ndvi = (sr_ndvi - np.min(sr_ndvi)) / (np.max(sr_ndvi) - np.min(sr_ndvi)) # normalize to 0-1
                input_ndvi = (input_nir - input_red) / (input_nir + input_red)
                input_ndvi = (input_ndvi - np.min(input_ndvi)) / (np.max(input_ndvi) - np.min(input_ndvi)) # normalize to 0-1
                print("input_nir min/max:", input_nir.min(), input_nir.max())
                print("input_red min/max:", input_red.min(), input_red.max())
                print("sr_ndvi min/max:", sr_ndvi.min(), sr_ndvi.max())
                print("input_ndvi min/max:", input_ndvi.min(), input_ndvi.max())
            #calculate PSNR
            psnr = util.calculate_psnr(sr_ndvi, input_ndvi)
            if math.isnan(psnr):
                print("PSNR is nan")
            else:
                print("psnr", psnr)
                all_NVDI_psnr_values_from_RNIR.append(psnr)
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
                output_rgbnir = src.read([1,2,3,4])
                output_rgbnir = np.moveaxis(output_rgbnir, 0, -1)
                if output_rgbnir.dtype != np.uint8:
                    img_min, img_max = output_rgbnir.min(), output_rgbnir.max()
                    if img_max > img_min:
                        output_rgbnir = ((output_rgbnir - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                    else:
                        print("output min and max are equal")
                        output_rgbnir = np.zeros_like(output_rgbnir, dtype=np.uint8)
                
                print("img_rgb min/max:", output_rgbnir.min(), output_rgbnir.max())
                print("img_rgb_dtype:", output_rgbnir.dtype)
                print("output_rgb_shape:", output_rgbnir.shape)
                
                # For SR
                img_rgb = Image.fromarray(output_rgbnir[:, :, :3], 'RGB') #SR
                img_rgb.save(os.path.join(dataset_dir, img_name + "_SR.png"))
                img_rgb_gray = cv2.cvtColor(output_rgbnir[:, :, :3], cv2.COLOR_RGB2GRAY)
                img_rgb_gray = Image.fromarray(img_rgb_gray, 'L')
                img_rgb_gray.save(os.path.join(dataset_dir, img_name + "_SR_gray.png"))
                # Save only the R band (first band) as a PNG image
                img_r = Image.fromarray(output_rgbnir[:, :, 0], 'L')
                img_r.save(os.path.join(dataset_dir, img_name + "_SR_R.png"))
                # Save only the NIR band (second band) as a PNG image
                img_nir = Image.fromarray(output_rgbnir[:, :, 3], 'L')
                img_nir.save(os.path.join(dataset_dir, img_name + "_SR_NIR.png"))
                
                #RGB edge image
                canny_edge_rgb_img = canny_edge_detection(output_rgbnir[:, :, :3])
                canny_edge_rgb_img = Image.fromarray(canny_edge_rgb_img)
                canny_edge_rgb_img.save(os.path.join(dataset_dir, img_name + "_SR_RGB_canny_edge.png"))
                #Sobel edge image
                sobel_edge_rgb_img = sobel_edge_detection(output_rgbnir[:, :, :3])
                sobel_edge_rgb_img = Image.fromarray(sobel_edge_rgb_img.squeeze())
                sobel_edge_rgb_img.save(os.path.join(dataset_dir, img_name + "_SR_RGB_sobel_edge.png"))
                
                # R band edge image
                canny_edge_r_img = canny_edge_detection(output_rgbnir[:, :, 0])
                canny_edge_r_img = Image.fromarray(canny_edge_r_img)
                canny_edge_r_img.save(os.path.join(dataset_dir, img_name + "_SR_R_canny_edge.png"))
                # Sobel edge image
                sobel_edge_r_img = sobel_edge_detection(output_rgbnir[:, :, 0])
                sobel_edge_r_img = Image.fromarray(sobel_edge_r_img.squeeze())
                sobel_edge_r_img.save(os.path.join(dataset_dir, img_name + "_SR_R_sobel_edge.png"))
                
                # NIR band edge image
                canny_edge_nir_img = canny_edge_detection(output_rgbnir[:, :, 3])
                canny_edge_nir_img = Image.fromarray(canny_edge_nir_img)
                canny_edge_nir_img.save(os.path.join(dataset_dir, img_name + "_SR_NIR_canny_edge.png"))
                # Sobel edge image
                sobel_edge_nir_img = sobel_edge_detection(output_rgbnir[:, :, 3])
                sobel_edge_nir_img = Image.fromarray(sobel_edge_nir_img.squeeze())
                sobel_edge_nir_img.save(os.path.join(dataset_dir, img_name + "_SR_NIR_sobel_edge.png"))
                
                # For GT
                if gt_img.dtype != np.uint8:
                    img_min, img_max = gt_img.min(), gt_img.max()
                    if img_max > img_min:
                        gt_img = ((gt_img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                    else:
                        print("GT min and max are equal")
                        gt_img = np.zeros_like(gt_img, dtype=np.uint8)
                gt_rgb = Image.fromarray(gt_img[:, :, :3], 'RGB') #GT
                gt_rgb.save(os.path.join(dataset_dir, img_name + "_GT.png"))
                gt_rgb_gray = cv2.cvtColor(gt_img[:, :, :3], cv2.COLOR_RGB2GRAY)
                gt_rgb_gray = Image.fromarray(gt_rgb_gray, 'L')
                gt_rgb_gray.save(os.path.join(dataset_dir, img_name + "_GT_gray.png"))
                # Save only the R band (first band) as a PNG image
                gt_r = Image.fromarray(gt_img[:, :, 0], 'L')
                gt_r.save(os.path.join(dataset_dir, img_name + "_GT_R.png"))
                # Save only the NIR band (second band) as a PNG image
                gt_nir = Image.fromarray(gt_img[:, :, 3], 'L')
                gt_nir.save(os.path.join(dataset_dir, img_name + "_GT_NIR.png"))
                #RGB edge image
                canny_edge_rgb_img_gt = canny_edge_detection(gt_img[:, :, :3])
                canny_edge_rgb_img_gt = Image.fromarray(canny_edge_rgb_img_gt)
                canny_edge_rgb_img_gt.save(os.path.join(dataset_dir, img_name + "_GT_RGB_canny_edge.png"))
                #Sobel edge image
                sobel_edge_rgb_img_gt = sobel_edge_detection(gt_img[:, :, :3])
                sobel_edge_rgb_img_gt = Image.fromarray(sobel_edge_rgb_img_gt.squeeze())
                sobel_edge_rgb_img_gt.save(os.path.join(dataset_dir, img_name + "_GT_RGB_sobel_edge.png"))
                # R band edge image
                canny_edge_r_img_gt = canny_edge_detection(gt_img[:, :, 0])
                canny_edge_r_img_gt = Image.fromarray(canny_edge_r_img_gt)
                canny_edge_r_img_gt.save(os.path.join(dataset_dir, img_name + "_GT_R_canny_edge.png"))
                # Sobel edge image
                sobel_edge_r_img_gt = sobel_edge_detection(gt_img[:, :, 0])
                sobel_edge_r_img_gt = Image.fromarray(sobel_edge_r_img_gt.squeeze())
                sobel_edge_r_img_gt.save(os.path.join(dataset_dir, img_name + "_GT_R_sobel_edge.png"))
                # NIR band edge image
                canny_edge_nir_img_gt = canny_edge_detection(gt_img[:, :, 3])
                canny_edge_nir_img_gt = Image.fromarray(canny_edge_nir_img_gt)
                canny_edge_nir_img_gt.save(os.path.join(dataset_dir, img_name + "_GT_NIR_canny_edge.png"))
                # Sobel edge image
                sobel_edge_nir_img_gt = sobel_edge_detection(gt_img[:, :, 3])
                sobel_edge_nir_img_gt = Image.fromarray(sobel_edge_nir_img_gt.squeeze())
                sobel_edge_nir_img_gt.save(os.path.join(dataset_dir, img_name + "_GT_NIR_sobel_edge.png"))
                
                # calculate edge's psnr for RGB, NIR, R band
                canny_edge_rgb_img = np.array(canny_edge_rgb_img)
                canny_edge_rgb_img_gt = np.array(canny_edge_rgb_img_gt)
                sobel_edge_rgb_img = np.array(sobel_edge_rgb_img)
                sobel_edge_rgb_img_gt = np.array(sobel_edge_rgb_img_gt)
                canny_edge_r_img = np.array(canny_edge_r_img)
                canny_edge_r_img_gt = np.array(canny_edge_r_img_gt)
                sobel_edge_r_img = np.array(sobel_edge_r_img)
                sobel_edge_r_img_gt = np.array(sobel_edge_r_img_gt)
                canny_edge_nir_img = np.array(canny_edge_nir_img)
                canny_edge_nir_img_gt = np.array(canny_edge_nir_img_gt)
                sobel_edge_nir_img = np.array(sobel_edge_nir_img)
                sobel_edge_nir_img_gt = np.array(sobel_edge_nir_img_gt)
                
                #check dtype and shape and if gt and output are same
                # canny_edge_rgb_psnr = util.calculate_psnr(canny_edge_rgb_img, canny_edge_rgb_img_gt)
                # Append PSNR values to their respective lists for averaging later
                # canny_edge_r_psnr = util.calculate_psnr(canny_edge_r_img, canny_edge_r_img_gt)
                # canny_edge_nir_psnr = util.calculate_psnr(canny_edge_nir_img, canny_edge_nir_img_gt)
                # sobel_edge_rgb_psnr = util.calculate_psnr(sobel_edge_rgb_img, sobel_edge_rgb_img_gt)
                # sobel_edge_r_psnr = util.calculate_psnr(sobel_edge_r_img, sobel_edge_r_img_gt)
                # sobel_edge_nir_psnr = util.calculate_psnr(sobel_edge_nir_img, sobel_edge_nir_img_gt)
                canny_edge_rgb_l2 = np.mean((canny_edge_rgb_img - canny_edge_rgb_img_gt) ** 2)
                canny_edge_r_l2 = np.mean((canny_edge_r_img - canny_edge_r_img_gt) ** 2)
                canny_edge_nir_l2 = np.mean((canny_edge_nir_img - canny_edge_nir_img_gt) ** 2)
                sobel_edge_rgb_l2 = np.mean((sobel_edge_rgb_img - sobel_edge_rgb_img_gt) ** 2)
                sobel_edge_r_l2 = np.mean((sobel_edge_r_img - sobel_edge_r_img_gt) ** 2)
                sobel_edge_nir_l2 = np.mean((sobel_edge_nir_img - sobel_edge_nir_img_gt) ** 2)
                
                # all_canny_edge_rgb_psnr.append(canny_edge_rgb_psnr)
                # all_canny_edge_r_psnr.append(canny_edge_r_psnr)
                # all_canny_edge_nir_psnr.append(canny_edge_nir_psnr)
                # all_sobel_edge_rgb_psnr.append(sobel_edge_rgb_psnr)
                # all_sobel_edge_r_psnr.append(sobel_edge_r_psnr)
                # all_sobel_edge_nir_psnr.append(sobel_edge_nir_psnr)
                
                all_canny_edge_r_l2.append(canny_edge_r_l2)
                all_canny_edge_nir_l2.append(canny_edge_nir_l2)
                all_canny_edge_rgb_l2.append(canny_edge_rgb_l2)
                all_sobel_edge_r_l2.append(sobel_edge_r_l2)
                all_sobel_edge_nir_l2.append(sobel_edge_nir_l2)
                all_sobel_edge_rgb_l2.append(sobel_edge_rgb_l2)
                
                #add mask
                threshold = 0.05
                
                canny_edge_rgb_mask_l2, sobel_edge_rgb_mask_l2 = process_and_save_edge_masks(
                    canny_edge_rgb_img,
                    canny_edge_rgb_img_gt,
                    sobel_edge_rgb_img,
                    sobel_edge_rgb_img_gt,
                    dataset_dir,
                    img_name,
                    threshold,
                )
                all_canny_edge_rgb_mask_l2.append(canny_edge_rgb_mask_l2)
                all_sobel_edge_rgb_mask_l2.append(sobel_edge_rgb_mask_l2)
                
                canny_edge_r_mask_l2, sobel_edge_r_mask_l2 = process_and_save_edge_masks(
                    canny_edge_r_img,
                    canny_edge_r_img_gt,
                    sobel_edge_r_img,
                    sobel_edge_r_img_gt,
                    dataset_dir,
                    img_name,
                    threshold,
                    file_name="R",
                )
                all_canny_edge_r_mask_l2.append(canny_edge_r_mask_l2)
                all_sobel_edge_r_mask_l2.append(sobel_edge_r_mask_l2)
                
                canny_edge_nir_mask_l2, sobel_edge_nir_mask_l2 = process_and_save_edge_masks(
                    canny_edge_nir_img,
                    canny_edge_nir_img_gt,
                    sobel_edge_nir_img,
                    sobel_edge_nir_img_gt,
                    dataset_dir,
                    img_name,
                    threshold,
                    file_name="NIR",
                )
                all_canny_edge_r_mask_l2.append(canny_edge_nir_mask_l2)
                all_sobel_edge_r_mask_l2.append(sobel_edge_nir_mask_l2)
                
        elif output.shape[2] == 3:
            output = np.moveaxis(output, 0, -1)
            output = (output * 255).round().clip(0, 255).astype(np.uint8)
            util.save_img(output, save_png_image_path)
        elif output.shape[2] == 1:
            print("sr_ndvi min/max:", output.min(), output.max())
            print("input_ndvi min/max:", gt_img.min(), gt_img.max())
            psnr = util.calculate_psnr(output, gt_img)
            if math.isnan(psnr):
                print("PSNR is nan")
            else:
                print("psnr", psnr)
                all_NVDI_psnr_values_from_RNIR.append(psnr)
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
            sr_red = output[:, :, 0]
            sr_nir = output[:, :, 1]
            input_red = gt_img[:, :, 0]
            input_nir = gt_img[:, :, 1]
            sr_ndvi = (sr_nir - sr_red) / (sr_nir + sr_red)
            input_ndvi = (input_nir - input_red) / (input_nir + input_red)
            psnr = util.calculate_psnr(sr_ndvi, input_ndvi)
            if math.isnan(psnr):
                print("PSNR is nan")
            else:
                print("psnr", psnr)
                all_NVDI_psnr_values_from_RNIR.append(psnr)
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
    
    average_psnr_from_RNIR = np.nanmean(all_NVDI_psnr_values_from_RNIR)
    average_psnr_from_NDVI_channel = np.nanmean(all_NVDI_psnr_values_from_NDVI_channel)
    # average_canny_edge_rgb_psnr = np.nanmean(all_canny_edge_rgb_psnr)
    # average_canny_edge_r_psnr = np.nanmean(all_canny_edge_r_psnr)
    # average_canny_edge_nir_psnr = np.nanmean(all_canny_edge_nir_psnr)
    # average_sobel_edge_rgb_psnr = np.nanmean(all_sobel_edge_rgb_psnr)
    # average_sobel_edge_r_psnr = np.nanmean(all_sobel_edge_r_psnr)
    # average_sobel_edge_nir_psnr = np.nanmean(all_sobel_edge_nir_psnr)
    
    average_canny_edge_rgb_l2 = np.nanmean(all_canny_edge_rgb_l2)  
    average_canny_edge_r_l2 = np.nanmean(all_canny_edge_r_l2)
    average_canny_edge_nir_l2 = np.nanmean(all_canny_edge_nir_l2)
    average_sobel_edge_rgb_l2 = np.nanmean(all_sobel_edge_rgb_l2)
    average_sobel_edge_r_l2 = np.nanmean(all_sobel_edge_r_l2)
    average_sobel_edge_nir_l2 = np.nanmean(all_sobel_edge_nir_l2)
    average_canny_edge_rgb_mask_l2 = np.nanmean(all_canny_edge_rgb_mask_l2)
    average_sobel_edge_rgb_mask_l2 = np.nanmean(all_sobel_edge_rgb_mask_l2)
    average_canny_edge_r_mask_l2 = np.nanmean(all_canny_edge_r_mask_l2)
    average_sobel_edge_r_mask_l2 = np.nanmean(all_sobel_edge_r_mask_l2)
    average_canny_edge_nir_mask_l2 = np.nanmean(all_canny_edge_nir_mask_l2)
    average_sobel_edge_nir_mask_l2 = np.nanmean(all_sobel_edge_nir_mask_l2)
    
    all_channels_avg_psnr /= len(test_loader)
    print(f"Average PSNR for all channels: {all_channels_avg_psnr}")
    for i in range(len(each_channels_psnrs_avg)):
        each_channels_psnrs_avg[i] = np.nanmean(each_channels_psnrs_avg[i])
        print(f"Average PSNR for channel {i}: {each_channels_psnrs_avg[i]}")
    print(f"Average PSNR NDVI from RNIR: {average_psnr_from_RNIR}")
    print(f"Average PSNR NDVI from NDVI channel: {average_psnr_from_NDVI_channel}")
    # print(f"Average Canny Edge PSNR (RGB): {average_canny_edge_rgb_psnr}")
    # print(f"Average Canny Edge PSNR (R): {average_canny_edge_r_psnr}")
    # print(f"Average Canny Edge PSNR (NIR): {average_canny_edge_nir_psnr}")
    # print(f"Average Sobel Edge PSNR (RGB): {average_sobel_edge_rgb_psnr}")
    # print(f"Average Sobel Edge PSNR (R): {average_sobel_edge_r_psnr}")
    # print(f"Average Sobel Edge PSNR (NIR): {average_sobel_edge_nir_psnr}")
    
    print(f"Average Canny Edge PSNR (R): {average_canny_edge_r_l2}")
    print(f"Average Sobel Edge PSNR (R): {average_sobel_edge_r_l2}")
    print(f"Average Canny Edge PSNR (R): {average_canny_edge_r_mask_l2}")
    print(f"Average Sobel Edge PSNR (R): {average_sobel_edge_r_mask_l2}")
    print(f"Average Sobel Edge PSNR (RGB): {average_sobel_edge_rgb_l2}")
    print(f"Average Canny Edge PSNR (RGB): {average_canny_edge_rgb_mask_l2}")
    print(f"Average Sobel Edge PSNR (RGB): {average_sobel_edge_rgb_mask_l2}")
    print(f"Average Canny Edge PSNR (RGB): {average_canny_edge_rgb_l2}")
    print(f"Average Canny Edge PSNR (NIR): {average_canny_edge_nir_l2}")
    print(f"Average Sobel Edge PSNR (NIR): {average_sobel_edge_nir_l2}")
    print(f"Average Canny Edge PSNR (NIR): {average_canny_edge_nir_mask_l2}")
    print(f"Average Sobel Edge PSNR (NIR): {average_sobel_edge_nir_mask_l2}")
    
    logger.info(f"Average PSNR for all channels: {all_channels_avg_psnr}")
    for i in range(len(each_channels_psnrs_avg)):
        logger.info(f"Average PSNR for channel {i}: {each_channels_psnrs_avg[i]}")
    logger.info(f"Average PSNR NDVI from RNIR: {average_psnr_from_RNIR}")
    logger.info(f"Average PSNR NDVI from NDVI channel: {average_psnr_from_NDVI_channel}")
    # logger.info(f"Average Canny Edge PSNR (RGB): {average_canny_edge_rgb_psnr}")
    # logger.info(f"Average Canny Edge PSNR (R): {average_canny_edge_r_psnr}")
    # logger.info(f"Average Canny Edge PSNR (NIR): {average_canny_edge_nir_psnr}")
    # logger.info(f"Average Sobel Edge PSNR (RGB): {average_sobel_edge_rgb_psnr}")
    # logger.info(f"Average Sobel Edge PSNR (R): {average_sobel_edge_r_psnr}")
    # logger.info(f"Average Sobel Edge PSNR (NIR): {average_sobel_edge_nir_psnr}")
    
    logger.info(f"Average Canny Edge PSNR (R): {average_canny_edge_r_l2}")
    logger.info(f"Average Sobel Edge PSNR (R): {average_sobel_edge_r_l2}")
    logger.info(f"Average Canny Edge PSNR (R): {average_canny_edge_r_mask_l2}")
    logger.info(f"Average Sobel Edge PSNR (R): {average_sobel_edge_r_mask_l2}")
    logger.info(f"Average Canny Edge PSNR (RGB): {average_canny_edge_rgb_l2}")
    logger.info(f"Average Sobel Edge PSNR (RGB): {average_sobel_edge_rgb_l2}")
    logger.info(f"Average Canny Edge PSNR (RGB): {average_canny_edge_rgb_mask_l2}")
    logger.info(f"Average Sobel Edge PSNR (RGB): {average_sobel_edge_rgb_mask_l2}")
    logger.info(f"Average Canny Edge PSNR (NIR): {average_canny_edge_nir_l2}")
    logger.info(f"Average Sobel Edge PSNR (NIR): {average_sobel_edge_nir_l2}")
    logger.info(f"Average Canny Edge PSNR (NIR): {average_canny_edge_nir_mask_l2}")
    logger.info(f"Average Sobel Edge PSNR (NIR): {average_sobel_edge_nir_mask_l2}")
    print(f"average test time: {np.mean(test_times):.4f}")