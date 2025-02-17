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

def save_preprocessed_image(image, save_path):
    # Convert tensor or array to uint8 if needed
    if isinstance(image, np.ndarray):
        if image.max() <= 1.0:  # Normalize to [0, 255]
            image = (image * 255).astype(np.uint8)
    else:
        raise ValueError("Image must be a NumPy array.")

    # Resize image to 299x299
    img = Image.fromarray(image)
    # img = img.resize((299, 299), Image.ANTIALIAS)
    img = img.resize((299, 299), Image.Resampling.LANCZOS)
    print(f"Resized image shape: {img.size}")

    # Save resized image
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
    test_results["psnr"] = []
    test_results["ssim"] = []
    test_results["psnr_y"] = []
    test_results["ssim_y"] = []
    test_results["lpips"] = [] 
    test_results["DISTS"] = []
    test_times = []
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
        # LQ = test_data["LQ"]
        LQ = util.upscale(LQ, scale)
        noisy_state = sde.noise_state(LQ)

        # model.feed_data(noisy_state, LQ, GT)
        model.feed_data(noisy_state, LQ, GT)
        tic = time.time()
        model.test(sde, save_states=True)
        toc = time.time()
        test_times.append(toc - tic)

        # visuals = model.get_current_visuals(need_GT=False)
        visuals = model.get_current_visuals()
        SR_img = visuals["Output"]
        output = SR_img
        # output = util.tensor2img(SR_img.squeeze())  # uint8
        # LQ_ = util.tensor2img(visuals["Input"].squeeze())  
        # HR_ = util.tensor2img(visuals["GT"].squeeze())  
        # print(f"Type of output: {type(output)}")
        
        # # calculate PSNR and SSIM
        # if need_GT:
        #     # GT = util.tensor2img(GT.squeeze())
        #     GT = util.tensor2img(visuals["GT"].squeeze())
        #     psnr = util.calculate_psnr(output, GT)
        #     ssim = util.calculate_ssim(output, GT)
        #     psnr_y = util.calculate_psnr(bgr2ycbcr(output), bgr2ycbcr(GT))
        #     ssim_y = util.calculate_ssim(bgr2ycbcr(output), bgr2ycbcr(GT))
        #     # lpips = util.calculate_lpips(output, GT)
        #     output_tensor = visuals["Output"].unsqueeze(0).to(device) if visuals["Output"].ndimension() == 3 else visuals["Output"].to(device)
        #     HR_tensor = visuals["GT"].unsqueeze(0).to(device) if visuals["GT"].ndimension() == 3 else visuals["GT"].to(device)
            
        #     # dists = util.calculate_dists(GT, GT)
        #     test_results["psnr"].append(psnr)
        #     test_results["ssim"].append(ssim)
        #     test_results["psnr_y"].append(psnr_y)
        #     test_results["ssim_y"].append(ssim_y)
            
        #     dists = DISTS()
        #     dists = dists.cuda()
        #     dists_score = dists(output_tensor, HR_tensor)
        #     test_results["DISTS"].append(dists_score.item())
            
        #     loss_fn_vgg = lpips.LPIPS(net="vgg").to(device)
        #     mean = [0.5, 0.5, 0.5]
        #     std = [0.5, 0.5, 0.5]
        #     normalize(HR_tensor, mean, std, inplace=True)
        #     normalize(output_tensor, mean, std, inplace=True)
        #     lpips_score = loss_fn_vgg(output_tensor, HR_tensor)
        #     test_results["lpips"].append(lpips_score.item())
            
            
        #     logger.info(
        #         "{:3d} - {:25} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}; LPIPS: {:.6f}; DISTS: {:.6f}".format(
        #             i + 1, img_name, psnr, ssim, psnr_y, ssim_y, lpips_score.item(), dists_score.item()
        #         )
        #     )
        # else:
        #     logger.info("{:3d} - {:25}".format(i + 1, img_name))
        
        # # average the results
        # if (i + 1) % 10 == 0:
        #     ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
        #     ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])
        #     ave_psnr_y = sum(test_results["psnr_y"]) / len(test_results["psnr_y"])
        #     ave_ssim_y = sum(test_results["ssim_y"]) / len(test_results["ssim_y"])
        #     ave_lpips = sum(test_results["lpips"]) / len(test_results["lpips"])
        #     ave_dists = sum(test_results["DISTS"]) / len(test_results["DISTS"])
        #     logger.info(
        #         "Average: PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}; LPIPS: {:.6f}; DISTS: {:.6f}".format(
        #             ave_psnr, ave_ssim, ave_psnr_y, ave_ssim_y, ave_lpips, ave_dists
        #         )
        #     )
        
        suffix = opt["suffix"]
        print(f"Output shape: {output.shape}")
        output_np = output.cpu().numpy()
        if output_np.shape[0] == 7:
            output_np = np.transpose(output_np, (1, 2, 0))  # (H, W, C) に変換
            save_img_path = os.path.join(dataset_dir, img_name + ".tif")
            tifffile.imwrite(save_img_path, output_np.astype(np.float32))
        else:
            if suffix:
                save_img_path = os.path.join(dataset_dir, img_name + suffix + ".png")
            else:
                save_img_path = os.path.join(dataset_dir, img_name + ".png")
            util.save_img(output, save_img_path)

        # if need_GT:
        #     hr_fid_dir = os.path.join(dataset_dir, 'fid_HR')
        #     sr_fid_dir = os.path.join(dataset_dir, 'fid_SR')
            
        #     if os.path.exists(hr_fid_dir):
        #         shutil.rmtree(hr_fid_dir)  # Remove existing directory
        #     os.makedirs(hr_fid_dir, exist_ok=True)

        #     if os.path.exists(sr_fid_dir):
        #         shutil.rmtree(sr_fid_dir)  # Remove existing directory
        #     os.makedirs(sr_fid_dir, exist_ok=True)
            
            # hr_save_path = os.path.join(hr_fid_dir, f'{img_name}.png')
            # print(f"HR Save Path: {hr_save_path}")
            # sr_save_path = os.path.join(sr_fid_dir, f'{img_name}.png')
            # print(f"SR Save Path: {sr_save_path}")
            # Save GT and SR images
            # save_preprocessed_image(HR_, hr_save_path)
            # save_preprocessed_image(output, sr_save_path)
            
            # fid_value = fid_score.calculate_fid_given_paths(
            #     [hr_fid_dir, sr_fid_dir],  # Directories containing GT and SR images
            #     batch_size=50,             # Adjust based on your GPU memory
            #     device='cuda:0',           # Use GPU if available
            #     dims=2048                  # Standard FID Inception feature dimensions
            # )
            # logger.info(f"FID Score: {fid_value}")
            # print(f"FID Score: {fid_value}")
        
    print(f"average test time: {np.mean(test_times):.4f}")
