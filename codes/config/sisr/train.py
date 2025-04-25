import argparse
import logging
import math
import os
import random
import sys
import copy
import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
# from IPython import embed
import rasterio

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler

from data.util import bgr2ycbcr

# torch.autograd.set_detect_anomaly(True)
import matplotlib.pyplot as plt

def init_dist(backend="nccl", **kwargs):
    """ initialization for distributed training"""
    # if mp.get_start_method(allow_none=True) is None:
    # if (
    #     mp.get_start_method(allow_none=True) != "spawn"
    # ):  # Return the name of start method used for starting processes
    #     mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
    # rank = int(os.environ["RANK"])  # system env process ranks
    # num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available
    # torch.cuda.set_device(rank % num_gpus)
    # dist.init_process_group(
    #     backend=backend, **kwargs
    # )  # Initializes the default distributed process group

    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    torch.distributed.init_process_group(
        backend=backend,
        init_method="env://",  # Environment-based init
        **kwargs
    )

def main():
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, default="codes/config/sisr/options/train/setting.yml")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"  # none means disabled distributed training
    )
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    # choose small opt for SFTMD test, fill path of pre-trained model_F
    #### set random seed
    seed = opt["train"]["manual_seed"]
    
    if args.wandb:
        import wandb
        # wandb_run = wandb.init(project='super resolution ediffsr', name='Maryland_Multiband', resume='must', id='fdsp191u')
        # wandb_run = wandb.init(project='super resolution ediffsr', name='farmland_RNIR')
        # wandb_run = wandb.init(project='super resolution ediffsr', name='farmland_RGBNIR')
        # wandb_run = wandb.init(project='super resolution ediffsr', name='Maryland_Multiband2')
        wandb_run = wandb.init(project='super resolution ediffsr', name='farmland_MultibandNDVI_plus_ndvi_UNO_recalculate', resume='must', id='xntxsdld')
        # wandb_run = wandb.init(project='super resolution ediffsr', name='farmland_MultibandNDVI_plus_ndvi_UNO_recalculate_reduce_network', resume='must', id='js2egi47')
        wandb.config.update(opt, allow_val_change=True)

    #### distributed training settings
    if args.launcher == "none":  # disabled distributed training
        opt["dist"] = False
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        opt["dist"] = True
        
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["NCCL_TIMEOUT"] = "3600"
        
        init_dist()
        
        world_size = (
            torch.distributed.get_world_size()
        )  # Returns the number of processes in the current process group
        rank = torch.distributed.get_rank()  # Returns the rank of current process group
        # util.set_random_seed(seed)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        print(f"[DEBUG] Process {os.getpid()} assigned to GPU {local_rank}")
        device = torch.device("cuda", local_rank)
        
    torch.cuda.set_per_process_memory_fraction(0.9, device=torch.cuda.current_device())
    
    # torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    ###### Predictor&Corrector train ######

    #### loading resume state if exists
    if opt["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt, resume_state["iter"])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0-7)
        if resume_state is None:
            # Predictor path
            util.mkdir_and_rename(
                opt["path"]["experiments_root"]
            )  # rename experiment folder if exists
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                    and "pretrain_model" not in key
                    and "resume" not in key
                )
            )
            # os.system("rm ./log")
            # os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")

        # config loggers. Before it, the log will not work
        util.setup_logger(
            "base",
            opt["path"]["log"],
            "train_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        util.setup_logger(
            "val",
            opt["path"]["log"],
            "val_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    "You are using PyTorch {}. Tensorboard will use [tensorboardX]".format(
                        version
                    )
                )
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(opt["name"]))
    else:
        util.setup_logger(
            "base", opt["path"]["log"], "train", level=logging.INFO, screen=False
        )
        logger = logging.getLogger("base")


    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    edge_gradient = None
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt["dist"]:
                train_sampler = DistIterSampler(
                    train_set, world_size, rank, dataset_ratio
                )
                total_epochs = int(
                    math.ceil(total_iters / (train_size * dataset_ratio))
                )
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )
            
        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None
    assert val_loader is not None

    #### create model
    model = create_model(opt, wandb_run=wandb_run if args.wandb else None)
    device = model.device

    #### resume training
    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )

        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    sde.set_model(model.model)

    scale = opt['degradation']['scale']

    #### training
    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )

    best_psnr = 0.0
    best_iter = 0
    error = mp.Value('b', False)
    batch_size = opt["datasets"]["train"]["batch_size"]

    # -------------------------------------------------------------------------
    # -------------------------正式开始训练，前面都是废话---------------------------
    # -------------------------------------------------------------------------
    for epoch in range(start_epoch, total_epochs + 1):
        if opt["dist"]:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1

            if current_step > total_iters:
                break

            LQ, GT = train_data["LQ"], train_data["GT"]  #  b 3 32 32; b 3 128 128

            LQ = util.upscale(LQ, scale)  #  bicubic, which can be repleced by deep networks

            # random timestep and state (noisy map) via SDE
            timesteps, states = sde.generate_random_states(x0=GT, mu=LQ)  # t=batchsize，states [b 3 128 128]
            model.feed_data(states, LQ, GT)  # xt, mu, x0, 将加了噪声的LR图xt，LR以及GT输入改进的UNet进行去噪

            model.optimize_parameters(current_step, timesteps, sde)  # 优化UNet
            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )
            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.get_current_learning_rate()
                )
                for k, v in logs.items():
                    message += "{:s}: {:.4e} ".format(k, v)
                    # tensorboard logger
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)
                    if args.wandb:
                        if args.wandb:
                            wandb_run.log(
                                {
                                    "epoch": epoch,
                                    "iter": current_step,
                                    "lr": model.get_current_learning_rate(),
                                    "samples": current_step * batch_size,
                                    **logs
                                },
                                step=current_step * batch_size
                            )
                    

            # validation, to produce ker_map_list(fake)
            if current_step % opt["train"]["val_freq"] == 0 and rank <= 0:
                avg_psnr = 0.0
                each_channels_psnrs_avg = [[] for _ in range(opt["datasets"]["val"]["img_channel"])]
                idx = 0
                for _, val_data in enumerate(val_loader):

                    LQ, GT = val_data["LQ"], val_data["GT"]
                    LQ = util.upscale(LQ, scale)
                    noisy_state = sde.noise_state(LQ)  # 在LR上加噪声，得到噪声LR图，噪声是随机生成的
                    if not os.path.exists("~/EDiffSR/tmp"):
                        os.makedirs("~/EDiffSR/tmp")
                    
                    # valid Predictor
                    model.feed_data(noisy_state, LQ, GT)
                    model.test(sde)
                    visuals = model.get_current_visuals()
                    
                    # for channel in range(visuals["Output"].shape[0]):
                    #     print(f"output channel before {channel} min:", visuals["Output"][channel, :, :].min())
                    #     print(f"output channel before {channel} max:", visuals["Output"][channel, :, :].max())

                    print(visuals["Output"].shape)
                    print(visuals["GT"].shape)
                    
                    if visuals["Output"].shape[0] > 3:
                        # output = visuals["Output"].squeeze().cpu().numpy()
                        # gt_img = visuals["GT"].squeeze().cpu().numpy()
                        
                        output = util.tensor2img(visuals["Output"].squeeze(), out_type=np.float32)  # float32
                        # for channel in range(output.shape[2]):
                        #     print(f"output channel {channel} min:", output[:, :, channel].min())
                        #     print(f"output channel {channel} max:", output[:, :, channel].max())
                        gt_img = util.tensor2img(visuals["GT"].squeeze(), out_type=np.float32)  # uint8
                        # for channel in range(gt_img.shape[2]):
                        #     print(f"output channel {channel} min:", gt_img[:, :, channel].min())
                        #     print(f"output channel {channel} max:", gt_img[:, :, channel].max())
                        avg_tmp_psnr, each_channels_psnrs = util.calculate_psnr(output, gt_img)
                        avg_psnr += avg_tmp_psnr
                        for i in range(len(each_channels_psnrs)):
                            each_channels_psnrs_avg[i].append(each_channels_psnrs[i])
                        
                        save_path = str(opt["path"]["experiments_root"]) + '/val_images/' + str(current_step)
                        util.mkdirs(save_path)
                        save_name = f"{save_path}/{idx:03d}_multi_channel.tif"
                        print(output.shape)
                        if len(os.listdir(save_path)) < 100:    
                            with rasterio.open(save_name, 'w', height=output.shape[0], width=output.shape[1], count=output.shape[2], dtype=output.dtype) as dst:
                                for band in range(output.shape[2]):
                                    dst.write(output[:, :, band], band + 1)
            
                    elif visuals["Output"].shape[0] == 2:
                        output = util.tensor2img(visuals["Output"].squeeze())  # uint8
                        gt_img = util.tensor2img(visuals["GT"].squeeze())
                    
                        tmp_avg_psnr, tmp_each_channels_psnrs_avg = util.calculate_psnr(output, gt_img)
                        avg_psnr += tmp_avg_psnr
                        for i in range(len(tmp_each_channels_psnrs_avg)):
                            each_channels_psnrs_avg[i].append(tmp_each_channels_psnrs_avg[i])

                        print(f"each_channels_psnrs_avg: {each_channels_psnrs_avg}")
                        print(f"tmp_each_channels_psnrs_avg: {tmp_each_channels_psnrs_avg}")
                        # for i in range(len(each_channels_psnrs_avg)):
                        #     each_channels_psnrs_avg[i] += each_channels_psnrs_avg[i]
                        
                        # Create a fake third channel by duplicating one of the existing channels
                        fake_channel = visuals["Output"][0:1, :, :].clone()
                        output = torch.cat((visuals["Output"], fake_channel), dim=0)
                        output = util.tensor2img(output.squeeze())  # Convert to uint8 RGB image

                        # Save the RGB image
                        save_path = str(opt["path"]["experiments_root"]) + '/val_images/' + str(current_step)
                        util.mkdirs(save_path)
                        save_name = f"{save_path}/{idx:03d}_RGB.png"
                        util.save_img(output, save_name)
            
                    elif visuals["Output"].shape[0] == 1:
                        # Grayscale Image (1-band)
                        output = util.tensor2img(visuals["Output"].squeeze(), out_type=np.float32)  # uint8
                        gt_img = util.tensor2img(visuals["GT"].squeeze(), out_type=np.float32)

                        avg_psnr += util.calculate_psnr(output, gt_img)

                        save_path = str(opt["path"]["experiments_root"]) + '/val_images/' + str(current_step)
                        util.mkdirs(save_path)
                        save_name = f"{save_path}/{idx:03d}_Grayscale.png"

                        cv2.imwrite(save_name, output)
            
                    else:      
                        output = util.tensor2img(visuals["Output"].squeeze())  # uint8
                        gt_img = util.tensor2img(visuals["GT"].squeeze())  # uint8

                        # save the validation results
                        save_path = str(opt["path"]["experiments_root"]) + '/val_images/' + str(current_step)
                        util.mkdirs(save_path)
                        save_name = save_path + '/'+'{0:03d}'.format(idx) + '.png'
                        util.save_img(output, save_name)

                        # calculate PSNR
                        avg_psnr += util.calculate_psnr(output, gt_img)
                    idx += 1

                avg_psnr = avg_psnr / idx
                for i in range(len(each_channels_psnrs_avg)):
                    each_channels_psnrs_avg[i] = np.nanmean(each_channels_psnrs_avg[i])

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    best_iter = current_step

                # log
                if args.wandb:
                    log_data = {
                        "epoch": epoch,
                        "iter": current_step,
                        "all_channels_psnr": avg_psnr,
                        "samples": current_step * batch_size,
                    }
                    for i, channel_psnr in enumerate(each_channels_psnrs_avg):
                        log_data[f"channel_{i+1}_psnr"] = channel_psnr
                    wandb_run.log(log_data, step=current_step * batch_size)
                    
                logger.info("# Validation # PSNR: {:.6f}, Best PSNR: {:.6f}| Iter: {}, ".format(avg_psnr, best_psnr, best_iter))
                logger_val = logging.getLogger("val")  # validation logger
                logger_val.info(
                    "<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
                        epoch, current_step, avg_psnr
                    )
                )
                logger.info("Each Channel PSNR: {}".format(", ".join(["{:.6f}".format(psnr) for psnr in each_channels_psnrs_avg])))
                print("<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
                        epoch, current_step, avg_psnr
                    ))
                # tensorboard logger
                if opt["use_tb_logger"] and "debug" not in opt["name"]:
                    tb_logger.add_scalar("psnr", avg_psnr, current_step)

            if error.value:
                sys.exit(0)
            #### save models and training states
            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                if rank <= 0:
                    logger.info("Saving models and training states.")
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info("Saving the final model.")
        model.save("latest")
        logger.info("End of Predictor and Corrector training.")
    tb_logger.close()


if __name__ == "__main__":
    main()
