import logging
from collections import OrderedDict
import os
import numpy as np

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torchvision.utils as tvutils
from tqdm import tqdm
from ema_pytorch import EMA
from utils.sde_utils import IRSDE
import time
import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.optimizer import Lion
from pytorch_msssim import SSIM

from models.modules.loss import MatchingLoss
import kornia
from .base_model import BaseModel
import wandb
from torchvision.utils import save_image
import einops
from torch.profiler import record_function

import sys
sys.path.append('/home/yuki/research/EDiffSR/external/UNO')
from navier_stokes_uno2d import UNO, UNO_S256

# sys.path.append('/home/yuki/EDiffSR/external/galerkin_transformer/libs')
# from model import SimpleTransformerEncoderLayer

# sys.path.append('/home/yuki/EDiffSR/external/galerkin_transformer')
# from galerkin_attention import SimpleAttention

logger = logging.getLogger("base")

def make_coord(shape, ranges=None, flatten=True):
        """ Make coordinates at grid centers.
        """
        coord_seqs = []
        for i, n in enumerate(shape):
            if ranges is None:
                v0, v1 = -1, 1
            else:
                v0, v1 = ranges[i]
            r = (v1 - v0) / (2 * n)
            seq = v0 + r + (2 * r) * torch.arange(n).float()
            coord_seqs.append(seq)
        #ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
        ret = torch.stack(torch.meshgrid(*coord_seqs,indexing='ij'), dim=-1)
        if flatten:
            ret = ret.view(-1, ret.shape[-1])
        return ret
    
class DenoisingModel(BaseModel):
    def __init__(self, opt, wandb_run=None):
        super(DenoisingModel, self).__init__(opt)
        self.wandb_run = wandb_run
        self.opt = opt

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt["train"]

        # define network and load pretrained models
        
        if self.is_train:
            in_width = train_opt['uno_inwidth']
            width = train_opt['uno_width']
            self.save_path_model_input = opt["path"]["experiments_root"] + "/model_input_images"
        else:
            in_width = 12
            width = 32
            self.save_path_model_input = None
        
        if self.is_train:
            self.save_path_model_input = opt["path"]["experiments_root"] + "/model_input_images"
        else:
            self.save_path_model_input = None
        
        self.model = networks.define_G(opt).to(self.device)

        if opt["dist"]:
            self.model = DistributedDataParallel(
                self.model, device_ids=[torch.cuda.current_device()]
            )
        
        self.load()

        if self.is_train:
            self.model.train()

            is_weighted = opt['train']['is_weighted']
            loss_type = opt['train']['loss_type']
            self.loss_fn = MatchingLoss(loss_type, is_weighted).to(self.device)
            self.weight = opt['train']['weight']

            # optimizers
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            optim_params = []
            for (
                k,
                v,
            ) in self.model.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))

            if train_opt['optimizer'] == 'Adam':
                self.optimizer = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'AdamW':
                self.optimizer = torch.optim.AdamW(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'Lion':
                self.optimizer = Lion(
                    optim_params, 
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            else:
                print('Not implemented optimizer, default using Adam!')
            self.optimizers.append(self.optimizer)
            
            # schedulers
            print("self.optimizers: ", self.optimizers)
            if train_opt["lr_scheme"] == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            train_opt["lr_steps"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
                    )
            elif train_opt["lr_scheme"] == "TrueCosineAnnealingLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, 
                            T_max=train_opt["niter"],
                            eta_min=train_opt["eta_min"])
                    )
            else:
                raise NotImplementedError("MultiStepLR learning rate scheme is enough.")

            self.ema = EMA(self.model, beta=0.995, update_every=10).to(self.device)
            self.log_dict = OrderedDict()
            self.counter = 0

    def feed_data(self, state, LQ, GT=None):
        self.state = state.to(self.device)    # noisy_state
        self.condition = LQ.to(self.device)  # LQ
        if GT is not None:
            self.state_0 = GT.to(self.device)  # GT

    def optimize_parameters(self, step, timesteps, sde:IRSDE=None, use_total_loss=False):
        sde.set_mu(self.condition) #LR image
        self.optimizer.zero_grad()
        timesteps = timesteps.to(self.device)
                
        # Get noise and score
        if self.model.uno is not None:
            freq_features = self.model.uno(self.condition.permute(0, 2, 3, 1))
        noise = sde.noise_fn(self.state, timesteps.squeeze(), freq_features=freq_features)

        score = sde.get_score_from_noise(noise, timesteps)
        
        # Learning the maximum likelihood objective for state x_{t-1}
        xt_1_expection = sde.reverse_sde_step_mean(self.state, score, timesteps)
        xt_1_optimum = sde.reverse_optimum_step(self.state, self.state_0, timesteps)
        
        base_loss = self.weight * self.loss_fn(xt_1_expection, xt_1_optimum)
        if use_total_loss:
            total_loss = self.compute_combined_loss(xt_1_expection, self.state_0, base_loss)
        else:
            total_loss = base_loss
        # total_loss = self.compute_combined_loss(xt_1_expection, self.state_0, base_loss)
            
        total_loss.backward()
        self.optimizer.step()
        
        self.ema.update()
        # set log
        self.log_dict["base_loss"] = base_loss.item()
        self.log_dict["total_loss"] = total_loss.item()
    
    def extract_luminance(self, pred, target):
        if pred.shape[1] >= 3:
            pred_extract = kornia.color.rgb_to_ycbcr(pred[:, :3, :, :])[:, 0:1, :, :]
            target_extract = kornia.color.rgb_to_ycbcr(target[:, :3, :, :])[:, 0:1, :, :]
        else:
            pred_extract = pred[:, 0:1, :, :]
            target_extract = target[:, 0:1, :, :]
        
        if pred_extract.min() < 0 or pred_extract.max() > 1:
            pred_extract = (pred_extract - pred_extract.min()) / (pred_extract.max() - pred_extract.min() + 1e-8)
        if target_extract.min() < 0 or target_extract.max() > 1:
            target_extract = (target_extract - target_extract.min()) / (target_extract.max() - target_extract.min() + 1e-8)
        
        return pred_extract, target_extract
    
    def compute_combined_loss(self, pred, target, base_loss):
        pred_y, target_y = self.extract_luminance(pred, target)
        #calculate ssim loss
        ssim_module = SSIM(data_range=1.0, size_average=True, channel=1)
        ssim_loss = 1 - ssim_module(pred_y, target_y)
        tv_loss = kornia.losses.total_variation(pred_y, reduction='mean').mean()
        l2_loss = F.mse_loss(pred, target, reduction='none')
        l2_loss = einops.reduce(l2_loss, 'b ... -> b (...)', 'mean').mean()
        total_loss = 0.6 * base_loss + 0.3 * l2_loss + 0.05 * ssim_loss + 0.05 * tv_loss
        
        self.log_dict["ssim_loss"] = ssim_loss.item()
        self.log_dict["tv_loss"] = tv_loss.item()
        self.log_dict["l2_loss"] = l2_loss.item()
        self.log_dict["total_loss"] = total_loss.item()
        return total_loss
        
    def compute_l2_loss(self, pred, target, step):
        batch_size = pred.shape[0]
        
        #only red
        # pred_extract = pred[:, 0:1, :, :]
        # target_extract = target[:, 0:1, :, :]
        
        #only NIR
        # pred_extract = pred[:, 3:4, :, :]
        # target_extract = target[:, 3:4, :, :]
            
        #RGB
        if pred.shape[1] >= 3:
            pred_extract = kornia.color.rgb_to_grayscale(pred[:, :3, :, :])
            target_extract = kornia.color.rgb_to_grayscale(target[:, :3, :, :])

        
        for i in range(batch_size):
            if step % 100 == 0:
                self.counter = 0
                pred_img = pred_extract[i].detach().cpu()
                target_img = target_extract[i].detach().cpu()

                pred_img = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min() + 1e-8)
                target_img = (target_img - target_img.min()) / (target_img.max() - target_img.min() + 1e-8)

                step_dir = f"/home/yuki/research/EDiffSR/codes/config/sisr/model_input_images/step_{step:05d}"

                output_dir_pred = os.path.join(step_dir, "SR_images")
                output_dir_target = os.path.join(step_dir, "GT_images")
                # output_dir_output = os.path.join(step_dir, "output_images")
                # output_dir_state0 = os.path.join(step_dir, "state0_images")

                os.makedirs(output_dir_pred, exist_ok=True)
                os.makedirs(output_dir_target, exist_ok=True)
                # os.makedirs(output_dir_output, exist_ok=True)
                # os.makedirs(output_dir_state0, exist_ok=True)
                
                # save_image(self.tmp_output_rgb, os.path.join(output_dir_output, f'output_{self.counter}_{i}.png'))
                # save_image(self.state_0_rgb, os.path.join(output_dir_state0, f'state0_{self.counter}_{i}.png'))
                save_image(pred_img, os.path.join(output_dir_pred, f'pred_{self.counter}_{i}.png'))
                save_image(target_img, os.path.join(output_dir_target, f'target_{self.counter}_{i}.png'))
                if self.wandb_run is not None:
                    if self.counter < 3:
                        pred_img_wandb = (pred_img.squeeze().numpy() * 255).astype(np.uint8)
                        target_img_wandb = (target_img.squeeze().numpy() * 255).astype(np.uint8)
                        # state_0_img_wandb = (self.state_0_rgb.squeeze().numpy() * 255).astype(np.uint8)
                        # output_img_wandb = (self.tmp_output_rgb.squeeze().numpy() * 255).astype(np.uint8)
                        try:
                            self.wandb_run.log({
                                f"pred_{self.counter}_{i}": wandb.Image(pred_img_wandb),
                                f"target_{self.counter}_{i}": wandb.Image(target_img_wandb),
                                # f"state0_{self.counter}_{i}": wandb.Image(state_0_img_wandb),
                                # f"output_{self.counter}_{i}": wandb.Image(output_img_wandb)
                            })
                        except Exception as e:
                            print("wandb log error: ", e)
                            pass
                    
        # Sobel filter for edge detection
        pred_edges = kornia.filters.sobel(pred_extract) 
        target_edges = kornia.filters.sobel(target_extract)  
        # output_edges = kornia.filters.sobel(self.tmp_output_rgb)  
        # state_0_edges = kornia.filters.sobel(self.state_0_rgb) 
            
        # Canny edge detection
        # v = torch.median(pred_extract).item()
        # sigma = 0.33
        # lower = max(0.01, (1.0 - sigma) * v)
        # upper = max(lower + 1e-5, min(1.0, (1.0 + sigma) * v))
        # pred_edges, _ = kornia.filters.canny(pred_extract, low_threshold=lower, high_threshold=upper)
        # target_edges, _ = kornia.filters.canny(target_extract, low_threshold=lower, high_threshold=upper)
        
        # print("pred_edges: ", pred_edges.shape)
        # print("target_edges: ", target_edges.shape)
        print("pred_edges: ", pred_edges.shape)
            
        if pred_edges.shape[1] == 2:
            pred_magnitude = torch.sqrt(pred_edges[:, 0] ** 2 + pred_edges[:, 1] ** 2)
            target_magnitude = torch.sqrt(target_edges[:, 0] ** 2 + target_edges[:, 1] ** 2)
        else:
            pred_magnitude = pred_edges.squeeze(1)
            target_magnitude = target_edges.squeeze(1)
        
        #Mask
        therehold = 0.02
        mask = (target_magnitude > therehold).float()
        
        #loss
        diff = torch.abs(pred_magnitude - target_magnitude) ** 2
        masked_loss = (diff * mask).sum() / (mask.sum().clamp(min=1e-8))
            
        #save images
        if step % 100 == 0:
            # if self.counter < 10:
            for i in range(batch_size):
                pred_img = (pred_magnitude[i]).detach().cpu()
                target_img = (target_magnitude[i]).detach().cpu()
                
                step_edge_dir = f"/home/yuki/research/EDiffSR/codes/config/sisr/model_input_images/step_{step:05d}/edges"
                if not os.path.exists(step_edge_dir):
                    os.makedirs(step_edge_dir)
                
                output_dir_pred = os.path.join(step_edge_dir, "./model_input_images/SR_edges")
                output_dir_target = os.path.join(step_edge_dir,"./model_input_images/GT_edges")
    
                os.makedirs(output_dir_pred, exist_ok=True)
                os.makedirs(output_dir_target, exist_ok=True)

                save_image(pred_img, os.path.join(output_dir_pred, f'pred_edge{self.counter}_{i}.png'))
                save_image(target_img, os.path.join(output_dir_target, f'target_edge{self.counter}_{i}.png'))

                if self.wandb_run is not None:
                    if self.counter < 3:
                        pred_img_wandb = (pred_img.squeeze().numpy() * 255).astype(np.uint8)
                        target_img_wandb = (target_img.squeeze().numpy() * 255).astype(np.uint8)
                        self.wandb_run.log({
                            f"pred_edge_{self.counter}_{i}": wandb.Image(pred_img_wandb),
                            f"target_edge_{self.counter}_{i}": wandb.Image(target_img_wandb),
                        })
            self.counter += 1
        
        return masked_loss
    
    def test(self, sde=None, save_states=False):
        sde.set_mu(self.condition)

        self.model.eval()
        with torch.no_grad():
            freq_features = None
            if self.model.uno is not None:
                freq_features = self.model.uno(self.condition.permute(0, 2, 3, 1))

            self.output = sde.reverse_sde(self.state, save_states=save_states, freq_features=freq_features)

        self.model.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["Input"] = self.condition.detach()[0].float().cpu()
        out_dict["Output"] = self.output.detach()[0].float().cpu()
        if need_GT:
            out_dict["GT"] = self.state_0.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel) or isinstance(
            self.model, DistributedDataParallel
        ):
            net_struc_str = "{} - {}".format(
                self.model.__class__.__name__, self.model.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(self.model.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                "Network G structure: {}, with parameters: {:,d}".format(
                    net_struc_str, n
                )
            )
            logger.info(s)

    def load(self):
        load_path_G = self.opt["path"]["pretrain_model_G"]
        if load_path_G is not None:
            logger.info("Loading model for G [{:s}] ...".format(load_path_G))
            self.load_network(load_path_G, self.model, self.opt["path"]["strict_load"])

    def save(self, iter_label):
        self.save_network(self.model, "G", iter_label)
        self.save_network(self.ema.ema_model, "EMA", 'lastest')
