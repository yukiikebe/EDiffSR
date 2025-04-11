import os
import random
import sys

import cv2
import lmdb
import numpy as np
import torch
import torch.utils.data as data

try:
    sys.path.append("..")
    import data.util as util
except ImportError:
    pass


class LQGTDataset(data.Dataset):
    """
    Read LR (Low Quality, here is LR) and GT image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.LR_paths, self.GT_paths = None, None
        self.LR_env, self.GT_env = None, None  # environment for lmdb
        self.LR_size, self.GT_size = opt["LR_size"], opt["GT_size"]
        self.i = 0
        self.j = 0
        #############
        self.img_channel = opt["img_channel"]
        #############

        # read image list from lmdb or image files
        if opt["data_type"] == "lmdb":
            self.LR_paths, self.LR_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_LQ"]
            )
            self.GT_paths, self.GT_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_GT"]
            )
        elif opt["data_type"] == "img":
            self.LR_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_LQ"]
            )  # LR list
            self.GT_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_GT"]
            )  # GT list
        else:
            print("Error: data_type is not matched in Dataset")
        assert self.GT_paths, "Error: GT paths are empty."
        if self.LR_paths and self.GT_paths:
            assert len(self.LR_paths) == len(
                self.GT_paths
            ), "GT and LR datasets have different number of images - {}, {}.".format(
                len(self.LR_paths), len(self.GT_paths)
            )
        self.random_scale_list = [1]

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(
            self.opt["dataroot_GT"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.LR_env = lmdb.open(
            self.opt["dataroot_LQ"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def __getitem__(self, index):
        if self.opt["data_type"] == "lmdb":
            if (self.GT_env is None) or (self.LR_env is None):
                self._init_lmdb()

        GT_path, LR_path = None, None
        scale = self.opt["scale"] if self.opt["scale"] else 1
        GT_size = self.opt["GT_size"]
        LR_size = self.opt["LR_size"]

        # get GT image
        GT_path = self.GT_paths[index]
        if self.opt["data_type"] == "lmdb":
            resolution = [int(s) for s in self.GT_sizes[index].split("_")]
        else:
            resolution = None
        
        ########################################
        if self.opt["color"] == "RGB":
            img_GT = util.read_img(self.GT_env, GT_path, resolution)
        else:
            img_GT = util.read_tif(GT_path, self.opt["color"])
        ########################################
        # print(img_GT.shape)
        ########################################
        if img_GT.ndim == 2:
            img_GT = np.expand_dims(img_GT, axis=2)
        if img_GT.shape[2] != self.img_channel:
            raise ValueError(f"Expected {self.img_channel} bands, but got {img_GT.shape[2]} in {GT_path}")
        ########################################
        
        # modcrop in the validation / test phase
        if self.opt["phase"] != "train":
            img_GT = util.modcrop(img_GT, scale)

        # get LR image
        if self.LR_paths:  # LR exist
            LR_path = self.LR_paths[index]
            if self.opt["data_type"] == "lmdb":
                resolution = [int(s) for s in self.LR_sizes[index].split("_")]
            else:
                resolution = None
            
            if self.opt["color"] == "RGB":
                img_LR = util.read_img(self.LR_env, LR_path, resolution)
            else:
                img_LR = util.read_tif(LR_path, self.opt["color"])
            
            if img_LR.ndim == 2:
                img_LR = np.expand_dims(img_LR, axis=2)
            if img_LR.shape[2] != self.img_channel:
                raise ValueError(f"Expected {self.img_channel} bands, but got {img_LR.shape[2]} in {LR_path}")
        
        else:  # down-sampling on-the-fly
            # randomly scale during training
            if self.opt["phase"] == "train":
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_GT.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, GT_size)
                W_s = _mod(W_s, random_scale, scale, GT_size)
                img_GT = cv2.resize(
                    np.copy(img_GT), (W_s, H_s), interpolation=cv2.INTER_LINEAR
                )
                # force to 3 channels
                if img_GT.ndim == 2:
                    img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_GT.shape
            # using matlab imresize
            img_LR = util.imresize(img_GT, 1 / scale, True)
            if img_LR.ndim == 2:
                img_LR = np.expand_dims(img_LR, axis=2)

        if self.opt["phase"] == "train":
            H, W, C = img_LR.shape
            assert LR_size == GT_size // scale, "GT size does not match LR size"

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LR_size))
            rnd_w = random.randint(0, max(0, W - LR_size))
            img_LR = img_LR[rnd_h : rnd_h + LR_size, rnd_w : rnd_w + LR_size, :]
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[
                rnd_h_GT : rnd_h_GT + GT_size, rnd_w_GT : rnd_w_GT + GT_size, :
            ]

            # Save img_GT and img_LR before augmentation
            # self.save_augmented_images(img_GT, output_dir="../../../model_input_images/GT_before_aug")
            # self.save_augmented_images(img_LR, output_dir="../../../model_input_images/LR_before_aug")
            # augmentation - flip, rotate, etc.
            img_LR, img_GT = util.augment(
                [img_LR, img_GT],
                self.opt["use_noise"],
                self.opt["use_bright"],
                self.opt["use_blur"],
                self.opt["use_flip"],
                self.opt["use_rot"],
                self.opt["mode"],
                self.opt["use_swap"],
            )
        elif LR_size is not None:
            H, W, C = img_LR.shape
            assert LR_size == GT_size // scale, "GT size does not match LR size"

            if LR_size < H and LR_size < W:
                # center crop
                rnd_h = H // 2 - LR_size//2
                rnd_w = W // 2 - LR_size//2
                img_LR = img_LR[rnd_h : rnd_h + LR_size, rnd_w : rnd_w + LR_size, :]
                rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
                img_GT = img_GT[
                    rnd_h_GT : rnd_h_GT + GT_size, rnd_w_GT : rnd_w_GT + GT_size, :
                ]

        # change color space if necessary
        if self.opt["color"]:
            H, W, C = img_LR.shape
            img_LR = util.channel_convert(C, self.opt["color"], [img_LR])[
                0
            ]  # TODO during val no definition
            img_GT = util.channel_convert(img_GT.shape[2], self.opt["color"], [img_GT])[
                0
            ]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
        
        # Save img_GT and img_LR
        # print("shape of img_GT: ", img_GT.shape)
        # print("shape of img_LR: ", img_LR.shape)
        # self.save_augmented_images(img_GT, output_dir="../../../model_input_images/GT")
        # self.save_augmented_images(img_LR, output_dir="../../../model_input_images/LR")
        
        img_GT = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))
        ).float()
        img_LR = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))
        ).float()
        
        edges = self.edge_detection(img_GT)

        if LR_path is None:
            LR_path = GT_path

        return {"LQ": img_LR, "GT": img_GT, "LQ_path": LR_path, "GT_path": GT_path, "Edge": edges}
    # "Edge": None

    def __len__(self):
        return len(self.GT_paths)
    
    ###########################
    def save_augmented_images(self, img, output_dir="../../../model_input_images"):
        os.makedirs(output_dir, exist_ok=True)  
        if isinstance(img, list):  
            for im in img:
                if im.dtype == np.float32:
                    im = np.clip(im, 0, 1)
                filename = os.path.join(output_dir, f"augmented_{self.i}.png")
                cv2.imwrite(filename, (im * 255).astype(np.uint8))  
                print(f"Saved: {filename}")
                self.i += 1
        else:  
            # img is already a numpy array
            # img = np.transpose(img, (1, 2, 0))  # CHW to HWC
            filename = os.path.join(output_dir, "augmented.png")
            cv2.imwrite(filename, (img * 255).astype(np.uint8))  
            print(f"Saved: {filename}")
            
    def edge_detection(self, img):
        # edge detection 
        img_copy = img.clone()
        img_copy = img_copy.cpu().numpy()
        
        if img_copy.dtype != np.uint8:
            if img_copy.max() <= 1.0:
                img_copy = (img_copy * 255).clip(0, 255).astype(np.uint8)
            else:
                img_copy = img_copy.clip(0, 255).astype(np.uint8)
            
        # only for red channel
        # print("img shape: ", img.shape)
        # img_copy = img_copy[0, :, :]
        
        # RGB to grayscale
        img_copy = img_copy[:3, :, :]  # Extract the first three channels (RGB)
        img_copy = np.transpose(img_copy, (1, 2, 0))  # Convert from CHW to HWC format
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)  # Convert RGB to grayscale
        
        # # NIR
        # img_copy = img_copy[3, :, :]
        
        v = np.median(img_copy)
        
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        
        edges = cv2.Canny(img_copy, lower, upper)
        
        # Save the edge-detected image as a PNG
        if self.j < 10:
            output_dir = "../../../model_input_images/edges"
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"edge_{self.j}.png")
            cv2.imwrite(filename, edges)
            # print(f"Saved edge-detected image: {filename}")
            self.j += 1
        
        return edges
    ###########################
