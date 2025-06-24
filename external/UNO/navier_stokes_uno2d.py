# Codes for section: Results on Navier Stocks Equation (2D)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import matplotlib.pyplot as plt
from integral_operators import *
import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utilities3 import *
from Adam import Adam
import sys

torch.manual_seed(0)
np.random.seed(0)


# UNO model more aggressive domian contraction and expansion (factor of 1/2)
class UNO_P(nn.Module):
    def __init__(self, in_width, width, pad=0, factor=1):
        super(UNO_P, self).__init__()

        """
        The overall network. It contains 7 integral operator.
        1. Lift the input to the desire channel dimension by  self.fc, self.fc0 .
        2. 7 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        input: the solution of the first 10 timesteps (u(1), ..., u(10)).
        input shape: (batchsize, x=S, y=S, t=10)
        output: the solution of the next timesteps
        output shape: (batchsize, x=S, y=S, t=1)
        Here SxS is the spatial resolution
        in_width = 12 (10 input time steps + (x,y) location)
        with = uplifting dimension
        pad = padding the domian for non-periodic input
        factor = factor for scaling up/down the co-domain dimension at each integral operator
        """
        self.in_width = in_width  # input channel
        self.width = width

        self.padding = pad  # pad the domain if input is non-periodic

        self.fc = nn.Linear(self.in_width, self.width // 2)

        self.fc0 = nn.Linear(
            self.width // 2, self.width
        )  # input channel is 3: (a(x, y), x, y)

        self.L0 = OperatorBlock_2D(self.width, 2 * factor * self.width, 32, 32, 14, 14)

        self.L1 = OperatorBlock_2D(
            2 * factor * self.width, 4 * factor * self.width, 16, 16, 6, 6
        )

        self.L2 = OperatorBlock_2D(
            4 * factor * self.width, 8 * factor * self.width, 8, 8, 3, 3
        )

        self.L3 = OperatorBlock_2D(
            8 * factor * self.width, 8 * factor * self.width, 8, 8, 3, 3
        )

        self.L4 = OperatorBlock_2D(
            8 * factor * self.width, 4 * factor * self.width, 16, 16, 3, 3
        )

        self.L5 = OperatorBlock_2D(
            8 * factor * self.width, 2 * factor * self.width, 32, 32, 6, 6
        )

        self.L6 = OperatorBlock_2D(
            4 * factor * self.width, self.width, 64, 64, 14, 14
        )  # will be reshaped

        self.fc1 = nn.Linear(2 * self.width, 3 * self.width)
        # self.fc2 = nn.Linear(3 * self.width + self.width // 2, 1) # try
        self.fc2 = nn.Linear(3 * self.width + self.width // 2, 2)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x_fc = self.fc(x)
        x_fc = F.gelu(x_fc)

        x_fc0 = self.fc0(x_fc)
        x_fc0 = F.gelu(x_fc0)

        x_fc0 = x_fc0.permute(0, 3, 1, 2)
        x_fc0 = F.pad(x_fc0, [self.padding, self.padding, self.padding, self.padding])

        D1, D2 = x_fc0.shape[-2], x_fc0.shape[-1]

        x_c0 = self.L0(x_fc0, D1 // 2, D2 // 2)

        x_c1 = self.L1(x_c0, D1 // 4, D2 // 4)

        x_c2 = self.L2(x_c1, D1 // 8, D2 // 8)

        x_c3 = self.L3(x_c2, D1 // 8, D2 // 8)

        x_c4 = self.L4(x_c3, D1 // 4, D2 // 4)
        x_c4 = torch.cat([x_c4, x_c1], dim=1)

        x_c5 = self.L5(x_c4, D1 // 2, D2 // 2)
        x_c5 = torch.cat([x_c5, x_c0], dim=1)

        x_c6 = self.L6(x_c5, D1, D2)
        x_c6 = torch.cat([x_c6, x_fc0], dim=1)

        if self.padding != 0:
            x_c6 = x_c6[..., self.padding : -self.padding, self.padding : -self.padding]

        x_c6 = x_c6.permute(0, 2, 3, 1)

        x_fc1 = self.fc1(x_c6)
        x_fc1 = F.gelu(x_fc1)

        x_fc1 = torch.cat([x_fc1, x_fc], dim=3)
        x_out = self.fc2(x_fc1)

        return x_out

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 2 * np.pi, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 2 * np.pi, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat(
            (torch.sin(gridx), torch.sin(gridy), torch.cos(gridx), torch.cos(gridy)),
            dim=-1,
        ).to(device)


#####
# UNO model
# it has less aggressive scaling factors for domains and co-domains.
#####
class UNO(nn.Module):
    def __init__(self, in_width, width, pad=0, factor=3 / 4):
        super(UNO, self).__init__()

        self.in_width = in_width  # input channel
        self.width = width
        self.factor = factor
        self.padding = pad
        
        print("in_width", in_width)
        print("width", width)
        print("factor", factor)

        self.fc = nn.Linear(self.in_width + 4, self.width // 2)

        self.fc0 = nn.Linear(
            self.width // 2, self.width
        )  # input channel is 3: (a(x, y), x, y)

        self.L0 = OperatorBlock_2D(self.width, 2 * factor * self.width, 48, 48, 22, 22)

        self.L1 = OperatorBlock_2D(
            2 * factor * self.width, 4 * factor * self.width, 32, 32, 14, 14
        )

        self.L2 = OperatorBlock_2D(
            4 * factor * self.width, 8 * factor * self.width, 16, 16, 6, 6
        )

        self.L3 = OperatorBlock_2D(
            8 * factor * self.width, 16 * factor * self.width, 8, 8, 4, 4
        )

        # self.L3 = OperatorBlock_2D(
        #     8 * factor * self.width, 8 * factor * self.width, 16, 16, 6, 6
        # )

        # self.L4 = OperatorBlock_2D(
        #     8 * factor * self.width, 4 * factor * self.width, 32, 32, 6, 6
        # )

        # self.L5 = OperatorBlock_2D(
        #     8 * factor * self.width, 2 * factor * self.width, 48, 48, 14, 14
        # )

        # self.L6 = OperatorBlock_2D(
        #     4 * factor * self.width, self.width, 64, 64, 22, 22
        # )  # will be reshaped

        # self.fc1 = nn.Linear(2 * self.width, 4 * self.width)
        # self.fc2 = nn.Linear(4 * self.width, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x_fc = self.fc(x)
        x_fc = F.gelu(x_fc)

        x_fc0 = self.fc0(x_fc)
        x_fc0 = F.gelu(x_fc0)

        x_fc0 = x_fc0.permute(0, 3, 1, 2)

        x_fc0 = F.pad(x_fc0, [self.padding, self.padding, self.padding, self.padding])

        D1, D2 = x_fc0.shape[-2], x_fc0.shape[-1]

        x_c0 = self.L0(x_fc0, int(D1 * self.factor), int(D2 * self.factor))
        x_c1 = self.L1(x_c0, D1 // 2, D2 // 2)

        x_c2 = self.L2(x_c1, D1 // 4, D2 // 4)
        x_c3 = self.L3(x_c2, D1 // 8, D2 // 8)
        # x_c4 = self.L4(x_c3, D1 // 2, D2 // 2)
        # x_c4 = torch.cat([x_c4, x_c1], dim=1)
        # x_c5 = self.L5(x_c4, int(D1 * self.factor), int(D2 * self.factor))
        # x_c5 = torch.cat([x_c5, x_c0], dim=1)
        # x_c6 = self.L6(x_c5, D1, D2)
        # x_c6 = torch.cat([x_c6, x_fc0], dim=1)

        # if self.padding != 0:
        #     x_c6 = x_c6[..., : -self.padding, : -self.padding]

        # x_c6 = x_c6.permute(0, 2, 3, 1)

        # x_fc1 = self.fc1(x_c6)
        # x_fc1 = F.gelu(x_fc1)

        # x_out = self.fc2(x_fc1)

        return x_c0, x_c1, x_c2, x_c3

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 2 * np.pi, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 2 * np.pi, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat(
            (torch.sin(gridx), torch.sin(gridy), torch.cos(gridx), torch.cos(gridy)),
            dim=-1,
        ).to(device)


#####
# UNO HiLoc model (whole image and pacthes)
# it has less aggressive scaling factors for domains and co-domains.
# ####
class UNO_HiLoc(nn.Module):
    def __init__(self, in_width, width, pad=0, factor=3 / 4, galerkin_encorders=None):
        super(UNO, self).__init__()

        self.in_width = in_width  # input channel
        self.width = width
        self.factor = factor
        self.padding = pad
        self.galerkin_encorders = galerkin_encorders
        
        print("in_width", in_width)
        print("width", width)
        print("factor", factor)

        self.fc = nn.Linear(self.in_width + 4, self.width // 2)

        self.fc0 = nn.Linear(
            self.width // 2, self.width
        )  # input channel is 3: (a(x, y), x, y)
        
        def make_Layer(width, factor, input_factor, output_factor, dim1, dim2, mode1, mode2):
            return (
                OperatorBlock_2D(
                    int(input_factor * factor * width), int(output_factor * factor * width), dim1, dim2, mode1, mode2
                ),
                OperatorBlock_2D(
                    int(input_factor * factor * width), int(output_factor * factor * width), dim1, dim2, mode1, mode2, Apply_linear_transform=False
                ),
            )

        self.L0_whole = OperatorBlock_2D(self.width, int(2 * factor * self.width), 48, 48, 22, 22)
        self.L0_patches = OperatorBlock_2D(self.width, int(2 * factor * self.width), 48, 48, 22, 22, Apply_linear_transform=False)
        
        self.L1_whole, self.L1_patches = make_Layer(self.width, self.factor, 2, 4, 32, 32, 14, 14)
        self.L2_whole, self.L2_patches = make_Layer(self.width, self.factor, 4, 8, 16, 16, 6, 6)
        self.L3_whole, self.L3_patches = make_Layer(self.width, self.factor, 8, 16, 8, 8, 4, 4)
        # self.L4_whole, self.L4_patches = make_Layer(self.width, self.factor, 16, 16, 8, 8, 4, 4)
        # self.L5_whole, self.L5_patches = make_Layer(self.width, self.factor, 16, 8, 16, 16, 6, 6)
        # self.L6_whole, self.L6_patches = make_Layer(self.width, self.factor, 8, 4, 32, 32, 6, 6)
        # self.L7_whole, self.L7_patches = make_Layer(self.width, self.factor, 4, 2, 48, 48, 14, 14)
        # self.L8_whole, self.L8_patches = make_Layer(self.width, self.factor, 2, 1, 64, 64, 22, 22)

        # self.fc1 = nn.Linear(2 * self.width, 4 * self.width)
        # self.fc2 = nn.Linear(4 * self.width, 8)

    def forward(self, x_whole):
        #make patches
        x_whole_tmp = x_whole.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        x_patches = self.make_patches(x_whole_tmp)  # (B, C, H, W) -> (B, H, W, C)
        for i in range(len(x_patches)):
            x_patches[i] = x_patches[i].permute(0, 2, 3, 1)
            
        #layer 0 whole and patches
        # whole image process
        grid_whole = self.get_grid(x_whole.shape, x_whole.device)
        x_whole = torch.cat((x_whole, grid_whole), dim=-1) #B, H, W, C
        
        x_fc0_whole, D1_whole, D2_whole = self.forward_lift(x_whole)
        x_c0_whole = self.L0_whole(x_fc0_whole, int(D1_whole * self.factor), int(D2_whole * self.factor))      
        
        #patch process
        x_fc0_patches = []
        x_c_patches = [[] for _ in range(9)]  # 9 layers if needed

        for i in range(len(x_patches)):
            grid_patch = self.get_grid(x_patches[i].shape, x_patches[i].device)
            patch_input = torch.cat((x_patches[i], grid_patch), dim=-1)
            x_fc0_patch, _, _ = self.forward_lift(patch_input)
            x_fc0_patches.append(x_fc0_patch)
        assert len(x_fc0_patches) == 16, f"Expected 16 x_fc0_patches, got {len(x_fc0_patches)}"
        D1_patch, D2_patch = x_fc0_patches[0].shape[-2], x_fc0_patches[0].shape[-1]

        dim1, dim2 = int(D1_patch * self.factor), int(D2_patch * self.factor)
        x_c_patches[0] = self.process_integral_operator_patches(self.L0_patches, x_fc0_patches, dim1, dim2)
        assert len(x_c_patches[0]) == 16, f"Expected 16 patches after L0, got {len(x_c_patches[0])}"

        x_c0 = self.combine_whole_and_patches(x_c0_whole, x_c_patches[0])
        x_c0 = self.galerkin_process(self.galerkin_encorders[0], x_c0)
        
        #layer 1
        x_c0_patches = self.make_patches(x_c0)
        x_c1_whole = self.L1_whole(x_c0, D1_whole // 2, D2_whole // 2)
        dim1, dim2 = int(D1_patch // 2), int(D2_patch // 2)
        x_c_patches[1] = self.process_integral_operator_patches(self.L1_patches, x_c0_patches, dim1, dim2)
        x_c1 = self.combine_whole_and_patches(x_c1_whole, x_c_patches[1])
        x_c1 = self.galerkin_process(self.galerkin_encorders[1], x_c1)
        
        #layer 2
        x_c1_patches = self.make_patches(x_c1)
        x_c2_whole = self.L2_whole(x_c1, D1_whole // 4, D2_whole // 4)
        dim1, dim2 = int(D1_patch // 4), int(D2_patch // 4)
        x_c_patches[2] = self.process_integral_operator_patches(self.L2_patches, x_c1_patches, dim1, dim2)
        x_c2 = self.combine_whole_and_patches(x_c2_whole, x_c_patches[2])
        x_c2 = self.galerkin_process(self.galerkin_encorders[2], x_c2)
    
        #layer 3
        x_c2_patches = self.make_patches(x_c2)
        x_c3_whole = self.L3_whole(x_c2, D1_whole // 8, D2_whole // 8)
        dim1, dim2 = int(D1_patch // 8), int(D2_patch // 8)
        x_c_patches[3] = self.process_integral_operator_patches(self.L3_patches, x_c2_patches, dim1, dim2)
        x_c3 = self.combine_whole_and_patches(x_c3_whole, x_c_patches[3])
        x_c3 = self.galerkin_process(self.galerkin_encorders[3], x_c3)
        
        return x_c0, x_c1, x_c2, x_c3
    
        # x_c4 = self.L4(x_c3, D1 // 2, D2 // 2)
        # x_c4 = torch.cat([x_c4, x_c1], dim=1)
        # x_c5 = self.L5(x_c4, int(D1 * self.factor), int(D2 * self.factor))
        # x_c5 = torch.cat([x_c5, x_c0], dim=1)
        # x_c6 = self.L6(x_c5, D1, D2)
        # x_c6 = torch.cat([x_c6, x_fc0], dim=1)

        # if self.padding != 0:
        #     x_c6 = x_c6[..., : -self.padding, : -self.padding]

        # x_c6 = x_c6.permute(0, 2, 3, 1) 

        # x_fc1 = self.fc1(x_c6)
        # x_fc1 = F.gelu(x_fc1)

        # x_out = self.fc2(x_fc1)

        # return x_out

    def forward_lift(self, x):
        """
        Lift the input to the desired channel dimension.
        :param x: input tensor of shape (B, H, W, C)
        """
        x_fc = self.fc(x)
        x_fc = F.gelu(x_fc)
        
        x_fc0 = self.fc0(x_fc)
        x_fc0 = F.gelu(x_fc0)
        x_fc0 = x_fc0.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
        
        x_fc0 = F.pad(x_fc0, [self.padding] * 4)  
        D1, D2 = x_fc0.shape[-2], x_fc0.shape[-1]
        return x_fc0, D1, D2

    def combine_whole_and_patches(self, x_c_whole, x_c_patche):
        rows = [torch.cat(x_c_patche[i*4:(i+1)*4], dim=3) for i in range(4)]  # concat width
        x_c_patches_all = torch.cat(rows, dim=2)  # concat height
        
        assert x_c_whole.shape == x_c_patches_all.shape, f"Expected shapes to match, got {x_c_whole.shape} and {x_c_patches_all.shape}"
        x_c = torch.cat([x_c_whole, x_c_patches_all], dim=1)
        print("x_c_combine", x_c.shape)
        return x_c    
    
    def galerkin_process(self, galerkin_encoder, x_c):
        B, C, H, W = x_c.shape
        x_c_gelu = F.gelu(x_c)
        garlerkin_encorder_input = x_c_gelu.permute(0, 2, 3, 1).reshape(B, H * W, C)        
        pos = self.get_2d_pos_encoding(B, H, W, x_c.device)
        
        galerkin_encorder_out = galerkin_encoder(node=garlerkin_encorder_input,edge=None, pos=pos)  
        C = int(C / 2)
        x_c = galerkin_encorder_out.permute(0, 2, 1).reshape(B, C, H, W)  # (B, C, H, W)
        print("x_c_galerkin", x_c.shape)
        assert x_c.shape == (B, C, H, W), f"Expected shape {(B, C, H, W)}, got {x_c.shape}"
        return x_c  
    
    def process_integral_operator_patches(self, L_patches, x_fc0_patches, dim1, dim2):
        x_c_patches = []
        for i in range(16):
            patch_out = L_patches(
                x_fc0_patches[i],
                dim1,
                dim2,
            )
            x_c_patches.append(patch_out)
        return x_c_patches

    def make_patches(self, x_c):
        B, C, H, W = x_c.shape
        x_c_patches = []
        x_c_height = x_c.chunk(4, dim=2)
        for row in x_c_height:
            x_c_patches.extend(row.chunk(4, dim=3))
            assert x_c_patches[-1].shape == (B, C, H // 4, W // 4), f"Expected shape {(B, C, H // 4, W // 4)}, got {x_c_patches[-1].shape}"
        return x_c_patches
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 2 * np.pi, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 2 * np.pi, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat(
            (torch.sin(gridx), torch.sin(gridy), torch.cos(gridx), torch.cos(gridy)),
            dim=-1,
        ).to(device)

    def get_2d_pos_encoding(self, B, H, W, device):
        gridx = torch.linspace(0, 1, W, device=device)
        gridy = torch.linspace(0, 1, H, device=device)
        gridx, gridy = torch.meshgrid(gridx, gridy, indexing="xy")
        grid = torch.stack((gridx, gridy), dim=-1)  # [H, W, 2]
        grid = grid.reshape(-1, 2)  # [H*W, 2]
        grid = grid.unsqueeze(0).repeat(B, 1, 1)  # [B, H*W, 2]
        return grid 


###
# UNO for high resolution (256x256) navier stocks simulations
###


class UNO_S256(nn.Module):
    def __init__(self, in_width, width, pad=0, factor=1):
        super(UNO_S256, self).__init__()
        self.in_width = in_width  # input channel
        self.width = width

        self.padding = pad  # pad the domain if input is non-periodic

        # self.fc = nn.Linear(self.in_width, 16)
        self.fc = nn.Linear(self.in_width + 4, self.width // 2)

        self.fc0 = nn.Linear(16, self.width)  # input channel is 3: (a(x, y), x, y)

        self.L0 = OperatorBlock_2D(self.width, 2 * factor * self.width, 64, 64, 32, 33)

        self.L1 = OperatorBlock_2D(
            2 * factor * self.width, 4 * factor * self.width, 16, 16, 8, 9
        )

        self.L2 = OperatorBlock_2D(
            4 * factor * self.width, 8 * factor * self.width, 8, 8, 4, 5
        )

        self.L3 = OperatorBlock_2D(
            8 * factor * self.width, 8 * factor * self.width, 8, 8, 4, 5
        )

        self.L4 = OperatorBlock_2D(
            8 * factor * self.width, 4 * factor * self.width, 16, 16, 4, 5
        )

        self.L5 = OperatorBlock_2D(
            8 * factor * self.width, 2 * factor * self.width, 64, 64, 8, 9
        )

        self.L6 = OperatorBlock_2D(
            4 * factor * self.width, self.width, 256, 256, 32, 32
        )  # will be reshaped

        self.fc1 = nn.Linear(2 * self.width, 3 * self.width)
        self.fc2 = nn.Linear(3 * self.width + 16, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x_fc = self.fc(x)
        x_fc = F.gelu(x_fc)

        x_fc0 = self.fc0(x_fc)
        x_fc0 = F.gelu(x_fc0)

        x_fc0 = x_fc0.permute(0, 3, 1, 2)
        x_fc0 = F.pad(x_fc0, [self.padding, self.padding, self.padding, self.padding])

        D1, D2 = x_fc0.shape[-2], x_fc0.shape[-1]

        x_c0 = self.L0(x_fc0, D1 // 4, D2 // 4)
        x_c1 = self.L1(x_c0, D1 // 16, D2 // 16)
        x_c2 = self.L2(x_c1, D1 // 32, D2 // 32)
        x_c3 = self.L3(x_c2, D1 // 32, D2 // 32)
        x_c4 = self.L4(x_c3, D1 // 16, D2 // 16)
        x_c4 = torch.cat([x_c4, x_c1], dim=1)
        x_c5 = self.L5(x_c4, D1 // 4, D2 // 4)
        x_c5 = torch.cat([x_c5, x_c0], dim=1)

        x_c6 = self.L6(x_c5, D1, D2)
        # print(x.shape)
        x_c6 = torch.cat([x_c6, x_fc0], dim=1)

        if self.padding != 0:
            x_c6 = x_c6[..., self.padding : -self.padding, self.padding : -self.padding]

        x_c6 = x_c6.permute(0, 2, 3, 1)

        x_fc1 = self.fc1(x_c6)
        x_fc1 = F.gelu(x_fc1)

        x_fc1 = torch.cat([x_fc1, x_fc], dim=3)
        x_out = self.fc2(x_fc1)

        return x_out

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 2 * np.pi, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 2 * np.pi, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat(
            (torch.sin(gridx), torch.sin(gridy), torch.cos(gridx), torch.cos(gridy)),
            dim=-1,
        ).to(device)
