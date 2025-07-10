import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
import copy
import sys
from einops import rearrange, einsum
sys.path.append('/home/yuki/research/EDiffSR/external/galerkin_transformer')
from galerkin_attention import simple_attn
# import sys
# import os
# sys.path.append(os.path.join(os.path.dirname(__file__), "..", "galerkin_transformer"))
# from layers import SimpleAttention


class SpectralConv1d_Uno(nn.Module):
    def __init__(self, in_codim, out_codim, dim1, modes1=None):
        super(SpectralConv1d_Uno, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT. 
        dim1 = Default output grid size along x (or 1st dimension of output domain) 
        Ratio of grid size of the input and the output implecitely 
        set the expansion or contraction farctor along each dimension of the domain.
        modes1 = Number of fourier modes to consider for the integral operator.
                Number of modes must be compatibale with the input grid size 
                and desired output grid size.
                i.e., modes1 <= min( dim1/2, input_dim1/2). 
                Here "input_dim1" is the grid size along x axis (or first dimension) of the input domain.
        in_codim = Input co-domian dimension
        out_codim = output co-domain dimension
        """
        in_codim = int(in_codim)
        out_codim = int(out_codim)
        self.in_channels = in_codim
        self.out_channels = out_codim
        self.dim1 = dim1  # output dimensions
        if modes1 is not None:
            self.modes1 = (
                modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
            )
        else:
            self.modes1 = dim1 // 2

        self.scale = (1 / (2 * in_codim)) ** (1.0 / 2.0)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.randn(in_codim, out_codim, self.modes1, dtype=torch.cfloat)
        )

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x, dim1=None):
        """
        input shape = (batch, in_codim, input_dim1)
        output shape = (batch, out_codim, dim1)
        """
        if dim1 is not None:
            self.dim1 = dim1
        batchsize = x.shape[0]

        x_ft = torch.fft.rfft(x, norm="forward")

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            self.dim1 // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(
            x_ft[:, :, : self.modes1], self.weights1
        )

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=self.dim1, norm="forward")
        return x


class pointwise_op_1D(nn.Module):
    """
    All variables are consistent with the SpectralConv1d_Uno class.
    """

    def __init__(self, in_codim, out_codim, dim1):
        super(pointwise_op_1D, self).__init__()
        self.conv = nn.Conv1d(int(in_codim), int(out_codim), 1)
        self.dim1 = int(dim1)

    def forward(self, x, dim1=None):
        if dim1 is None:
            dim1 = self.dim1
        x_out = self.conv(x)

        x_out = torch.nn.functional.interpolate(
            x_out, size=dim1, mode="linear", align_corners=True, antialias=True
        )
        return x_out


class OperatorBlock_1D(nn.Module):
    """
    Normalize = if true performs InstanceNorm1d on the output.
    Non_Lin = if true, applies point wise nonlinearity.
    All other variables are consistent with the SpectralConv1d_Uno class.
    """

    def __init__(self, in_codim, out_codim, dim1, modes1, Normalize=True, Non_Lin=True):
        super(OperatorBlock_1D, self).__init__()
        self.conv = SpectralConv1d_Uno(in_codim, out_codim, dim1, modes1)
        self.w = pointwise_op_1D(in_codim, out_codim, dim1)
        self.normalize = Normalize
        self.non_lin = Non_Lin
        if Normalize:
            self.normalize_layer = torch.nn.InstanceNorm1d(int(out_codim), affine=True)

    def forward(self, x, dim1=None):
        """
        input shape = (batch, in_codim, input_dim1)
        output shape = (batch, out_codim, dim1)
        """
        x1_out = self.conv(x, dim1)
        x2_out = self.w(x, dim1)
        x_out = x1_out + x2_out
        if self.normalize:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out


class SpectralConv2d_Uno(nn.Module):
    def __init__(self, in_codim, out_codim, dim1, dim2, modes1=None, modes2=None):
        super(SpectralConv2d_Uno, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT. 
        dim1 = Default output grid size along x (or 1st dimension of output domain) 
        dim2 = Default output grid size along y ( or 2nd dimension of output domain)
        Ratio of grid size of the input and the output implecitely 
        set the expansion or contraction farctor along each dimension.
        modes1, modes2 = Number of fourier modes to consider for the ontegral operator
                        Number of modes must be compatibale with the input grid size 
                        and desired output grid size.
                        i.e., modes1 <= min( dim1/2, input_dim1/2). 
                        Here "input_dim1" is the grid size along x axis (or first dimension) of the input domain.
                        Other modes also the have same constrain.
        in_codim = Input co-domian dimension
        out_codim = output co-domain dimension
        """

        in_codim = int(in_codim)
        out_codim = int(out_codim)
        self.in_channels = in_codim
        self.out_channels = out_codim
        self.dim1 = dim1
        self.dim2 = dim2
        if modes1 is not None:
            self.modes1 = modes1
            self.modes2 = modes2
        else:
            self.modes1 = dim1 // 2 - 1
            self.modes2 = dim2 // 2
        self.scale = (1 / (2 * in_codim)) ** (1.0 / 2.0)
        self.weights1 = nn.Parameter(
            self.scale
            * (
                torch.randn(
                    in_codim, out_codim, self.modes1, self.modes2, dtype=torch.cfloat
                )
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * (
                torch.randn(
                    in_codim, out_codim, self.modes1, self.modes2, dtype=torch.cfloat
                )
            )
        )

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        result = torch.einsum("bixy,ioxy->boxy", input, weights)
        return result

    def forward(self, x, dim1=None, dim2=None):
        if dim1 is not None:
            self.dim1 = dim1
            self.dim2 = dim2
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x, norm="forward")
                
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            self.dim1,
            self.dim2 // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        #LL-HL -> LL-LH
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
         
        out_ft[:, :, :self.modes1, -self.modes2:] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, -self.modes2:], self.weights2
        )
        
        # out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
        #     x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        # )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(self.dim1, self.dim2), norm="forward")
        return x


class pointwise_op_2D(nn.Module):
    """
    dim1 = Default output grid size along x (or 1st dimension)
    dim2 = Default output grid size along y ( or 2nd dimension)
    in_codim = Input co-domian dimension
    out_codim = output co-domain dimension
    """

    def __init__(self, in_codim, out_codim, dim1, dim2):
        super(pointwise_op_2D, self).__init__()
        self.conv = nn.Conv2d(int(in_codim), int(out_codim), 1)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)

    def forward(self, x, dim1=None, dim2=None):
        """
        input shape = (batch, in_codim, input_dim1,input_dim2)
        output shape = (batch, out_codim, dim1,dim2)
        """
        if dim1 is None:
            dim1 = self.dim1
            dim2 = self.dim2
        x_out = self.conv(x)

        # ft = torch.fft.rfft2(x_out)
        # ft_u = torch.zeros_like(ft)
        # ft_u[:dim1//2-1,:dim2//2-1] = ft[:dim1//2-1,:dim2//2-1]
        # ft_u[-(dim1//2-1):,:dim2//2-1] = ft[-(dim1//2-1):,:dim2//2-1]
        # x_out = torch.fft.irfft2(ft_u)
        x_out = torch.nn.functional.interpolate(
            x_out, size=(dim1, dim2), mode="bicubic", align_corners=True, antialias=True
        )
        return x_out

# class SpectralConv2d_UnoHiLoc(nn.Module):
#     def __init__(
#     self,
#     in_codim: int,
#     out_codim: int,
#     dim1: int,
#     dim2: int,
#     modes1: int = None,
#     modes2: int = None,
#     trunc_mode: str = None, # None, LL-LH, LH-HH, shared_sliding
#     patch_based: bool = False, 
#     factorize_mode: str = None # None, dep-sep: Depth-wise Separable Convolution
# ):    
#         super(SpectralConv2d_UnoHiLoc, self).__init__()

#         """
#         2D Fourier layer. It does FFT, linear transform, and Inverse FFT. 
#         dim1 = Default output grid size along x (or 1st dimension of output domain) 
#         dim2 = Default output grid size along y ( or 2nd dimension of output domain)
#         Ratio of grid size of the input and the output implecitely 
#         set the expansion or contraction farctor along each dimension.
#         modes1, modes2 = Number of fourier modes to consider for the ontegral operator
#                         Number of modes must be compatibale with the input grid size 
#                         and desired output grid size.
#                         i.e., modes1 <= min( dim1/2, input_dim1/2). 
#                         Here "input_dim1" is the grid size along x axis (or first dimension) of the input domain.
#                         Other modes also the have same constrain.
#         in_codim = Input co-domian dimension
#         out_codim = output co-domain dimension
#         """

#         in_codim = int(in_codim)
#         out_codim = int(out_codim)
#         self.in_channels = in_codim
#         self.out_channels = out_codim
#         self.dim1 = dim1
#         self.dim2 = dim2
#         if modes1 is not None:
#             self.modes1 = modes1
#             self.modes2 = modes2
#         else:
#             self.modes1 = dim1 // 2 - 1
#             self.modes2 = dim2 // 2
#         self.scale = (1 / (2 * in_codim)) ** (1.0 / 2.0)
#         self.trunc_mode = trunc_mode
#         self.patch_based = patch_based
#         self.factorize_mode = factorize_mode
#         # Enforce mutual exclusivity
#         if patch_based and trunc_mode is not None:
#             raise ValueError("Cannot enable both patch_based=True and trunc_mode is not None")
#         assert factorize_mode is None or factorize_mode == "dep-sep", f"factorize_mode should be either real None or str(dep-sep), not {factorize_mode}"
#         if patch_based or trunc_mode == "shared_sliding":
#             if factorize_mode is None:
#                 self.weights = nn.Parameter(
#                     self.scale * torch.randn(
#                         in_codim, out_codim, self.modes1, self.modes2, dtype=torch.cfloat
#                     )
#                 )
#             elif factorize_mode == "dep-sep":
#                 self.depthwise_weights = nn.Parameter(
#                     self.scale * torch.randn(
#                         in_codim, self.modes1, self.modes2, dtype=torch.cfloat
#                     )
#                 )
#                 self.pointwise_weights = nn.Parameter(
#                     self.scale * torch.randn(
#                         in_codim, out_codim, dtype=torch.cfloat
#                     )
#                 )
#         elif trunc_mode in ["LL-LH", "LH-HH"]:
#             if factorize_mode is None:
#                 self.weights1 = nn.Parameter(
#                     self.scale * torch.randn(
#                         in_codim, out_codim, self.modes1, self.modes2, dtype=torch.cfloat
#                     )
#                 )
#                 self.weights2 = nn.Parameter(
#                     self.scale * torch.randn(
#                         in_codim, out_codim, self.modes1, self.modes2, dtype=torch.cfloat
#                     )
#                 )
#             elif factorize_mode == "dep-sep":
#                 # Depthwise weights: One set of weights per input channel (in_codim filters, each operating on one input channel)
#                 self.depthwise_weights1 = nn.Parameter(
#                     self.scale * torch.randn(
#                         in_codim, self.modes1, self.modes2, dtype=torch.cfloat
#                     )
#                 )
#                 self.depthwise_weights2 = nn.Parameter(
#                     self.scale * torch.randn(
#                         in_codim, self.modes1, self.modes2, dtype=torch.cfloat
#                     )
#                 )
#                 # Pointwise weights: Combine the in_codim channels into out_codim channels (1x1 operation in spectral domain)
#                 self.pointwise_weights1 = nn.Parameter(
#                     self.scale * torch.randn(
#                         in_codim, out_codim, dtype=torch.cfloat
#                     )
#                 )
#                 self.pointwise_weights2 = nn.Parameter(
#                     self.scale * torch.randn(
#                         in_codim, out_codim, dtype=torch.cfloat
#                     )
#                 )

#     # Complex multiplication
#     def compl_mul2d(self, inputs, weights, depthwise_weights=None, pointwise_weights=None):
#         if self.trunc_mode == "shared_sliding" :
#             # Non-overlapping sliding mode
#             batchsize, channel, inputs_dim1, inputs_dim2 = inputs.shape
#             numpx = inputs_dim1 // self.modes1 # number of patches along x dim
#             numpy = inputs_dim2 // self.modes2 # number of patches along y dim
#             # Assert input dimension are divisible by modes
#             assert inputs_dim1 % self.modes1 == 0, f"input_dim1 ({inputs_dim1}) not divisible for ({self.modes1})"
#             assert inputs_dim2 % self.modes2 == 0, f"input_dim1 ({inputs_dim2}) not divisible for ({self.modes2})"
#             # Reshape inputs into patches
#             inputs_patched = rearrange(
#                 inputs,
#                 'b c (numpx sizepx) (numpy sizepy) -> (b numpx numpy) c sizepx sizepy',
#                 numpx=numpx,
#                 numpy=numpy,
#                 sizepx=self.modes1,
#                 sizepy=self.modes2
#             )
#             if self.factorize_mode is None:
#                 out = einsum(
#                     inputs_patched,
#                     weights,
#                     'b_numpx_numpy in_c sizepx sizepy, in_c out_c sizepx sizepy -> b_numpx_numpy out_c sizepx sizepy'
#                 )
#             elif self.factorize_mode == "dep-sep":
#                 out_depthwise = einsum(
#                     inputs_patched,
#                     depthwise_weights,
#                     'b_numpx_numpy in_c sizepx sizepy, in_c sizepx sizepy -> b_numpx_numpy in_c sizepx sizepy'
#                 )
#                 out = einsum(
#                     out_depthwise,
#                     pointwise_weights,
#                     'b_numpx_numpy in_c sizepx sizepy, in_c out_c -> b_numpx_numpy out_c sizepx sizepy'
#                 )
#             out = rearrange(
#                 out,
#                 '(b numpx numpy) out_c sizepx sizepy -> b out_c (numpx sizepx) (numpy sizepy)',
#                 numpx=numpx,
#                 numpy=numpy,
#             )
#             return out
#         else:
#             if self.factorize_mode is None:
#                 # For non-sliding modes, keep the original einsum
#                 out = einsum(
#                     inputs,
#                     weights,
#                     'b i x y, i o x y-> b o x y'
#                 )
#                 return out
#             elif self.factorize_mode == "dep-sep":
#                 out_depthwise = einsum(
#                     inputs,
#                     depthwise_weights,
#                     'b i x y, i x y -> b i x y'
#                 )
#                 out = einsum(
#                     out_depthwise,
#                     pointwise_weights,
#                     'b i x y, i o -> b o x y'
#                 )
#                 return out

#     def forward(self, x, dim1=None, dim2=None):
#         if dim1 is None or dim2 is None:
#             dim1 = self.dim1
#             dim2 = self.dim2
#         batchsize, in_channel, input_dim1, input_dim2 = x.shape

#         if not self.patch_based:
#             # Compute Fourier coeffcients up to factor of e^(- something constant)
#             x_ft = torch.fft.rfft2(x, norm="forward")
#             # Multiply relevant Fourier modes
#             out_ft = torch.zeros(
#                 batchsize,
#                 self.out_channels,
#                 input_dim1,
#                 input_dim2 // 2 + 1,
#                 dtype=torch.cfloat,
#                 device=x.device,
#             )
#             if self.trunc_mode == "LL-LH":
#                 if self.factorize_mode is None:
#                     out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
#                         x_ft[:, :, :self.modes1, :self.modes2], weights=self.weights1
#                     )
#                     out_ft[:, :, :self.modes1, -self.modes2: ] = self.compl_mul2d(
#                         x_ft[:, :, :self.modes1, -self.modes2: ], weights=self.weights2
#                     )
#                 elif self.factorize_mode == "dep-sep":
#                     out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
#                         x_ft[:, :, :self.modes1, :self.modes2], weights=None, 
#                         depthwise_weights=self.depthwise_weights1, pointwise_weights=self.pointwise_weights1
#                     )
#                     out_ft[:, :, :self.modes1, -self.modes2: ] = self.compl_mul2d(
#                         x_ft[:, :, :self.modes1, -self.modes2: ], weights=None, 
#                         depthwise_weights=self.depthwise_weights2, pointwise_weights=self.pointwise_weights2
#                     )
#             elif self.trunc_mode == "LH-HH":
#                 if self.factorize_mode is None:
#                     out_ft[:, :, :self.modes1, -self.modes2:] = self.compl_mul2d(
#                         x_ft[:, :, :self.modes1, -self.modes2:], weights=self.weights1
#                     )
#                     out_ft[:, :, -self.modes1:, -self.modes2: ] = self.compl_mul2d(
#                         x_ft[:, :, -self.modes1:, -self.modes2: ], weights=self.weights2
#                     )
#                 elif self.factorize_mode == "dep-sep":
#                     out_ft[:, :, :self.modes1, -self.modes2:] = self.compl_mul2d(
#                         x_ft[:, :, :self.modes1, -self.modes2:], weights=None, 
#                         depthwise_weights=self.depthwise_weights1, pointwise_weights=self.pointwise_weights1
#                     )
#                     out_ft[:, :, -self.modes1:, -self.modes2: ] = self.compl_mul2d(
#                         x_ft[:, :, -self.modes1:, -self.modes2: ], weights=None, 
#                         depthwise_weights=self.depthwise_weights2, pointwise_weights=self.pointwise_weights2
#                     )
#             elif self.trunc_mode == "shared_sliding":
#                 if self.factorize_mode is None:
#                     out_ft[:, :, :input_dim1, :(input_dim2 // 2)] = self.compl_mul2d(x_ft[:, :, :input_dim1, :(input_dim2 // 2)], weights=self.weights) # no +1 cause we deliberately omit the last column of last dim
#                 elif self.factorize_mode == "dep-sep":
#                     out_ft[:, :, :input_dim1, :(input_dim2 // 2)] = self.compl_mul2d(x_ft[:, :, :input_dim1, :(input_dim2 // 2)], weights=None, 
#                     depthwise_weights=self.depthwise_weights, pointwise_weights=self.pointwise_weights) # no +1 cause we deliberately omit the last column of last dim
#             # Return to physical space
#             x = torch.fft.irfft2(out_ft, s=(input_dim1, input_dim2), norm="forward")
#             if self.dim1 != input_dim1 or self.dim2 != input_dim2:
#                 x = F.interpolate(
#                     x, size=(self.dim1, self.dim2), mode='bicubic', align_corners=True, antialias=True
#                 )
#             return x
#         elif self.patch_based:
#             # now each input is a batch of patches : B*M*num_patches, C, patch_size, patch_size
#             x_ft = torch.fft.rfft2(x, norm="forward")
#             # Multiply relevant Fourier modes
#             out_ft = torch.zeros(
#                 batchsize,
#                 self.out_channels,
#                 input_dim1,
#                 input_dim2 // 2 + 1,
#                 dtype=torch.cfloat,
#                 device=x.device,
#             )
#             if self.factorize_mode is None:
#                 out_ft = self.compl_mul2d(x_ft, weights=self.weights)
#             elif self.factorize_mode == "dep-sep":
#                 out_ft = self.compl_mul2d(x_ft, weights=None, depthwise_weights=self.depthwise_weights, pointwise_weights=self.pointwise_weights)
#             x = torch.fft.irfft2(out_ft, s=(input_dim1, input_dim2), norm="forward")
#             return x



# class pointwise_op_2D_HiLoc(nn.Module):
#     """
#     dim1 = Default output grid size along x (or 1st dimension)
#     dim2 = Default output grid size along y ( or 2nd dimension)
#     in_codim = Input co-domian dimension
#     out_codim = output co-domain dimension
#     """

#     def __init__(self, in_codim, out_codim, dim1, dim2, patch_based):
#         super(pointwise_op_2D, self).__init__()
#         self.conv = nn.Conv2d(int(in_codim), int(out_codim), 1)
#         self.dim1 = int(dim1)
#         self.dim2 = int(dim2)
#         self.patch_based = patch_based

#     def forward(self, x, dim1=None, dim2=None):
#         """
#         input shape = (batch, in_codim, input_dim1,input_dim2)
#         output shape = (batch, out_codim, dim1,dim2)
#         """
#         if dim1 is None:
#             dim1 = self.dim1
#             dim2 = self.dim2
#         x_out = self.conv(x)

#         # ft = torch.fft.rfft2(x_out)
#         # ft_u = torch.zeros_like(ft)
#         # ft_u[:dim1//2-1,:dim2//2-1] = ft[:dim1//2-1,:dim2//2-1]
#         # ft_u[-(dim1//2-1):,:dim2//2-1] = ft[-(dim1//2-1):,:dim2//2-1]
#         # x_out = torch.fft.irfft2(ft_u)
#         if not self.patch_based:
#             x_out = torch.nn.functional.interpolate(
#                 x_out, size=(dim1, dim2), mode="bicubic", align_corners=True, antialias=True
#             )
#         return x_out
    
class OperatorBlock_2D(nn.Module):
    """
    Normalize = if true performs InstanceNorm2d on the output.
    Non_Lin = if true, applies point wise nonlinearity.
    All other variables are consistent with the SpectralConv2d_Uno class.
    """

    def __init__(
        self,
        in_codim,
        out_codim,
        dim1,
        dim2,
        modes1,
        modes2,
        Normalize=False,
        Non_Lin=True,
        Apply_linear_transform=True
    ):
        super(OperatorBlock_2D, self).__init__()
        self.conv = SpectralConv2d_Uno(in_codim, out_codim, dim1, dim2, modes1, modes2)
        self.Apply_linear_transform = Apply_linear_transform
        if self.Apply_linear_transform:
            self.w = pointwise_op_2D(in_codim, out_codim, dim1, dim2)
        self.normalize = Normalize
        self.non_lin = Non_Lin
        if Normalize:
            self.normalize_layer = torch.nn.InstanceNorm2d(int(out_codim), affine=True)

    def forward(self, x, dim1=None, dim2=None):
        """
        input shape = (batch, in_codim, input_dim1,input_dim2)
        output shape = (batch, out_codim, dim1,dim2)
        """        
        x1_out = self.conv(x, dim1, dim2)
        if self.Apply_linear_transform:
            x2_out = self.w(x, dim1, dim2)
            x_out = x1_out + x2_out
        else:
            x_out = x1_out
        if self.normalize:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out

# class OperatorBlock_2D_HiLoc(nn.Module):
#     """
#     Normalize = if true performs InstanceNorm2d on the output.
#     Non_Lin = if true, applies point wise nonlinearity.
#     All other variables are consistent with the SpectralConv2d_Uno class.
#     """

#     def __init__(
#         self,
#         in_codim,
#         out_codim,
#         dim1,
#         dim2,
#         modes1,
#         modes2,
#         Normalize=True,
#         Non_Lin=True,
#         factorize_mode = False,
#         trunc_mode = None,
#         patch_based = False,
#         patch_size = 64,
#         use_attn = False,
#         num_heads = 8,
#         num_splits = 4
#     ):
#         super(OperatorBlock_2D_HiLoc, self).__init__()
#         self.out_channels = out_codim
#         self.in_channels = in_codim
#         self.patch_based = patch_based
#         self.patch_size = patch_size
#         self.trunc_mode = trunc_mode
#         self.use_attn = use_attn
#         self.num_heads = num_heads
#         self.normalize = Normalize
#         self.non_lin = Non_Lin
#         self.num_splits = num_splits
        
#         assert trunc_mode is not None or patch_based, "Either trunc_mode or patch_based must be enable"
        
#         # Global convolution
#         if trunc_mode is not None:
#             self.conv_trunc = SpectralConv2d_UnoHiLoc(
#                 in_codim, out_codim, dim1, dim2, modes1, modes2, trunc_mode=trunc_mode, patch_based=False, factorize_mode = factorize_mode
#             )
#         else:
#             self.conv_trunc = None
        
#         # Local convolution
#         if patch_based:
#             self.conv_patch = SpectralConv2d_UnoHiLoc(
#                 in_codim, out_codim, dim1, dim2, modes1, modes2, trunc_mode=None, patch_based=True, factorize_mode=factorize_mode
#             )
#         else:
#             self.conv_patch = None
        
#         # Linear transform
#         self.w = pointwise_op_2D(in_codim, out_codim, dim1, dim2, patch_based=None)
        
#         if use_attn:
#             self.attn = simple_attn(midc=out_codim, heads=num_heads)
            
#         if Normalize:
#             self.normalize_layer = torch.nn.InstanceNorm2d(int(out_codim), affine=True)

#     def forward(self, x, dim1=None, dim2=None):
#         """
#         input shape = (batch, in_codim, input_dim1,input_dim2)
#         output shape = (batch, out_codim, dim1,dim2)
#         """        
#         # initialize output tensor
#         B = x.shape[0]
#         x_out = torch.zeros_like(
#             B, self.out_channels, dim1, dim2, dtype=x.dtype, device=x.device
#         )
        
#         # Global convolution
#         if self.conv_trunc is not None:
#             x_global = self.conv_trunc(x, dim1, dim2)
#             x_out += x_global
        
#         # Local convolution
#         if self.conv_patch is not None:
#             x_patches = self._make_patches(x, num_split=self.num_splits)
#             x_local = self.conv_patch(x_patches, dim1, dim2)
            
#             x_local = self._reconstruct_from_patches(x_local, num_split=self.num_splits)
#             x_out += x_local
        
#         # linear transform
#         x_out += self.w(x, dim1, dim2)        
            
#         if self.normalize:
#             x_out = self.normalize_layer(x_out)
#         if self.non_lin:
#             x_out = F.gelu(x_out)
#         if self.use_attn:
#             residual = x_out
#             x_out = self.attn(x_out)
#             x_out += residual
            
#         return x_out
    
#     def _make_patches(self, x, num_split=4):
#         print("x shape: ", x.shape)
#         B, C, H, W = x.shape
#         x_c_patches = []
#         x_c_height = x.chunk(num_split, dim=2)
#         for row in x_c_height:
#             x_c_patches.extend(row.chunk(num_split, dim=3))
#             assert x_c_patches[-1].shape == (B, C, H // num_split, W // num_split), f"Expected shape {(B, C, H // num_split, W // num_split)}, got {x_c_patches[-1].shape}"
#         print("x_c_patches shape: ", len(x_c_patches), x_c_patches[0].shape)
#         return x_c_patches

#     def _reconstruct_from_patches(self, patches, num_split=4):
#         print(type(patches))
#         print(len(patches))
#         assert len(patches) == num_split * num_split, f"Expected {num_split * num_split} patches, got {len(patches)}"
        
#         B, C, H, W = patches[0].shape
#         rows = []
#         for i in range(num_split):
#             row = torch.cat(patches[i * num_split : (i + 1) * num_split], dim=3)  # concat in width
#             rows.append(row)
#         full = torch.cat(rows, dim=2)  # concat in height
#         assert full.shape == (B, C, H*num_split, W*num_split), f"Expected shape {(B, C, H*num_split, W*num_split)}, got {full.shape}"
#         return full 
        
    
class SpectralConv3d_Uno(nn.Module):
    def __init__(
        self,
        in_codim,
        out_codim,
        dim1,
        dim2,
        dim3,
        modes1=None,
        modes2=None,
        modes3=None,
    ):
        super(SpectralConv3d_Uno, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT. 
        dim1 = Default output grid size along x (or 1st dimension of output domain) 
        dim2 = Default output grid size along y ( or 2nd dimension of output domain)
        dim3 = Default output grid size along time t ( or 3rd dimension of output domain)
        Ratio of grid size of the input and output grid size (dim1,dim2,dim3) implecitely 
        set the expansion or contraction farctor along each dimension.
        modes1, modes2, modes3 = Number of fourier modes to consider for the ontegral operator
                                Number of modes must be compatibale with the input grid size 
                                and desired output grid size.
                                i.e., modes1 <= min( dim1/2, input_dim1/2).
                                      modes2 <= min( dim2/2, input_dim2/2)
                                Here input_dim1, input_dim2 are respectively the grid size along 
                                x axis and y axis (or first dimension and second dimension) of the input domain.
                                Other modes also have the same constrain.
        in_codim = Input co-domian dimension
        out_codim = output co-domain dimension   
        """
        in_codim = int(in_codim)
        out_codim = int(out_codim)
        self.in_channels = in_codim
        self.out_channels = out_codim
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        if modes1 is not None:
            self.modes1 = modes1
            self.modes2 = modes2
            self.modes3 = modes3
        else:
            self.modes1 = dim1
            self.modes2 = dim2
            self.modes3 = dim3 // 2 + 1

        self.scale = (1 / (2 * in_codim)) ** (1.0 / 2.0)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.randn(
                in_codim,
                out_codim,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.randn(
                in_codim,
                out_codim,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights3 = nn.Parameter(
            self.scale
            * torch.randn(
                in_codim,
                out_codim,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights4 = nn.Parameter(
            self.scale
            * torch.randn(
                in_codim,
                out_codim,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x, dim1=None, dim2=None, dim3=None):
        """
        dim1,dim2,dim3 are the output grid size along (x,y,t)
        input shape = (batch, in_codim, input_dim1, input_dim2, input_dim3)
        output shape = (batch, out_codim, dim1,dim2,dim3)
        """
        if dim1 is not None:
            self.dim1 = dim1
            self.dim2 = dim2
            self.dim3 = dim3

        batchsize = x.shape[0]

        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1], norm="forward")

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            self.dim1,
            self.dim2,
            self.dim3 // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4
        )

        # Return to physical space
        x = torch.fft.irfftn(
            out_ft, s=(self.dim1, self.dim2, self.dim3), norm="forward"
        )
        return x


class pointwise_op_3D(nn.Module):
    def __init__(self, in_codim, out_codim, dim1, dim2, dim3):
        super(pointwise_op_3D, self).__init__()
        self.conv = nn.Conv3d(int(in_codim), int(out_codim), 1)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)
        self.dim3 = int(dim3)

    def forward(self, x, dim1=None, dim2=None, dim3=None):
        """
        dim1,dim2,dim3 are the output dimensions (x,y,t)
        """
        if dim1 is None:
            dim1 = self.dim1
            dim2 = self.dim2
            dim3 = self.dim3
        x_out = self.conv(x)

        ft = torch.fft.rfftn(x_out, dim=[-3, -2, -1])
        ft_u = torch.zeros_like(ft)
        ft_u[:, :, : (dim1 // 2), : (dim2 // 2), : (dim3 // 2)] = ft[
            :, :, : (dim1 // 2), : (dim2 // 2), : (dim3 // 2)
        ]
        ft_u[:, :, -(dim1 // 2) :, : (dim2 // 2), : (dim3 // 2)] = ft[
            :, :, -(dim1 // 2) :, : (dim2 // 2), : (dim3 // 2)
        ]
        ft_u[:, :, : (dim1 // 2), -(dim2 // 2) :, : (dim3 // 2)] = ft[
            :, :, : (dim1 // 2), -(dim2 // 2) :, : (dim3 // 2)
        ]
        ft_u[:, :, -(dim1 // 2) :, -(dim2 // 2) :, : (dim3 // 2)] = ft[
            :, :, -(dim1 // 2) :, -(dim2 // 2) :, : (dim3 // 2)
        ]

        x_out = torch.fft.irfftn(ft_u, s=(dim1, dim2, dim3))

        x_out = torch.nn.functional.interpolate(
            x_out, size=(dim1, dim2, dim3), mode="trilinear", align_corners=True
        )
        return x_out


class OperatorBlock_3D(nn.Module):
    """
    Normalize = if true performs InstanceNorm3d on the output.
    Non_Lin = if true, applies point wise nonlinearity.
    All other variables are consistent with the SpectralConv3d_Uno class.
    """

    def __init__(
        self,
        in_codim,
        out_codim,
        dim1,
        dim2,
        dim3,
        modes1,
        modes2,
        modes3,
        Normalize=False,
        Non_Lin=True,
    ):
        super(OperatorBlock_3D, self).__init__()
        self.conv = SpectralConv3d_Uno(
            in_codim, out_codim, dim1, dim2, dim3, modes1, modes2, modes3
        )
        self.w = pointwise_op_3D(in_codim, out_codim, dim1, dim2, dim3)
        self.normalize = Normalize
        self.non_lin = Non_Lin
        if Normalize:
            self.normalize_layer = torch.nn.InstanceNorm3d(int(out_codim), affine=True)

    def forward(self, x, dim1=None, dim2=None, dim3=None):
        """
        input shape = (batch, in_codim, input_dim1, input_dim2, input_dim3)
        output shape = (batch, out_codim, dim1,dim2,dim3)
        """
        x1_out = self.conv(x, dim1, dim2, dim3)
        x2_out = self.w(x, dim1, dim2, dim3)
        x_out = x1_out + x2_out
        if self.normalize:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out
