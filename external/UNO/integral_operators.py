import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
import copy
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
        self.weights_high = nn.Parameter(
            self.scale
            * (
                torch.randn(
                    in_codim, out_codim, self.modes1, self.modes2, dtype=torch.cfloat
                )
            )
        )
        
        self.weights3 = nn.Parameter(
            self.scale
            * (
                torch.randn(
                    in_codim, out_codim, self.modes1, self.modes2, dtype=torch.cfloat
                )
            )
        )
        self.weights4 = nn.Parameter(
            self.scale
            * (
                torch.randn(
                    in_codim, out_codim, self.modes1, self.modes2, dtype=torch.cfloat
                )
            )
        )
        # print("init_weights1.shape", self.weights1.shape)
        # print("init_weights2.shape", self.weights2.shape)

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, dim1=None, dim2=None, region=None):
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

        if region == "low": #low-low and high-low
            out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
                x_ft[:, :, : self.modes1, : self.modes2], self.weights1
            )
            out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
                x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
            )
        elif region == "high": #high-high
            out_ft[:, :, -self.modes1 :, -self.modes2 :] = self.compl_mul2d(
                x_ft[:, :, -self.modes1 :, -self.modes2 :], self.weights_high
            )
        elif region == "all": # high-low, low-low, high-high, and low-high
            out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
                x_ft[:, :, : self.modes1, : self.modes2], self.weights1
            )
            out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
                x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
            )
            out_ft[:, :, -self.modes1 :, -self.modes2 :] = self.compl_mul2d(
                x_ft[:, :, -self.modes1 :, -self.modes2 :], self.weights3
            )
            out_ft[:, :, : self.modes1, -self.modes2 :] = self.compl_mul2d(
                x_ft[:, :, : self.modes1, -self.modes2 :], self.weights4
            )

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

#############################
# class SimpleAttention(nn.Module):
    # '''
    # Simple Galerkin Attention module.
    # Uses (Q (K^T V)) without softmax for integral operator approximation.

    # In this implementation, output is (N, L, E).
    # batch_first will be added in the next version of PyTorch: https://github.com/pytorch/pytorch/pull/55285

    # Reference: code base modified from
    # https://nlp.seas.harvard.edu/2018/04/03/attention.html
    # - added xavier init gain
    # - added layer norm <-> attn norm switch
    # - added diagonal init

    # In https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py
    # the linear attention in each head is implemented as an Einstein sum
    # attn_matrix = torch.einsum('bhnd,bhne->bhde', k, v)
    # attn = torch.einsum('bhnd,bhde->bhne', q, attn_matrix)
    # return attn.reshape(*q.shape)
    # here in our implementation this is achieved by a slower transpose+matmul
    # but can conform with the template Harvard NLP gave
    # '''

    # def __init__(self, n_head, d_model,
    #              pos_dim: int = 1,
    #              dropout=0.1,
    #              xavier_init=1e-4,
    #              diagonal_weight=1e-2,
    #              symmetric_init=False,
    #              norm=False,
    #              norm_type='layer',
    #              eps=1e-5,
    #              debug=False):
    #     super(SimpleAttention, self).__init__()
    #     assert d_model % n_head == 0
    #     self.d_k = d_model // n_head
    #     self.n_head = n_head
    #     self.pos_dim = pos_dim
    #     self.linears = nn.ModuleList(
    #         [copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(3)])
    #     self.xavier_init = xavier_init
    #     self.diagonal_weight = diagonal_weight
    #     self.symmetric_init = symmetric_init
    #     if self.xavier_init > 0:
    #         self._reset_parameters()
    #     self.add_norm = norm
    #     self.norm_type = norm_type
    #     if norm:
    #         self._get_norm(eps=eps)

    #     if pos_dim > 0:
    #         self.fc = nn.Linear(d_model + n_head*pos_dim, d_model)

    #     self.attn_weight = None
    #     self.dropout = nn.Dropout(dropout)
    #     self.debug = debug

    # def forward(self, query, key, value, pos=None, weight=None):
    #     bsz = query.size(0)
    #     if weight is not None:
    #         query, key = weight*query, weight*key

    #     query, key, value = \
    #         [layer(x).view(bsz, -1, self.n_head, self.d_k).transpose(1, 2)
    #          for layer, x in zip(self.linears, (query, key, value))]

    #     if self.add_norm:
    #         if self.norm_type == 'instance':
    #             key, value = key.transpose(-2, -1), value.transpose(-2, -1)

    #         key = torch.stack(
    #             [norm(x) for norm, x in
    #                 zip(self.norm_K, (key[:, i, ...] for i in range(self.n_head)))], dim=1)
    #         value = torch.stack(
    #             [norm(x) for norm, x in
    #                 zip(self.norm_V, (value[:, i, ...] for i in range(self.n_head)))], dim=1)

    #         if self.norm_type == 'instance':
    #             key, value = key.transpose(-2, -1), value.transpose(-2, -1)

    #     if pos is not None and self.pos_dim > 0:
    #         assert pos.size(-1) == self.pos_dim
    #         pos = pos.unsqueeze(1)
    #         pos = pos.repeat([1, self.n_head, 1, 1])
    #         query, key, value = [torch.cat([pos, x], dim=-1)
    #                              for x in (query, key, value)]

    #     x, self.attn_weight = linear_attention(query, key, value, dropout=self.dropout)

    #     out_dim = self.n_head * self.d_k if pos is None else self.n_head * \
    #         (self.d_k + self.pos_dim)
    #     att_output = x.transpose(1, 2).contiguous().view(bsz, -1, out_dim)

    #     if pos is not None and self.pos_dim > 0:
    #         att_output = self.fc(att_output)

    #     return att_output, self.attn_weight

    # def _reset_parameters(self):
    #     for param in self.linears.parameters():
    #         if param.ndim > 1:
    #             xavier_uniform_(param, gain=self.xavier_init)
    #             if self.diagonal_weight > 0.0:
    #                 param.data += self.diagonal_weight * \
    #                     torch.diag(torch.ones(
    #                         param.size(-1), dtype=torch.float))
    #             if self.symmetric_init:
    #                 param.data += param.data.T
    #                 # param.data /= 2.0
    #         else:
    #             constant_(param, 0)

    # def _get_norm(self, eps):
    #     if self.norm_type == 'instance':
    #         self.norm_K = self._get_instancenorm(self.d_k, self.n_head,
    #                                                 eps=eps,
    #                                                 affine=True)
    #         self.norm_V = self._get_instancenorm(self.d_k, self.n_head,
    #                                                 eps=eps,
    #                                                 affine=True)
    #     elif self.norm_type == 'layer':
    #         self.norm_K = self._get_layernorm(self.d_k, self.n_head,
    #                                             eps=eps)
    #         self.norm_V = self._get_layernorm(self.d_k, self.n_head,
    #                                             eps=eps)

    # @staticmethod
    # def _get_layernorm(normalized_dim, n_head, **kwargs):
    #     return nn.ModuleList(
    #         [copy.deepcopy(nn.LayerNorm(normalized_dim, **kwargs)) for _ in range(n_head)])

    # @staticmethod
    # def _get_instancenorm(normalized_dim, n_head, **kwargs):
    #     return nn.ModuleList(
    #         [copy.deepcopy(nn.InstanceNorm1d(normalized_dim, **kwargs)) for _ in range(n_head)])

# def linear_attention(query, key, value, dropout=None):
    # '''
    # Adapted from lucidrains' implementaion
    # https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py
    # to https://nlp.seas.harvard.edu/2018/04/03/attention.html template
    # linear_attn function
    # Compute the Scaled Dot Product Attention globally
    # '''

    # seq_len = query.size(-2)
    # scores = torch.matmul(key.transpose(-2, -1), value)
    # p_attn = scores / seq_len

    # if dropout is not None:
    #     p_attn = F.dropout(p_attn)

    # out = torch.matmul(query, p_attn)
    # return out, p_attn
#########################

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
    ):
        super(OperatorBlock_2D, self).__init__()
        self.conv = SpectralConv2d_Uno(in_codim, out_codim, dim1, dim2, modes1, modes2)
        self.w = pointwise_op_2D(in_codim, out_codim, dim1, dim2)
        self.normalize = Normalize
        self.non_lin = Non_Lin
        if Normalize:
            self.normalize_layer = torch.nn.InstanceNorm2d(int(out_codim), affine=True)

    def forward(self, x, dim1=None, dim2=None, region=None):
        """
        input shape = (batch, in_codim, input_dim1,input_dim2)
        output shape = (batch, out_codim, dim1,dim2)
        """        
        x1_out = self.conv(x, dim1, dim2, region)
        x2_out = self.w(x, dim1, dim2)
        x_out = x1_out + x2_out
        if self.normalize:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out
        
    
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
