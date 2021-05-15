#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import math
from torch import nn
from torch.nn import init, Parameter
from torch.nn.modules.utils import _triple

from misc import ModuleWrapper
from metrics import calculate_kl as KL_DIV
from dcn.functions.deform_conv_func import DeformConvFunction


class BayesDeformConv(ModuleWrapper):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, groups=1, deformable_groups=1,
                 im2col_step=64, bias=True, priors=None):
        super(BayesDeformConv, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels {} must be divisible by groups {}'.format(in_channels, groups))
        if out_channels % groups != 0:
            raise ValueError('out_channels {} must be divisible by groups {}'.format(out_channels, groups))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        
        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.W_rho = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        if self.use_bias:
            self.bias_mu = Parameter(torch.Tensor(out_channels))
            self.bias_rho = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, input, offset,sample=True):
        assert 3 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] == offset.shape[1]
        
        self.W_sigma = torch.log1p(torch.exp(self.W_rho))
        if self.use_bias:
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_var = self.bias_sigma ** 2
        else:
            self.bias_sigma = bias_var = None

        act_mu = DeformConvFunction.apply(
            input, offset, self.W_mu, self.bias_mu, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups, self.im2col_step)
            
        act_var = 1e-16 + DeformConvFunction.apply(
            input ** 2, offset, self.W_sigma ** 2, bias_var, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups, self.im2col_step)
            
        act_std = torch.sqrt(act_var)

        if self.training or sample:
            eps = torch.empty(act_mu.size()).normal_(0, 1).to(self.device)
            return act_mu + act_std * eps
        else:
            return act_mu
    
    def kl_loss(self):
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl
            
        

_DeformConv = DeformConvFunction.apply

class BayesDeformConvPack(BayesDeformConv):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True, 
                 priors=None, lr_mult=0.1):
        super(BayesDeformConvPack, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, groups, deformable_groups, im2col_step, bias, priors)


        out_channels = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.conv_offset = nn.Conv3d(self.in_channels,
                                          out_channels,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        self.conv_offset.lr_mult = lr_mult
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, input,sample=True):
        offset = self.conv_offset(input)
        
        self.W_sigma = torch.log1p(torch.exp(self.W_rho))
        if self.use_bias:
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_var = self.bias_sigma ** 2
        else:
            self.bias_sigma = bias_var = None

        act_mu = DeformConvFunction.apply(
            input, offset, self.W_mu, self.bias_mu, self.stride, self.padding, self.dilation,
            self.groups, self.deformable_groups, self.im2col_step)
            
        act_var = 1e-16 + DeformConvFunction.apply(
            input ** 2, offset, self.W_sigma ** 2, bias_var, self.stride, self.padding,
            self.dilation, self.groups, self.deformable_groups, self.im2col_step)
            
        act_std = torch.sqrt(act_var)

        if self.training or sample:
            eps = torch.empty(act_mu.size()).normal_(0, 1).to(self.device)
            return act_mu + act_std * eps
        else:
            return act_mu
        
        
    def kl_loss(self):
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl
        


































class DeformConv_d(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dimension='THW', dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True):
        super(DeformConv_d, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels {} must be divisible by groups {}'.format(in_channels, groups))
        if out_channels % groups != 0:
            raise ValueError('out_channels {} must be divisible by groups {}'.format(out_channels, groups))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.dimension = dimension
        self.length = len(dimension)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step
        self.use_bias = bias

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        if not self.use_bias:
            self.bias.requires_grad = False

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, temp):
        dimension_T = 'T' in self.dimension
        dimension_H = 'H' in self.dimension
        dimension_W = 'W' in self.dimension
        b, c, t, h, w = temp.shape
        if self.length == 2:
            temp1 = temp.clone()[:, 0:81 - c, :, :, :]
            offset = torch.cat((temp.clone(), temp1), dim=1)
            if dimension_T == False:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = 0  # T
                    offset[:, i * 3 + 1, :, :, :] = temp[:, i * 2, :, :, :]
                    offset[:, i * 3 + 2, :, :, :] = temp[:, i * 2 + 1, :, :, :]
            if dimension_H == False:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = temp[:, i * 2, :, :, :]
                    offset[:, i * 3 + 1, :, :, :] = 0  # H
                    offset[:, i * 3 + 2, :, :, :] = temp[:, i * 2 + 1, :, :, :]
            if dimension_W == False:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = temp[:, i * 2, :, :, :]
                    offset[:, i * 3 + 1, :, :, :] = temp[:, i * 2 + 1, :, :, :]
                    offset[:, i * 3 + 2, :, :, :] = 0  # W

        if self.length == 1:
            temp1 = temp.clone()
            offset = torch.cat((temp.clone(), temp1, temp1), dim=1)
            if dimension_T == True:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = temp[:, i, :, :, :]  # T
                    offset[:, i * 3 + 1, :, :, :] = 0
                    offset[:, i * 3 + 2, :, :, :] = 0
            if dimension_H == True:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = 0
                    offset[:, i * 3 + 1, :, :, :] = temp[:, i, :, :, :]  # H
                    offset[:, i * 3 + 2, :, :, :] = 0
            if dimension_W == True:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = 0
                    offset[:, i * 3 + 1, :, :, :] = 0
                    offset[:, i * 3 + 2, :, :, :] = temp[:, i, :, :, :]  # W

        return DeformConvFunction.apply(input, offset,
                                        self.weight,
                                        self.bias,
                                        self.stride,
                                        self.padding,
                                        self.dilation,
                                        self.groups,
                                        self.deformable_groups,
                                        self.im2col_step)


_DeformConv = DeformConvFunction.apply


class DeformConvPack_d(DeformConv_d):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dimension='THW',
                 dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True, lr_mult=0.1):
        super(DeformConvPack_d, self).__init__(in_channels, out_channels,
                                             kernel_size, stride, padding, dimension, dilation, groups, deformable_groups,
                                             im2col_step, bias)
        self.dimension = dimension
        self.length = len(dimension)
        out_channels = self.deformable_groups * self.length * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        self.conv_offset = nn.Conv3d(self.in_channels,
                                     out_channels,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding,
                                     bias=True)
        self.conv_offset.lr_mult = lr_mult
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, input):
        temp = self.conv_offset(input)
        dimension_T = 'T' in self.dimension
        dimension_H = 'H' in self.dimension
        dimension_W = 'W' in self.dimension
        b, c, t, h, w = temp.shape
        if self.length == 2:
            temp1 = temp.clone()[:, 0:81 - c, :, :, :]
            offset = torch.cat((temp.clone(), temp1), dim=1)
            if dimension_T == False:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = 0  # T
                    offset[:, i * 3 + 1, :, :, :] = temp[:, i * 2, :, :, :]
                    offset[:, i * 3 + 2, :, :, :] = temp[:, i * 2 + 1, :, :, :]
            if dimension_H == False:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = temp[:, i * 2, :, :, :]
                    offset[:, i * 3 + 1, :, :, :] = 0  # H
                    offset[:, i * 3 + 2, :, :, :] = temp[:, i * 2 + 1, :, :, :]
            if dimension_W == False:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = temp[:, i * 2, :, :, :]
                    offset[:, i * 3 + 1, :, :, :] = temp[:, i * 2 + 1, :, :, :]
                    offset[:, i * 3 + 2, :, :, :] = 0  # W

        if self.length == 1:
            temp1 = temp.clone()
            offset = torch.cat((temp.clone(), temp1, temp1), dim=1)
            if dimension_T == True:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = temp[:, i, :, :, :]  # T
                    offset[:, i * 3 + 1, :, :, :] = 0
                    offset[:, i * 3 + 2, :, :, :] = 0
            if dimension_H == True:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = 0
                    offset[:, i * 3 + 1, :, :, :] = temp[:, i, :, :, :]  # H
                    offset[:, i * 3 + 2, :, :, :] = 0
            if dimension_W == True:
                for i in range(
                        self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]):
                    offset[:, i * 3, :, :, :] = 0
                    offset[:, i * 3 + 1, :, :, :] = 0
                    offset[:, i * 3 + 2, :, :, :] = temp[:, i, :, :, :]  # W

        return DeformConvFunction.apply(input, offset,
                                        self.weight,
                                        self.bias,
                                        self.stride,
                                        self.padding,
                                        self.dilation,
                                        self.groups,
                                        self.deformable_groups,
                                        self.im2col_step)
