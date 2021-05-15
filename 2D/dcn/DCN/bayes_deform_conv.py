#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import math
from torch import nn
from torch.nn import init, Parameter
from torch.nn.modules.utils import _pair

from misc import ModuleWrapper
from metrics import calculate_kl as KL_DIV
from dcn.DCN.dcn_v2 import DCN, DCNv2, _DCNv2


class BayesDeformConv2D(ModuleWrapper):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, deformable_groups=1,
                 bias=True, priors=None):
        super(BayesDeformConv2D, self).__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups
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

    def forward(self, input, offset, mask, sample=True):
        assert 2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == offset.shape[1]
        
        self.W_sigma = torch.log1p(torch.exp(self.W_rho))
        if self.use_bias:
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_var = self.bias_sigma ** 2
        else:
            self.bias_sigma = bias_var = None

        act_mu = _DCNv2.apply(
            input, offset, mask, self.W_mu, self.bias_mu, self.stride, self.padding, self.dilation, self.deformable_groups)
            
        act_var = 1e-16 + _DCNv2.apply(
            input ** 2, offset, mask, self.W_sigma ** 2, bias_var, self.stride, self.padding, self.dilation, self.deformable_groups)
            
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
            
        

dcn_v2_conv = _DCNv2.apply

class BayesDeformConvPack2D(BayesDeformConv2D):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1,
                 dilation=1, deformable_groups=1, bias=True, 
                 priors=None, lr_mult=0.1):
        super(BayesDeformConvPack2D, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, deformable_groups, bias, priors)


        offset_channels = self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1]
        mask_channels = self.deformable_groups * 1 * self.kernel_size[0] * self.kernel_size[1]
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.conv_offset = nn.Conv2d(self.in_channels,
                                          offset_channels,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
                                          
        self.conv_mask   = nn.Conv2d(self.in_channels,
                                          mask_channels,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        self.conv_offset.lr_mult = lr_mult
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()
        self.conv_mask.weight.data.zero_()
        self.conv_mask.bias.data.zero_()

    def forward(self, input, sample=True):
        offset = self.conv_offset(input)
        mask = torch.sigmoid(self.conv_mask(input))
        
        self.W_sigma = torch.log1p(torch.exp(self.W_rho))
        if self.use_bias:
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_var = self.bias_sigma ** 2
        else:
            self.bias_sigma = bias_var = None

        act_mu = _DCNv2.apply(
            input, offset, mask, self.W_mu, self.bias_mu, self.stride, self.padding, self.dilation,
            self.deformable_groups)
            
        act_var = 1e-16 + _DCNv2.apply(
            input ** 2, offset, mask, self.W_sigma ** 2, bias_var, self.stride, self.padding,
            self.dilation, self.deformable_groups)
            
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
        







