#!/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
#  EMA Helper Class
#
#  Author: Max W. Y. Lam (maxwylam@tencent.com)
#  Copyright (c) 2021Tencent. All Rights Reserved
#
########################################################################


import torch.nn as nn


class EMAHelper(object):

    def __init__(self, mu=0.999):
        """
        Exponential Moving Average Training Helper Class

        Parameters:
            mu (float): decaying rate
        """
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        """
        Register module by copying all learnable parameters to self.shadow

        Parameters:
            module (nn.Module): model to be trained
        """
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        """
        Update self.shadow using the module parameters

        Parameters:
            module (nn.Module): model in training
        """
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        """
        Copy self.shadow to the module parameters

        Parameters:
            module (nn.Module): model in training
        """
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        """
        Initialize a new module using self.shadow as the parameters

        Parameters:
            module (nn.Module): model in training
        """
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(
                inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        """
        Get self.shadow as the state dict

        Returns:
            shadow (dict): state dict
        """
        return self.shadow

    def load_state_dict(self, state_dict):
        """
        Load a state dict to self.shadow

        Parameters:
            state dict (dict): state dict to be copied to self.shadow
        """
        self.shadow = state_dict
