#!/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
#  BMUF Multi-GPU Training Method
#
#  Author: Max W. Y. Lam (maxwylam@tencent.com)
#  Copyright (c) 2021Tencent. All Rights Reserved
#
########################################################################


import os
import sys
import psutil
import torch
import torch.nn as nn
import torch.distributed as dist


class BmufTrainer(object):

    def __init__(self, master_node, rank, world_size, model, block_momentum, block_lr):
        """
        Basic BMUF Trainer Class, implements Nesterov Block Momentum

        Parameters:
            master_node (int):      master node index, zero in most cases
            rank (int):             local rank, eg, 0-7 if 8GPUs are used
            world_size (int):       total number of workers
            model (nn.Module):      PyTorch model
            block_momentum (float): block momentum value
            block_lr (float):       block learning rate
        """
        assert torch.cuda.is_available(), "Distributed mode requires CUDA."
        self.master_node = 0  # By default, use device 0 as master node
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.block_momentum = block_momentum
        self.block_lr = block_lr
        dist.init_process_group(backend="nccl", init_method="env://")
        param_vec = nn.utils.parameters_to_vector(model.parameters())
        self.param = param_vec.data.clone()
        dist.broadcast(tensor=self.param, src=self.master_node, async_op=False)
        num_param = self.param.numel()
        self.delta_prev = torch.FloatTensor([0]*num_param).to(self.param.device)
        if self.rank == self.master_node:
            self.delta_prev = torch.FloatTensor([0]*num_param).cuda(self.master_node)
        else:
            self.delta_prev = None
            self._copy_vec_to_param(self.param)
        # for fining the best model among different ranks
        self.min_val_loss = torch.FloatTensor([float('inf')]).to(self.param.device)

    def check_all_processes_running(self):
        """
        Check whether all processes are healthy
        """
        pid = os.getpid()
        for p in range(pid-self.rank, pid-self.rank+self.world_size):
            if not psutil.pid_exists(p):
                sys.exit(-1)

    def get_average_valid_loss(self, val_loss):
        """
        Get average validatin loss  through all reduce operations

        Parameters:
            val_loss (Tensor): calculated validation loss in each process
        Returns:
            save_or_not (bool): whether save the current validated model or not
        """
        save_or_not = False
        dist.all_reduce(tensor=val_loss)
        val_loss = val_loss / float(self.world_size)
        # save the min valid loss among all gpus
        if 'min_val_loss' not in self.__dict__ or val_loss < self.min_val_loss:
            self.min_val_loss = val_loss.clone()
            save_or_not = True
        return save_or_not

    def update_and_sync(self, val_loss=None):
        """
        Update and synchronize block gradients

        Parameters:
            val_loss (tensor): calculated validation loss in each process
        Returns:
            save_or_not (bool): whether save the current validated model or not
        """
        self.check_all_processes_running()
        save_or_not = False
        if val_loss is not None:
            save_or_not = self.get_average_valid_loss(val_loss)

        cur_param_vec = nn.utils.parameters_to_vector(self.model.parameters()).data
        delta = self.param - cur_param_vec
        # Gather block gradients into delta
        dist.reduce(tensor=delta, dst=self.master_node)

        # Check if model params are still healthy
        if torch.isnan(delta).sum().item():
            print('Found nan, exit!', flush=True)
            sys.exit(-1)
        if self.rank == self.master_node:
            # Local rank is master node
            delta = delta / float(self.world_size)
            self.delta_prev = self.block_momentum * self.delta_prev + \
                              (self.block_lr * (1 - self.block_momentum) * delta)
            self.param -= (1+self.block_momentum) * self.delta_prev
        dist.broadcast(tensor=self.param, src=self.master_node, async_op=False)
        self._copy_vec_to_param(self.param)
        return save_or_not

    def _copy_vec_to_param(self, vec):
        """
        Copy a vectorized array to the model parameters

        Parameters:
            vec (tensor): a single vector represents the parameters of a model.
        """
        # Ensure vec of type Tensor
        if not isinstance(vec, torch.Tensor):
            raise TypeError('expected torch.Tensor, but got: {}'
                            .format(torch.typename(vec)))
        # Pointer for slicing the vector for each parameter
        pointer = 0
        for param in self.model.parameters():
            # The length of the parameter
            num_param = param.numel()
            # Slice the vector, reshape it, and replace the old data of the parameter
            param.data = param.data.copy_(vec[pointer:pointer + num_param]
                                          .view_as(param).data)
            # Increment the pointer
            pointer += num_param
