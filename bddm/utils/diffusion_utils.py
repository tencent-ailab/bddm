#!/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
#  Diffusion Utils: Pre-compute Variables
#
#  Author: Max W. Y. Lam (maxwylam@tencent.com)
#  Copyright (c) 2021Tencent. All Rights Reserved
#
########################################################################


import torch


def compute_diffusion_params(beta):
    """
    Compute the diffusion parameters defined in BDDMs

    Parameters:
        beta (tensor):      the beta schedule
    Returns:
        diff_params (dict): a dictionary of diffusion hyperparameters including:
            T (int), beta/alpha/sigma (torch.tensor on cpu, shape=(T, ))
            These cpu tensors are changed to cuda tensors on each individual gpu
    """

    alpha = 1 - beta
    sigma = beta + 0
    for t in range(1, len(beta)):
        alpha[t] *= alpha[t-1]
        sigma[t] *= (1-alpha[t-1]) / (1-alpha[t])
    alpha = torch.sqrt(alpha)
    sigma = torch.sqrt(sigma)
    diff_params = {"T": len(beta), "beta": beta, "alpha": alpha, "sigma": sigma}
    return diff_params


def map_noise_scale_to_time_step(alpha_infer, alpha):
    """
    Map an alpha_infer to an approx. time step in the alpha tensor.
        (Modified from Algorithm 3 Fast Sampling in DiffWave)

    Parameters:
        alpha_infer (float): noise scale at time `n` for inference
        alpha (tensor):      noise scales used in training, shape=(T, )
    Returns:
        t (float):           approximated time step in alpha tensor
    """

    if alpha_infer < alpha[-1]:
        return len(alpha) - 1
    if alpha_infer > alpha[0]:
        return 0
    for t in range(len(alpha) - 1):
        if alpha[t+1] <= alpha_infer <= alpha[t]:
            step_diff = alpha[t] - alpha_infer
            step_diff /= alpha[t] - alpha[t+1]
            return t + step_diff.item()
    return -1
