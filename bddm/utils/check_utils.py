#!/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
#  Check Utils: Find Checkpoints
#
#  Author: Max W. Y. Lam (maxwylam@tencent.com)
#  Copyright (c) 2021Tencent. All Rights Reserved
#
########################################################################


import os
import glob
import torch


def check_score_network(config):
    """
    Check if the score network is trained by searching the ${exp_dir}/score_nets

    Parameters:
        config (namespace): the configuration given by the user
    Returns:
        result (bool):      a boolean to determine if the score network is trained
        path (str):         the path to the score network checkpoint if trained
    """
    if config.load != '':
        ckpt = torch.load(config.load)
        if 'score_net_training_step' in ckpt.keys():
            if ckpt['score_net_training_step'] >= config.score_net_training_steps:
                return True, config.load
        else:
            # We suppose that an external checkpoint is already well-trained
            return True, config.load
    max_training_steps = 0
    for path in glob.glob(os.path.join(config.exp_dir, 'score_nets', '[0-9]*.pkl')):
        n_training_steps = int(path.split('/')[-1].split('.')[0])
        max_training_steps = max(max_training_steps, n_training_steps)
    if max_training_steps == 0:
        return False, None
    load_path = os.path.join(config.exp_dir, 'score_nets', '%d.pkl'%(max_training_steps))
    return True, load_path


def check_schedule_network(config):
    """
    Check if the schedule network is trained by searching the ${exp_dir}/schedule_nets

    Parameters:
        config (namespace): the configuration given by the user
    Returns:
        result (bool):      a boolean to determine if the schedule network is trained
        path (str):         the path to the BDDM checkpoint if trained
    """
    if config.load != '':
        ckpt = torch.load(config.load)
        if 'schedule_net_training_step' in ckpt.keys():
            return True, config.load
        else:
            return False, config.load
    max_training_steps = 0
    for path in glob.glob(os.path.join(config.exp_dir, 'schedule_nets', '[0-9]*.pkl')):
        n_training_steps = int(path.split('/')[-1].split('.')[0])
        max_training_steps = max(max_training_steps, n_training_steps)
    if max_training_steps == 0:
        # We suppose that an external checkpoint only contains a well-trained score network
        return False, config.load
    schedule_net_load_path = os.path.join(
        config.exp_dir, 'schedule_nets', '%d.pkl'%(max_training_steps))
    return True, schedule_net_load_path
