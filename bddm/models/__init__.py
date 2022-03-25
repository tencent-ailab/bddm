#!/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
#  Globally Attentive Locally Recurrent (GALR) Networks
#  (https://arxiv.org/abs/2101.05014)
#
#  Author: Max W. Y. Lam (maxwylam@tencent.com)
#  Copyright (c) 2021Tencent. All Rights Reserved
#
########################################################################


from .diffwave import DiffWave
from .galr import GALR


def get_score_network(config):
    if config.score_net == 'DiffWave':
        conf_keys = DiffWave.__init__.__code__.co_varnames
        model_config = {k: v for k, v in vars(config).items() if k in conf_keys}
        return DiffWave(**model_config)


def get_schedule_network(config):
    if config.schedule_net == 'GALR':
        conf_keys = GALR.__init__.__code__.co_varnames
        model_config = {k: v for k, v in vars(config).items() if k in conf_keys}
        return GALR(**model_config)
