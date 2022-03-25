#!/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
#  Log Utils: Print Log with Time/PID
#
#  Author: Max W. Y. Lam (maxwylam@tencent.com)
#  Copyright (c) 2021Tencent. All Rights Reserved
#
########################################################################

import os
import sys
import time


def ctime():
    """
    Get time now

    Returns:
        time_string (str): current time in string
    """
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))


def head():
    """
    Get header for logging: time and PID of the current process

    Returns:
        header_string (str): the header string for logging
    """
    return "%s %d" % (ctime(), os.getpid())


def log(msg, config):
    """
    Save log to the device[id].log & print device0.log to STDOUT

    Parameters:
        msg (sting):        message to be logged
        config (namespace): BDDM Configuration
    """
    if config.local_rank == 0:
        sys.stdout.write("[%s] %s\n" % (head(), msg))
        sys.stdout.flush()
    os.makedirs(config.exp_dir, exist_ok=True)
    with open(os.path.join(config.exp_dir, 'device%d.log'%(config.local_rank)), 'a') as f:
        f.write("[%s] %s\n" % (head(), msg))
        f.flush()
