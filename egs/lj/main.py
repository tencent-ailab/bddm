import os
import sys
sys.path.append('../../')
import json
import shutil
import hashlib
import argparse
import numpy as np
import yaml

import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from bddm import trainer, sampler
from bddm.utils.log_utils import log


def dict_hash_5char(dictionary):
    ''' Map a unique dictionary into a 5-character string '''
    dhash = hashlib.md5()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()[:5]


def start_exp(config, config_hash):
    ''' Create experiment directory or set it to an existing directory '''
    if config.load != '' and '_nets' in config.load:
        config.exp_dir = '/'.join(config.load.split('/')[:-2])
    else:
        config.exp_dir += '/%s-%s_conf-hash-%s' % (
            config.score_net, config.schedule_net, config_hash)
    if config.local_rank != 0:
        return
    log('Experiment directory: %s' % (config.exp_dir), config)
    # Backup the config file
    shutil.copyfile(config.config, os.path.join(config.exp_dir, 'conf.yml'))
    # Create a backup scripts sub-folder
    os.makedirs(os.path.join(config.exp_dir, 'backup_scripts'), exist_ok=True)
    # Backup all .py files under bddm/
    backup_files = []
    for root, _, files in os.walk("../../"):
        if 'egs' in root:
            continue
        for f in files:
            if f.endswith(".py"):
                backup_files.append(os.path.join(root, f))
    for src_file in backup_files:
        basename = src_file
        while '../' in basename:
            basename = basename.replace('../', '')
        basename = basename.replace('./', '')
        dst_file = os.path.join(config.exp_dir, 'backup_scripts', basename)
        dst_dir = '/'.join(dst_file.split('/')[:-1])
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        shutil.copyfile(src_file, dst_file)
    # Prepare sub-folders for saving model checkpoints
    os.makedirs(os.path.join(config.exp_dir, 'score_nets'), exist_ok=True)
    os.makedirs(os.path.join(config.exp_dir, 'schedule_nets'), exist_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bilateral Denoising Diffusion Models')
    parser.add_argument('--command',
                        type=str,
                        default='train',
                        help='available commands: train | search | generate')
    parser.add_argument('--config',
                        '-c',
                        type=str,
                        default='conf.yml',
                        help='config .yml path')
    parser.add_argument('--local_rank',
                        type=int,
                        default=0,
                        help='process device ID for multi-GPU training')

    arg_config = parser.parse_args()

    # Parse yaml and define configurations
    config = arg_config.__dict__
    with open(arg_config.config) as f:
        yaml_config = yaml.safe_load(f)
    HASH = dict_hash_5char(yaml_config)
    for key in yaml_config:
        config[key] = yaml_config[key]
    config = argparse.Namespace(**config)

    # Set random seed for reproducible results
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.cuda.set_device(config.local_rank)

    # Check if the command is valid or not
    commands = ['train', 'schedule', 'generate']
    assert config.command in commands, 'Error: %s command not found.'%(config.command)

    # Create/retrieve exp dir
    start_exp(config, HASH)
    log('Argv: %s' % (' '.join(sys.argv)), config)

    try:
        if config.command == 'train':
            # Create Trainer for training
            trainer = trainer.Trainer(config)
            trainer.train()
        elif config.command == 'schedule':
            # Create Sampler for noise scheduling
            sampler = sampler.Sampler(config)
            sampler.noise_scheduling_without_params()
        elif config.command == 'generate':
            # Create Sampler for generation
            # NOTE: Remember to define "gen_data_dir" in conf.yml before data generation
            sampler = sampler.Sampler(config)
            sampler.generate()
        log('-' * 80, config)

    except KeyboardInterrupt:
        log('-' * 80, config)
        log('Exiting early', config)
