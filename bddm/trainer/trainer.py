#!/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
#  BDDM Trainer (Support Multi-GPU Training using BMUF method)
#
#  Author: Max W. Y. Lam (maxwylam@tencent.com)
#  Copyright (c) 2021Tencent. All Rights Reserved
#
########################################################################


from __future__ import absolute_import

import os
import time
import copy
import torch

from bddm.sampler import Sampler
from bddm.trainer.ema import EMAHelper
from bddm.trainer.bmuf import BmufTrainer
from bddm.trainer.loss import ScoreLoss, StepLoss
from bddm.utils.log_utils import log
from bddm.utils.check_utils import check_score_network
from bddm.utils.diffusion_utils import compute_diffusion_params
from bddm.models import get_score_network, get_schedule_network
from bddm.loader.dataset import create_train_and_valid_dataloader


class Trainer(object):

    def __init__(self, config):
        """
        Trainer Class, implements a general multi-GPU training framework in PyTorch

        Parameters:
            config (namespace): BDDM Configuration
        """
        self.config = config
        self.exp_dir = config.exp_dir
        self.clip = config.grad_clip
        self.load = config.load
        self.model = get_score_network(config).cuda()
        # Define training target
        if self.config.resume_training and self.load != '':
            self.training_target = 'score_nets'
        else:
            score_net_trained, score_net_path = check_score_network(config)
            self.training_target = 'schedule_nets' if score_net_trained else 'score_nets'
        torch.autograd.set_detect_anomaly(True)
        # Initialize diffusion parameters using a pre-specified linear schedule
        noise_schedule = torch.linspace(config.beta_0, config.beta_T, config.T).cuda()
        self.diff_params = compute_diffusion_params(noise_schedule)
        if self.training_target == 'schedule_nets':
            if self.load == '':
                self.load = score_net_path
            self.diff_params["tau"] = config.tau
            for p in self.model.parameters():
                p.requires_grad = False
            # Define the schedule net as a sub-module of the score net for convenience
            self.model.schedule_net = get_schedule_network(config).cuda()
            self.loss_func = StepLoss(config, self.diff_params)
            self.n_training_steps = config.schedule_net_training_steps
            # In practice using batch size = 1 would lead to much lower step loss
            config.batch_size = 1
            model_to_train = self.model.schedule_net
        else:
            self.loss_func = ScoreLoss(config, self.diff_params)
            self.n_training_steps = config.score_net_training_steps
            model_to_train = self.model
        # Define optimizer
        self.optimizer = torch.optim.AdamW(model_to_train.parameters(),
            lr=config.lr, weight_decay=config.weight_decay, amsgrad=True)
        # Define EMA training helper
        self.ema_helper = EMAHelper(mu=config.ema_rate)
        self.ema_helper.register(model_to_train)
        # Initialize BMUF trainer
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.bmuf_trainer = BmufTrainer(0, config.local_rank, self.world_size,
            model_to_train, config.bmuf_block_momentum, config.bmuf_block_lr)
        self.sync_period = config.bmuf_sync_period
        self.device = torch.device("cuda:{}".format(config.local_rank))
        self.local_rank = config.local_rank
        # Get data loaders
        self.tr_loader, self.vl_loader = create_train_and_valid_dataloader(config)
        if self.training_target == 'score_nets':
            # Define a Sampler for quality validation (should be added after BMUF)
            self.valid_sampler = Sampler(config)
            self.valid_sampler.model = get_score_network(config).cuda()
        self.reset()

    def reset(self):
        """
        Reset training environment
        """
        self.tr_loss, self.vl_loss = [], []
        self.training_step = 0
        if self.load != '':
            package = torch.load(self.load, map_location=lambda storage, loc: storage.cuda())
            init_state_dict = self.model.state_dict()
            mismatch_params = set()
            # Remove the checkpoint params that are not found in model
            for key in list(package['model_state_dict'].keys()):
                if key not in init_state_dict.keys():
                    param = copy.deepcopy(package['model_state_dict'][key])
                    del package['model_state_dict'][key]
                    log('ignored: %s in checkpoint not found in model'%key, self.config)
                elif package['model_state_dict'][key].size() != init_state_dict[key].size():
                    log(package['model_state_dict'][key].size(), self.config)
                    log(init_state_dict[key].size(), self.config)
                    log('ignored: %s in checkpoint size mismatched'%key, self.config)
                    del package['model_state_dict'][key]
            # Replace the ignored checkpoint params by the init params
            for key in list(init_state_dict.keys()):
                if key not in package['model_state_dict'].keys():
                    mismatch_params.add(key)
                    log('ignored: %s in model not found in checkpoint'%key, self.config)
                    package['model_state_dict'][key] = init_state_dict[key]
            self.model.load_state_dict(package['model_state_dict'])
            if self.config.resume_training and len(mismatch_params) == 0:
                # Load steps to resume training
                if self.training_target == 'score_nets' and 'score_net_training_step' in package:
                    self.training_step = package['score_net_training_step']
                elif self.training_target == 'schedule_nets' and 'schedule_net_training_step' in package:
                    self.training_step = package['schedule_net_training_step']
            if self.config.freeze_checkpoint_params and len(mismatch_params) > 0:
                # Only update new parameters defined in model
                for key, param in self.model.named_parameters():
                    if key not in mismatch_params:
                        param.requires_grad = False
            log('Loaded checkpoint %s' % self.load, self.config)
        # Create save folder
        os.makedirs(self.exp_dir, exist_ok=True)
        self.prev_val_loss, self.min_val_loss = float("inf"), float("inf")
        self.val_no_impv, self.halving = 0, 0

    def train(self):
        """
        Start the main training process
        """
        best_state = copy.deepcopy(self.model.state_dict())
        while self.training_step < self.n_training_steps:
            # Train one epoch
            log("Start training %s from step %d ......."%(
                self.training_target, self.training_step), self.config)
            self.bmuf_trainer.check_all_processes_running()
            self.model.train()
            start = time.time()
            tr_avg_loss = self._run_one_epoch(validate=False)
            log('-' * 85, self.config)
            log('Train Summary | Step {} | Time {:.2f}s | Train Loss {:.5f}'.format(
                self.training_step, time.time()-start, tr_avg_loss), self.config)
            log('-' * 85, self.config)
            # Start validation
            log('Start validation ......', self.config)
            self.model.eval()
            start = time.time()
            with torch.no_grad():
                val_loss = self._run_one_epoch(validate=True)
            log('-' * 85, self.config)
            log('Valid Summary | Step {} | Time {:.2f}s | Valid Loss {:.5f}'.format(
                self.training_step, time.time()-start, val_loss.item()), self.config)
            log('-' * 85, self.config)
            save_or_not = self.bmuf_trainer.update_and_sync(val_loss=val_loss)
            if save_or_not:
                self.val_no_impv = 0
                if self.bmuf_trainer.rank == self.bmuf_trainer.master_node:
                    model_serialized = self.serialize()
                    file_path = os.path.join(self.exp_dir, self.training_target,
                                             '%d.pkl' % self.training_step)
                    torch.save(model_serialized, file_path)
                    log("Found better model, saved to %s" % file_path, self.config)
            if val_loss >= self.min_val_loss:
                # LR decays
                self.val_no_impv += 1
                if self.val_no_impv == self.config.patience:
                    log("No imporvement for %d epochs, early stopped!"%(
                        self.config.patience), self.config)
                    self.bmuf_trainer.kill_all_processes()
                    break
                if self.val_no_impv >= self.config.patience // 2:
                    self.model.load_state_dict(best_state)
            else:
                self.val_no_impv = 0
                self.min_val_loss = val_loss
                best_state = copy.deepcopy(self.ema_helper.state_dict())

    def _run_one_epoch(self, validate=False):
        """
        Run one epoch

        Parameters:
            validate (bool):      whether to run a valiation epoch or a training epoch
        Returns:
            average loss (float): the average training/validation loss
        """
        start = time.time()
        total_loss, total_cnt = 0, 0
        if validate and self.training_target == 'score_nets':
            # Use EMA state dict for validation
            self.valid_sampler.model.load_state_dict(self.ema_helper.state_dict())
            # To validate score nets, we use Sampler to test sample quality instead of loss
            generated_audio, _ = self.valid_sampler.sampling()
            # Compute objective scores (score = PESQ)
            quality_score = self.valid_sampler.assess(generated_audio)[0]
            return - torch.FloatTensor([quality_score])[0].cuda()
        data_loader = self.vl_loader if validate else self.tr_loader
        data_loader.dataset.reset()
        start_step = self.training_step
        for i, batch in enumerate(data_loader):
            mels, audios = list(map(lambda x: x.cuda(), batch))
            loss = self.loss_func(self.model, mels, audios)
            total_loss += loss.detach().sum()
            total_cnt += len(loss)
            avg_loss = loss.mean()
            if not validate:
                self.optimizer.zero_grad()
                avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()
                # Apply block momentum and sync parameters
                self.bmuf_trainer.update_and_sync()
                self.bmuf_trainer.check_all_processes_running()
                # n_gpus * batch_size
                self.training_step += len(loss) * self.bmuf_trainer.world_size
                if i % self.config.log_period == 0:
                    log('Train Step {} | Avg. Loss {:.5f} | New Loss {:.5f} | {:.2f}s/step'.format(
                        self.training_step, total_loss / total_cnt, avg_loss,
                        (time.time() - start) / (self.training_step - start_step)), self.config)
                if self.training_target == 'schedule_nets':
                    self.ema_helper.update(self.model.schedule_net)
                else:
                    self.ema_helper.update(self.model)
                if self.training_step >= self.n_training_steps or\
                        max(i, self.training_step - start_step) >= self.config.steps_per_epoch:
                    # Release grad memory
                    self.optimizer.zero_grad()
                    return total_loss / total_cnt
            else:
                if i % self.config.log_period == 0:
                    log('Valid Step {} | Avg. Loss {:.5f} | New Loss {:.5f} | {:.2f}s/step'.format(
                        i + 1, total_loss/total_cnt, avg_loss,
                        (time.time() - start) / (i + 1)), self.config)
        if not validate:
            # Release grad memory
            self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        return total_loss / total_cnt

    def serialize(self):
        """
        Pack the model and configurations into a dictionary

        Returns:
            package (dict): the serialized package to be saved
        """
        if self.training_target == 'schedule_nets':
            model_state = copy.deepcopy(self.model.state_dict())
            ema_state = copy.deepcopy(self.ema_helper.state_dict())
            for p in self.ema_helper.state_dict():
                model_state['schedule_net.'+p] =  ema_state[p]
        else:
            model_state = copy.deepcopy(self.ema_helper.state_dict())
        if self.config.save_fp16:
            for p in model_state:
                model_state[p] = model_state[p].half()
        package = {
            # hyper-parameter
            'config': self.config,
            # state
            'model_state_dict': model_state
        }
        if self.training_target == 'score_nets':
            package['score_net_training_step'] = self.training_step
            package['schedule_net_training_step'] = 0
        else:
            package['score_net_training_step'] = self.config.score_net_training_steps
            package['schedule_net_training_step'] = self.training_step
        return package
