#!/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
#  BDDM Sampler (Supports Noise Scheduling and Sampling)
#
#  Author: Max W. Y. Lam (maxwylam@tencent.com)
#  Copyright (c) 2021Tencent. All Rights Reserved
#
########################################################################


from __future__ import absolute_import

import os
import librosa

import torch
import numpy as np
from scipy.io.wavfile import write as wavwrite
from pystoi import stoi
from pypesq import pesq

from bddm.utils.log_utils import log
from bddm.utils.check_utils import check_score_network
from bddm.utils.check_utils import check_schedule_network
from bddm.utils.diffusion_utils import compute_diffusion_params
from bddm.utils.diffusion_utils import map_noise_scale_to_time_step
from bddm.models import get_score_network, get_schedule_network
from bddm.loader.dataset import create_generation_dataloader
from bddm.loader.dataset import create_train_and_valid_dataloader


class Sampler(object):

    metric2index = {"PESQ": 0, "STOI": 1}

    def __init__(self, config):
        """
        Sampler Class, implements the Noise Scheduling and Sampling algorithms in BDDMs

        Parameters:
            config (namespace): BDDM Configuration
        """
        self.config = config
        self.exp_dir = config.exp_dir
        self.clip = config.grad_clip
        self.load = config.load
        self.model = get_score_network(config).cuda().eval()
        self.schedule = None
        # Initialize diffusion parameters using a pre-specified linear schedule
        noise_schedule = torch.linspace(config.beta_0, config.beta_T, config.T).cuda()
        self.diff_params = compute_diffusion_params(noise_schedule)
        if self.config.command != 'train':
            # Find schedule net, if not trained then use DDPM or DDIM sampling mode
            schedule_net_trained, schedule_net_path = check_schedule_network(config)
            if not schedule_net_trained:
                _, score_net_path = check_score_network(config)
                assert score_net_path is not None, 'Error: No score network can be found!'
                self.config.load = score_net_path
            else:
                self.config.load = schedule_net_path
                self.model.schedule_net = get_schedule_network(config).cuda().eval()

        # Perform noise scheduling when noise schedule file (.schedule) is not given
        if self.config.command == 'generate' and self.config.sampling_noise_schedule != '':
            # Generation mode given pre-searched noise schedule
            self.schedule = torch.load(self.config.sampling_noise_schedule, map_location='cpu')

        self.reset()

    def reset(self):
        """
        Reset sampling environment
        """
        if self.config.command != 'train' and self.load != '':
            package = torch.load(self.load, map_location=lambda storage, loc: storage.cuda())
            self.model.load_state_dict(package['model_state_dict'])
            log('Loaded checkpoint %s' % self.load, self.config)

        if self.config.command == 'generate':
            # Given Mel-spectrogram directory for speech vocoding
            self.gen_loader = create_generation_dataloader(self.config)
        else:
            # Sample a reference audio sample for noise scheduling
            _, self.vl_loader = create_train_and_valid_dataloader(self.config)
            self.draw_reference_data_pair()

    def draw_reference_data_pair(self):
        """
        Draw a new input-output pair for noise scheduling
        """
        self.ref_spec, self.ref_audio = next(iter(self.vl_loader))
        self.ref_spec, self.ref_audio = self.ref_spec.cuda(), self.ref_audio.cuda()

    def generate(self):
        """
        Start the generation process
        """
        generate_dir = os.path.join(self.exp_dir, 'generated')
        os.makedirs(generate_dir, exist_ok=True)
        scores = {metric: [] for metric in self.metric2index}
        for filepath, mel_spec, audio in self.gen_loader:
            mel_spec = mel_spec.cuda()
            generated_audio, n_steps = self.sampling(schedule=self.schedule,
                                                     condition=mel_spec,
                                                     ddim=self.config.use_ddim_steps)
            audio_key = '.'.join(filepath[0].split('/')[-1].split('.')[:-1])

            if len(audio) > 0:
                # Assess the generated audio with the reference audio
                self.ref_audio = audio
                score = self.assess(generated_audio, audio_key=audio_key)
                for metric in self.metric2index:
                    scores[metric].append(score[self.metric2index[metric]])

            model_name = 'BDDM' if self.schedule is not None else (
                'DDIM' if self.config.use_ddim_steps else 'DDPM')
            generated_file = os.path.join(generate_dir,
                '%s_by_%s-%d.wav'%(audio_key, model_name, n_steps))
            wavwrite(generated_file, self.config.sampling_rate,
                     generated_audio.squeeze().cpu().numpy())
            log('Generated '+generated_file, self.config)
        log('Avg Scores: PESQ = %.3f +/- %.3f, %.5f +/- %.5f'%(
            np.mean(scores['PESQ']), 1.96 * np.std(scores['PESQ']),
            np.mean(scores['STOI']), 1.96 * np.std(scores['STOI'])), self.config)

    def noise_scheduling(self, ddim=False):
        """
        Start the noise scheduling process

        Parameters:
            ddim (bool): whether to use the DDIM's p_theta for noise scheduling or not
        Returns:
            ts_infer (tensor): the step indices estimated by BDDM
            a_infer (tensor):  the alphas estimated by BDDM
            b_infer (tensor):  the betas estimated by BDDM
            s_infer (tensor):  the std. deviations estimated by BDDM
        """
        max_steps = self.diff_params["N"]
        alpha = self.diff_params["alpha"]
        alpha_param = self.diff_params["alpha_param"]
        beta_param = self.diff_params["beta_param"]
        min_beta = self.diff_params["beta"].min()
        betas = []
        x = torch.normal(0, 1, size=self.ref_audio.shape).cuda()
        with torch.no_grad():
            b_cur = torch.ones(1, 1, 1).cuda() * beta_param
            a_cur = torch.ones(1, 1, 1).cuda() * alpha_param
            for n in range(max_steps - 1, -1, -1):
                step = map_noise_scale_to_time_step(a_cur.squeeze().item(), alpha)
                if step >= 0:
                    betas.append(b_cur.squeeze().item())
                else:
                    break
                ts = (step * torch.ones((1, 1))).cuda()
                e = self.model((x, self.ref_spec.clone(), ts,))
                a_nxt = a_cur / (1 - b_cur).sqrt()
                if ddim:
                    c1 = a_nxt / a_cur
                    c2 = -(1 - a_cur**2.).sqrt() * c1
                    x = c1 * x + c2 * e
                    c3 = (1 - a_nxt**2.).sqrt()
                    x = x + c3 * e
                else:
                    x = x - b_cur / torch.sqrt(1 - a_cur**2.) * e
                    x = x / torch.sqrt(1 - b_cur)
                    if n > 0:
                        z = torch.normal(0, 1, size=self.ref_audio.shape).cuda()
                        x = x + torch.sqrt((1 - a_nxt**2.) / (1 - a_cur**2.) * b_cur) * z
                a_nxt, beta_nxt = a_cur, b_cur
                a_cur = a_nxt / (1 - beta_nxt).sqrt()
                if a_cur > 1:
                    break
                b_cur = self.model.schedule_net(x.squeeze(1),
                    (beta_nxt.view(-1, 1), (1 - a_cur**2.).view(-1, 1)))
                if b_cur.squeeze().item() < min_beta:
                    break
        b_infer = torch.FloatTensor(betas[::-1]).cuda()
        a_infer = 1 - b_infer
        s_infer = b_infer + 0
        for n in range(1, len(b_infer)):
            a_infer[n] *= a_infer[n-1]
            s_infer[n] *= (1 - a_infer[n-1]) / (1 - a_infer[n])
        a_infer = torch.sqrt(a_infer)
        s_infer = torch.sqrt(s_infer)

        # Mapping noise scales to time steps
        ts_infer = []
        for n in range(len(b_infer)):
            step = map_noise_scale_to_time_step(a_infer[n], alpha)
            if step >= 0:
                ts_infer.append(step)
        ts_infer = torch.FloatTensor(ts_infer)
        return ts_infer, a_infer, b_infer, s_infer

    def sampling(self, schedule=None, condition=None,
                 ddim=0, return_sequence=False, audio_size=None):
        """
        Perform the sampling algorithm

        Parameters:
            schedule (list):        the [ts_infer, a_infer, b_infer, s_infer]
                                    returned by the noise scheduling algorithm
            condition (tensor):     the condition for computing scores
            ddim (bool):            whether to use the DDIM for sampling or not
            return_sequence (bool): whether returning all steps' samples or not
            audio_size (list):      the shape of the audio to be sampled
        Returns:
            xs (list):              (if return_sequence) the list of generated audios
            x (tensor):             the generated audio(s) in shape=audio_size
            N (int):                the number of sampling steps
        """

        n_steps = self.diff_params["T"]
        if condition is None:
            condition = self.ref_spec

        if audio_size is None:
            audio_length = condition.size(-1) * self.config.hop_len
            audio_size = (1, 1, audio_length)

        if schedule is None:
            if ddim > 1:
                # Use DDIM (linear) for sampling ({ddim} steps)
                ts_infer = torch.linspace(0, n_steps - 1, ddim).long()
                a_infer = self.diff_params["alpha"].index_select(0, ts_infer.cuda())
                b_infer = self.diff_params["beta"].index_select(0, ts_infer.cuda())
                s_infer = self.diff_params["sigma"].index_select(0, ts_infer.cuda())
            else:
                # Use DDPM for sampling (complete T steps)
                # P.S. if ddim = 1, run DDIM reverse process for T steps
                ts_infer = torch.linspace(0, n_steps - 1, n_steps)
                a_infer = self.diff_params["alpha"]
                b_infer = self.diff_params["beta"]
                s_infer = self.diff_params["sigma"]
        else:
            ts_infer, a_infer, b_infer, s_infer = schedule

        sampling_steps = len(ts_infer)

        x = torch.normal(0, 1, size=audio_size).cuda()
        if return_sequence:
            xs = []
        with torch.no_grad():
            for n in range(sampling_steps - 1, -1, -1):
                if sampling_steps > 50 and (sampling_steps - n) % 50 == 0:
                    # Log progress per 50 steps when sampling_steps is large
                    log('\tComputed %d / %d steps !'%(
                        sampling_steps - n, sampling_steps), self.config)
                ts = (ts_infer[n] * torch.ones((1, 1))).cuda()
                e = self.model((x, condition, ts,))
                if ddim:
                    if n > 0:
                        a_nxt = a_infer[n - 1]
                    else:
                        a_nxt = a_infer[n] / (1 - b_infer[n]).sqrt()
                    c1 = a_nxt / a_infer[n]
                    c2 = -(1 - a_infer[n]**2.).sqrt() * c1
                    c3 = (1 - a_nxt**2.).sqrt()
                    x = c1 * x + (c2 + c3) * e
                else:
                    x = x - b_infer[n] / torch.sqrt(1 - a_infer[n]**2.) * e
                    x = x / torch.sqrt(1 - b_infer[n])
                    if n > 0:
                        z = torch.normal(0, 1, size=audio_size).cuda()
                        x = x + s_infer[n] * z
                if return_sequence:
                    xs.append(x)
        if return_sequence:
            return xs
        return x, sampling_steps

    def noise_scheduling_with_params(self, alpha_param, beta_param):
        """
        Run noise scheduling for once given the (alpha_param, beta_param) pair

        Parameters:
            alpha_param (float): a hyperparameter defining the alpha_hat value at step N
            beta_param (float):  a hyperparameter defining the beta_hat value at step N
        """
        log('TRY alpha_param=%.2f, beta_param=%.2f:'%(
            alpha_param, beta_param), self.config)
        # Define the pair key
        key = '%.2f,%.2f' % (alpha_param, beta_param)
        # Set alpha_param and beta_param in self.diff_params
        self.diff_params['alpha_param'] = alpha_param
        self.diff_params['beta_param'] = beta_param
        # Use DDPM reverse process for noise scheduling
        ddpm_schedule = self.noise_scheduling(ddim=False)
        log("\tSearched a %d-step schedule using DDPM reverse process" % (
            len(ddpm_schedule[0])), self.config)
        generated_audio, _ = self.sampling(schedule=ddpm_schedule)
        # Compute objective scores
        ddpm_score = self.assess(generated_audio)
        # Get the number of sampling steps with this schedule
        steps = len(ddpm_schedule[0])
        # Compare the performance with previous same-step schedule using the metric
        if steps not in self.steps2score:
            # Save the first schedule with this number of steps
            self.steps2score[steps] = [key, ] + ddpm_score
            self.steps2schedule[steps] = ddpm_schedule
            log('\tFound the first %d-step schedule: (PESQ, STOI) = (%.2f, %.3f)'%(
                steps, ddpm_score[0], ddpm_score[1]), self.config)
        elif ddpm_score[0] > self.steps2score[steps][1] and ddpm_score[1] > self.steps2score[steps][2]:
            # Found a better same-step schedule achieving a higher score
            log('\tFound a better %d-step schedule: (PESQ, STOI) = (%.2f, %.3f) -> (%.2f, %.3f)'%(
                steps, self.steps2score[steps][1], self.steps2score[steps][2],
                ddpm_score[0], ddpm_score[1]), self.config)
            self.steps2score[steps] = [key, ] + ddpm_score
            self.steps2schedule[steps] = ddpm_schedule
        # Use DDIM reverse process for noise scheduling
        ddim_schedule = self.noise_scheduling(ddim=True)
        log("\tSearched a %d-step schedule using DDIM reverse process" % (
            len(ddim_schedule[0])), self.config)
        generated_audio, _ = self.sampling(schedule=ddim_schedule)
        # Compute objective scores
        ddim_score = self.assess(generated_audio)
        # Get the number of sampling steps with this schedule
        steps = len(ddim_schedule[0])
        # Compare the performance with previous same-step schedule using the metric
        if steps not in self.steps2score:
            # Save the first schedule with this number of steps
            self.steps2score[steps] = [key, ] + ddim_score
            self.steps2schedule[steps] = ddim_schedule
            log('\tFound the first %d-step schedule: (PESQ, STOI) = (%.2f, %.3f)'%(
                steps, ddim_score[0], ddim_score[1]), self.config)
        elif ddim_score[0] > self.steps2score[steps][1] and ddim_score[1] > self.steps2score[steps][2]:
            # Found a better same-step schedule achieving a higher score
            log('\tFound a better %d-step schedule: (PESQ, STOI) = (%.2f, %.3f) -> (%.2f, %.3f)'%(
                steps, self.steps2score[steps][1], self.steps2score[steps][2],
                ddim_score[0], ddim_score[1]), self.config)
            self.steps2score[steps] = [key, ] + ddim_score
            self.steps2schedule[steps] = ddim_schedule

    def noise_scheduling_without_params(self):
        """
        Search for the best noise scheduling hyperparameters: (alpha_param, beta_param)
        """
        # Noise scheduling mode, given N
        self.reverse_process = 'BDDM'
        assert 'N' in vars(self.config).keys(), 'Error: N is undefined for BDDM!'
        self.diff_params["N"] = self.config.N
        # Init search result dictionaries
        self.steps2schedule, self.steps2score = {}, {}
        search_bins = int(self.config.bddm_search_bins)
        # Define search range of alpha_param
        alpha_last = self.diff_params["alpha"][-1].squeeze().item()
        alpha_first = self.diff_params["alpha"][0].squeeze().item()
        alpha_diff = (alpha_first - alpha_last) / (search_bins + 1)
        alpha_param_list = [alpha_last + alpha_diff * (i + 1) for i in range(search_bins)]
        # Define search range of beta_param
        beta_diff = 1. / (search_bins + 1)
        beta_param_list = [beta_diff * (i + 1) for i in range(search_bins)]
        # Search for beta_param and alpha_param, take O(search_bins^2)
        for beta_param in beta_param_list:
            for alpha_param in alpha_param_list:
                if alpha_param > (1 - beta_param) ** 0.5:
                    # Invalid range
                    continue
                # Update the scores and noise schedules with (alpha_param, beta_param)
                self.noise_scheduling_with_params(alpha_param, beta_param)
        # Lastly, repeat the random starting point (x_hat_N) and choose the best schedule
        noise_schedule_dir = os.path.join(self.exp_dir, 'noise_schedules')
        os.makedirs(noise_schedule_dir, exist_ok=True)
        steps_list = list(self.steps2score.keys())
        for steps in steps_list:
            log("-"*80, self.config)
            log("Select the best out of %d x_hat_N ~ N(0,I) for %d steps:"%(
                self.config.noise_scheduling_attempts, steps), self.config)
            # Get current best pair
            key = self.steps2score[steps][0]
            # Get back the best (alpha_param, beta_param) pair for a given steps
            alpha_param, beta_param = list(map(float, key.split(',')))
            # Repeat K times for a given number of steps
            for _ in range(self.config.noise_scheduling_attempts):
                # Random +/- 5%
                _alpha_param = alpha_param * (0.95 + np.random.rand() * 0.1)
                _beta_param = beta_param * (0.95 + np.random.rand() * 0.1)
                # Update the scores and noise schedules with (alpha_param, beta_param)
                self.noise_scheduling_with_params(_alpha_param, _beta_param)
        # Save the best searched noise schedule ({N}steps_{key}_{metric}{best_score}.ns)
        for steps in sorted(self.steps2score.keys(), reverse=True):
            filepath = os.path.join(noise_schedule_dir, '%dsteps_PESQ%.2f_STOI%.3f.ns'%(
                steps, self.steps2score[steps][1], self.steps2score[steps][2]))
            torch.save(self.steps2schedule[steps], filepath)
            log("Saved searched schedule: %s" % filepath, self.config)

    def assess(self, generated_audio, audio_key=None):
        """
        Assess the generated audio using objective metrics: PESQ and STOI.
            P.S. Users should first install pypesq and pystoi using pip

        Parameters:
            generated_audio (tensor): the generated audio to be assessed
            audio_key (str):          the key of the respective audio
        Returns:
            pesq_score (float):       the PESQ score (the higher the better)
            stoi_score (float):       the STOI score (the higher the better)
        """
        est_audio = generated_audio.squeeze().cpu().numpy()
        ref_audio = self.ref_audio.squeeze().cpu().numpy()
        if est_audio.shape[-1] > ref_audio.shape[-1]:
            est_audio = est_audio[..., :ref_audio.shape[-1]]
        else:
            ref_audio = ref_audio[..., :est_audio.shape[-1]]
        # Compute STOI using PySTOI
        # PySTOI (https://github.com/mpariente/pystoi)
        stoi_score = stoi(ref_audio, est_audio, self.config.sampling_rate, extended=False)
        # Resample audio to 16K Hz to compute PESQ using PyPESQ (supports only 8K / 16K)
        # PyPESQ (https://github.com/vBaiCai/python-pesq)
        if self.config.sampling_rate != 16000:
            est_audio_16k = librosa.resample(est_audio, self.config.sampling_rate, 16000)
            ref_audio_16k = librosa.resample(ref_audio, self.config.sampling_rate, 16000)
            pesq_score = pesq(ref_audio_16k, est_audio_16k, 16000)
        else:
            pesq_score = pesq(ref_audio, est_audio, 16000)
        # Log scores
        log('\t%sScores: PESQ = %.3f, STOI = %.5f'%(
            '' if audio_key is None else audio_key+' ', pesq_score, stoi_score), self.config)
        # Return scores: the higher the better
        return [pesq_score, stoi_score]
