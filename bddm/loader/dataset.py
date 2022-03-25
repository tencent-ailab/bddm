#!/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
#  Dataset and DataLoader for Neural Vocoding
#
#  Author: Max W. Y. Lam (maxwylam@tencent.com)
#  Copyright (c) 2021Tencent. All Rights Reserved
#
########################################################################


import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from scipy.io.wavfile import read

from .stft import TacotronSTFT


MAX_WAV_VALUE = 32768


class SpectrogramDataset(data.Dataset):

    def __init__(self, data_dir, n_gpus, is_sampling, sampling_rate,
            seg_len, fil_len, hop_len, win_len, mel_fmin, mel_fmax):
        """
        A torch.data.Dataset class that loads the audio files for training
            or loads the Mel-spectrogram files for sampling.

        Parameters:
            data_dir (str):      the path to the directory storing .wav/.mel files
            n_gpus (int):        the number of GPUs for training
            is_sampling (bool):  whether the dataset is used for sampling or not
            sampling_rate (int): the sampling rate of audios
            seg_len (int):       the segment length (number of samples) for training
            fil_len (int):       the filter length for computing STFT
            hop_len (int):       the hop length for computing STFT
            win_len (int):       the window length for computing STFT
            mel_fmin (int):      the minimum frequency for computing STFT
            mel_fmax (int):      the maximum frequency for computing STFT
        """
        self.n_gpus = n_gpus
        self.is_sampling = is_sampling
        self.seg_len = seg_len
        self.hop_len = hop_len
        self.n_mels = self.seg_len // self.hop_len
        self.sampling_rate = sampling_rate

        if is_sampling:
            # Find all Mel-spectrogram files in the given data directory
            self.mel_files = self.find_all_mels_in_dir(data_dir)
            if len(self.mel_files) == 0:
                # Find audios when no pre-computed mel spectrograms can be found
                self.audio_files = self.find_all_wavs_in_dir(data_dir)
                # Note that no mel file is loaded for generation
                self.mel_files = None
            else:
                # Note that no audio file is loaded for generation
                self.audio_files = None
        else:
            # Find all audio files in the given data directory
            self.audio_files = self.find_all_wavs_in_dir(data_dir)

        # Use the standard STFT operation defined in Tacotron 2
        self.stft = TacotronSTFT(filter_length=fil_len,
                                 hop_length=hop_len,
                                 win_length=win_len,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin,
                                 mel_fmax=mel_fmax)
        self.reset()

    def reset(self):
        """
        Reset the loader by shuffling the file list
        """
        if self.is_sampling and self.audio_files is None:
            np.random.shuffle(self.mel_files)
        else:
            np.random.shuffle(self.audio_files)
            self.n_mels = self.seg_len // self.hop_len
            self.n_mels = int(self.n_mels * (1 + np.random.rand()))
            # Make sure the number of samples is divisible by n_gpus
            if len(self.audio_files) % self.n_gpus != 0:
                remainder = len(self.audio_files) % self.n_gpus
                self.audio_files = self.audio_files[:-remainder]

    def find_all_wavs_in_dir(self, data_dir):
        """
        Load all .wav files in data_dir

        Parameters:
            data_dir (str):   the path to the directory storing .wav files
        Returns:
            files_list (list): the list of wav file paths
        """
        files = [f for f in Path(data_dir).glob('*.wav')]
        if len(files) == 0:
            files = [f for f in Path(data_dir).glob('*_wav.npy')]
        return files

    def find_all_mels_in_dir(self, data_dir):
        """
        Load all .mel files in data_dir

        Parameters:
            data_dir (str):   the path to the directory storing .mel files
        Returns:
            files_list (list): the list of mel file paths
        """
        files = [f for f in Path(data_dir).glob('*.mel')]
        return files

    def crop_audio_and_mel(self, audio, mel_spec):
        """
        Randomly crop audio and mel_spec into a fixed-length segment

        Parameters:
            audio (tensor):    the full audio
            mel_spec (tensor): the full mel-spectrogram computed using TacotronSTFT
        Returns:
            audio (tensor):    the cropped audio
            mel_spec (tensor): the cropped mel-spectrogram
        """
        n_mels = self.n_mels
        seg_len = n_mels * self.hop_len
        if audio.size(-1) >= seg_len:
            if mel_spec.size(-1) > n_mels:
                max_mel_start = mel_spec.size(-1) - n_mels
                mel_start = np.random.randint(0, max_mel_start)
                mel_spec = mel_spec[..., mel_start:mel_start+n_mels]
                audio_start = mel_start * self.hop_len
                audio = audio[..., audio_start:audio_start+seg_len]
            elif mel_spec.size(-1) == n_mels:
                audio = audio[..., :seg_len]
            else:
                audio = audio[..., :seg_len]
                mel_spec = F.pad(mel_spec, (0, n_mels - mel_spec.size(-1)), 'constant', 0)
        else:
            audio = F.pad(audio, (0, seg_len - audio.size(-1)), 'constant', 0)
            mel_spec = F.pad(mel_spec, (0, n_mels - mel_spec.size(-1)), 'constant', 0)
        return audio, mel_spec

    def __getitem__(self, index):
        """
        Get a pair of data (mel-spectrogram, audio) given an index

        Parameters:
            index (int):       the index for loading one sample
        Returns:
            audio_key (str):   the audio key
            mel_spec (tensor): the mel-spectrogram computed using TacotronSTFT
            audio (tensor):    the ground-truth audio
        """
        if self.is_sampling and self.audio_files is None:
            # Load Mel-spectrogram for sampling
            mel_spec = torch.load(self.mel_files[index], map_location='cpu').float()
            if mel_spec.ndim == 3:
                mel_spec = mel_spec[0]
            audio_key = str(self.mel_files[index])
            # Try to find the paired source audio
            wav_path = str(self.mel_files[index])[:-4]+'.wav'
            if os.path.isfile(wav_path):
                sampling_rate, audio = read(wav_path)
                audio = torch.from_numpy(audio[None]).float() / MAX_WAV_VALUE
                assert sampling_rate == self.sampling_rate
                return audio_key, mel_spec, audio
            else:
                return audio_key, mel_spec, []

        mel_spec = None
        # Load the audio file to torch.FloatTensor
        if str(self.audio_files[index])[-3:] == 'npy':
            audio = np.load(self.audio_files[index])[0]
            mel_spec = np.load(str(self.audio_files[index]).replace('wav', 'mel'))[0]
            # Load into torch.FloatTensor
            audio = torch.from_numpy(audio).float()
            mel_spec = torch.from_numpy(mel_spec).float()
        else:
            sampling_rate, audio = read(self.audio_files[index])
            # Make sure the sampling rate is correctly defined
            assert sampling_rate == self.sampling_rate
            # Normalize the audio into [-1, 1]
            audio = audio / MAX_WAV_VALUE
            # Load into torch.FloatTensor
            audio = torch.from_numpy(audio).float()
        # Compute Mel-spectrogram (shape = [T] -> [filter_length, L])
        if mel_spec is None:
            audio = audio[None]
            mel_spec = self.stft.mel_spectrogram(audio)
            mel_spec = torch.squeeze(mel_spec, 0)

        if not self.is_sampling:
            audio, mel_spec = self.crop_audio_and_mel(audio, mel_spec)

        if self.is_sampling:
            # Save ground-truth Mel-spectrogram into .mel file
            torch.save(mel_spec, str(self.audio_files[index])[:-4]+'.mel')
            return str(self.audio_files[index]), mel_spec, audio

        return mel_spec, audio

    def __len__(self):
        """
        Get the number of data

        Returns:
            data_len (int): the number of .wav/.mel files found in data_dir
        """
        if self.audio_files is None:
            return len(self.mel_files)
        return len(self.audio_files)


def create_train_and_valid_dataloader(config):
    """
    Create two torch.data.DataLoader for training and validation

    Parameters:
        config (namespace):     BDDM Configuration
    Returns:
        tr_loader (DataLoader): the data loader for training
        vl_loader (DataLoader): the data loader for validation
    """
    n_gpus = 1
    if 'WORLD_SIZE' in os.environ.keys():
        n_gpus = int(os.environ['WORLD_SIZE'])
    conf_keys = SpectrogramDataset.__init__.__code__.co_varnames
    data_config = {k: v for k, v in vars(config).items() if k in conf_keys}
    data_config["data_dir"] = config.train_data_dir
    data_config["is_sampling"] = False
    data_config["n_gpus"] = n_gpus
    dataset = SpectrogramDataset(**data_config)
    assert len(dataset) > 0, f"Error: No .wav can be found at {config.train_data_dir} !"
    sampler = data.distributed.DistributedSampler(dataset) if n_gpus > 1 else None
    tr_loader = data.DataLoader(dataset,
                                sampler=sampler,
                                batch_size=config.batch_size,
                                num_workers=config.n_worker,
                                pin_memory=False,
                                drop_last=True)
    data_config["data_dir"] = config.valid_data_dir
    dataset = SpectrogramDataset(**data_config)
    assert len(dataset) > 0, f"Error: No .wav can be found at {config.valid_data_dir} !"
    sampler = data.distributed.DistributedSampler(dataset) if n_gpus > 1 else None
    vl_loader = data.DataLoader(dataset,
                                sampler=sampler,
                                batch_size=1,
                                num_workers=config.n_worker,
                                pin_memory=False)
    return tr_loader, vl_loader


def create_generation_dataloader(config):
    """
    Create a torch.data.DataLoader for generation

    Parameters:
        config (namespace):      BDDM Configuration
    Returns:
        gen_loader (DataLoader): the data loader for generation
    """
    conf_keys = SpectrogramDataset.__init__.__code__.co_varnames
    data_config = {k: v for k, v in vars(config).items() if k in conf_keys}
    data_config["data_dir"] = config.gen_data_dir
    data_config["is_sampling"] = True
    data_config["n_gpus"] = 1
    dataset = SpectrogramDataset(**data_config)
    gen_loader = data.DataLoader(dataset,
                                 batch_size=1,  # variable audio length
                                 num_workers=config.n_worker,
                                 pin_memory=False)
    return gen_loader


if __name__ == "__main__":
    from argparse import Namespace
    data_config = {
        "data_dir": "LJSpeech-1.1/train_wavs",
        "sampling_rate": 22050,
        "training": True,
        "seg_len": 22050,
        "fil_len": 1024,
        "hop_len": 256,
        "win_len": 1024,
        "mel_fmin": 0.0,
        "mel_fmax": 8000.0,
        "extra_key": "extra_val"
    }
    data_config = Namespace(**data_config)
    loader = create_train_and_valid_dataloader(data_config)
    for batch in loader:
        print(len(batch))
        print(batch[0].shape, batch[1].shape)
        break
