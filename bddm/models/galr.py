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


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Encoder(nn.Module):

    def __init__(self, enc_dim, win_len):
        """
        1D Convoluation based Waveform Encoder

        Parameters:
            enc_dim (int): Dimension of each frame (e.g. choice in paper: ``128``).
            win_len (int): Window length for processing raw signal samples (e.g. a
                common choice: ``16``). By default, the windows are half-overlapping.
        """
        super().__init__()
        # 1D convolutional layer
        self.enc_conv = nn.Conv1d(1, enc_dim,
                                  kernel_size=win_len,
                                  stride=win_len//2, bias=False)

    def forward(self, signals):
        """
        Non-linearly encode signals from raw waveform to frame sequences.

        Parameters:
            signals (tensor): A batch of signals in shape `[B, T]`, where `T` is the
                maximum length of these `B` signals.
        Returns:
            frames (tensor): A batch of encoded feature (a.k.a. frames) sequences in shape
                `[B, D, L]`, where `B` is the batch size, `D` is the encoded feature
                dimension (enc_dim), and `L` is the length of this feature sequence.
        """
        frames = F.relu(self.enc_conv(signals.unsqueeze(1)))
        return frames


class BiLSTMproj(nn.Module):

    def __init__(self, enc_dim, hid_dim):
        """
        Locally Recurrent Layer (Sec 2.2.1 in https://arxiv.org/abs/2101.05014).
            It consists of a bi-directional LSTM followed by a linear projection.

        Parameters:
            enc_dim (int): Dimension of each frame (e.g. choice in paper: ``128``).
            hid_dim (int): Number of hidden nodes used in the Bi-LSTM.
        """
        super().__init__()
        # Bi-LSTM with learnable (h_0, c_0) state
        self.rnn = nn.LSTM(enc_dim, hid_dim,
                           1, dropout=0, batch_first=True, bidirectional=True)
        self.cell_init = nn.Parameter(torch.rand(1, 1, hid_dim))
        self.hidden_init = nn.Parameter(torch.rand(1, 1, hid_dim))

        # Linear projection layer
        self.proj = nn.Linear(hid_dim * 2, enc_dim)

    def forward(self, intra_segs):
        """
        Process through a locally recurrent layer along the intra-segment
            direction.

        Parameters:
        	frames (tensor): A batch of intra-segments in shape `[B*S, K, D]`, where
                `B` is the batch size, `S` is the number of segments, 'K' is the
                segment length (seg_len) and `D` is the feature dimension (enc_dim).
        Returns:
            lr_output (tensor): A batch of processed segments with the same shape as the input.
        """
        batch_size_seq_len = intra_segs.size(0)
        cell = self.cell_init.repeat(2, batch_size_seq_len, 1)
        hidden = self.hidden_init.repeat(2, batch_size_seq_len, 1)
        rnn_output, _ = self.rnn(intra_segs, (hidden, cell))
        lr_output = self.proj(rnn_output)
        return lr_output


class AttnPositionalEncoding(nn.Module):

    def __init__(self, enc_dim, attn_max_len=5000):
        """
        Positional Encoding for Multi-Head Attention

        Parameters:
            enc_dim (int): Dimension of each frame (e.g. choice in paper: ``128``).
            attn_max_len (int): Maximum length of the sequence to be processed by
                multi-head attention.
        """
        super().__init__()
        pe = torch.zeros(attn_max_len, enc_dim)
        position = torch.arange(0, attn_max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, enc_dim, 2).float() * (-math.log(10000.0)/enc_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Compute positional encoding

        Parameters:
        	x (tensor): the sequence to be addded with the positional encoding vector
        Returns:
            output (tensor): the encoded input
        """
        output = x + self.pe[:x.size(0), :]
        return output


class GlobalAttnLayer(nn.Module):

    def __init__(self, enc_dim, n_attn_head, attn_dropout):
        """
        Globally Attentive Layer (Sec 2.2.2 in
            https://arxiv.org/abs/2101.05014)

        Parameters:
            enc_dim (int): Dimension of each frame (e.g. choice in paper: ``128``).
            n_attn_head (int): Number of heads for the multi-head attention. (e.g.
                choice in paper: ``8``)
            attn_dropout (float): Dropout rate for multi-head attention (e.g. choice in
                paper: ``0.1``).
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(enc_dim,
            n_attn_head, dropout=attn_dropout)
        self.norm = nn.LayerNorm(enc_dim)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, inter_segs):
        """
        Process through a globally attentive layer along the inter-segment
            direction.

        Parameters:
        	inter_segs (tensor): A batch of inter-segments in shape `[S, B*K, D]`, where
                `B` is the batch size, `S` is the number of segments, 'K' is the
                segment length (seg_len) and `D` is the feature dimension (enc_dim).
        Returns:
            output (tensor): A batch of processed segments with the same shape as the input.
        """
        output, _ = self.attn(inter_segs, inter_segs, inter_segs)
        output = self.norm(output + self.dropout(output))
        return output


class DeepGlobalAttnLayer(nn.Module):

    def __init__(self, enc_dim, n_attn_head, attn_dropout, n_attn_layer=1):
        """
        A Stack of Globally Attentive Layers (Sec 2.2.2 in
            https://arxiv.org/abs/2101.05014)

        Parameters:
            enc_dim (int): Dimension of each frame (e.g. choice in paper: ``128``).
            n_attn_head (int): Number of heads for the multi-head attention. (e.g.
                choice in paper: ``8``)
            attn_dropout (float): Dropout rate for multi-head attention (e.g. choice in
                paper: ``0.1``).
            n_attn_layer (int): Number of globally attentive layers stacked. The
                setting in paper by default is ``1``.
        """
        super().__init__()
        self.attn_in_norm = nn.LayerNorm(enc_dim)
        self.pos_enc = AttnPositionalEncoding(enc_dim)
        self.attn_layer = nn.ModuleList([
            GlobalAttnLayer(enc_dim, n_attn_head, attn_dropout) for _ in range(n_attn_layer)])

    def forward(self, inter_segs):
        """
        Process through a stack of globally attentive layers.

        Parameters:
        	inter_segs (tensor): A batch of inter-segments in shape `[S, B*K, D]`, where
            `B` is the batch size, `S` is the number of segments, 'K' is the
            segment length (seg_len) and `D` is the feature dimension (enc_dim).
        Returns:
        	 output (tensor): A batch of processed segments with the same shape as the input.
        """
        output = self.attn_in_norm(inter_segs)
        output = self.pos_enc(output)
        for block in self.attn_layer:
            output = block(output)
        return output


class GALRBlock(nn.Module):

    def __init__(self, enc_dim, hid_dim, seg_len, low_dim=8, n_attn_head=8, attn_dropout=0.1):
        """
        Globally Attentive Locally Recurrent (GALR) Block (Sec. 2.2 in
            https://arxiv.org/abs/2101.05014)

        Parameters:
            enc_dim (int): Dimension of each frame (e.g. choice in paper: ``128``).
            hid_dim (int): Number of hidden nodes used in the Bi-LSTM.
            seg_len (int): Segment length for processing frame sequence (e.g. a
                ``64`` when win_len is ``16``). By default, the segments are
                half-overlapping.
            low_dim (int): Lower dimension for speeding up GALR. (Sec. 2.2.3)
                (e.g. ``8``  when seg_len is ``64``).
            n_attn_head (int): Number of heads for the multi-head attention. (e.g.
                choice in paper: ``8``)
            attn_dropout (float): Dropout rate for multi-head attention (e.g. choice in
                paper: ``0.1``).
        """
        super().__init__()
        self.low_dim = low_dim
        self.local_rnn = BiLSTMproj(enc_dim, hid_dim)
        self.local_norm = nn.GroupNorm(1, enc_dim)
        self.global_rnn = DeepGlobalAttnLayer(enc_dim, n_attn_head, attn_dropout)
        self.global_norm = nn.GroupNorm(1, enc_dim)
        self.ld_map = nn.Linear(seg_len, low_dim)
        self.ld_inv_map = nn.Linear(low_dim, seg_len)

    def forward(self, segments):
        """
        Process through a GALR block.

        Parameters:
        	segments (tensor): A batch of 3D segments in shape `[B, D, K, S]`, where
                `B` is the batch size, `S` is the number of segments, 'K' is the
                segment length (seg_len) and `D` is the feature dimension (enc_dim).
        Returns:
        	 segments (tensor): A batch of processed segments with the same shape as the input.
        """
        batch_size, feat_dim, seg_len, n_segs = segments.size()
        # Change the sequence direction for intra-segment processing
        local_input = segments.transpose(1, 3).reshape(batch_size * n_segs, seg_len, feat_dim)
        # Process through a locally recurrent layer
        local_output = self.local_rnn(local_input)
        # Reshape to match the dimensionality of the input for residual connection
        local_output = local_output.view(batch_size, n_segs, seg_len, feat_dim).transpose(1, 3).contiguous()
        # Add a layer normalization before the residual connection
        local_output = self.local_norm(local_output)
        # Add residual connection
        segments = segments + local_output

        # Change the sequence direction for intra-segment processing
        global_input = segments.permute(3, 2, 0, 1).contiguous().view(n_segs, seg_len, batch_size, feat_dim)
        # Perform low-dimensional mapping for speeding up GALR
        global_input = self.ld_map(global_input.transpose(1, -1))
        # Reshape for intra-segment processing (sequence, batch size, feature dim)
        global_input = global_input.transpose(1, -1).contiguous().view(n_segs, -1, feat_dim)
        # Process through a globally attentive layer
        global_output = self.global_rnn(global_input)
        # Reshape for low-dimensional inverse mapping
        global_output = global_output.view(n_segs, self.low_dim, -1, feat_dim).transpose(1, -1)
        # Map the low-dimensional features back to the original size
        global_output = self.ld_inv_map(global_output)
        # Reshape to match the dimensionality of the input for residual connection
        global_output = global_output.permute(2, 1, 3, 0).contiguous()
        # Add a layer normalization before the residual connection
        global_output = self.global_norm(global_output)
        # Add residual connection
        segments = segments + global_output

        return segments


class _GALR(nn.Module):

    def __init__(self, n_block, enc_dim, hid_dim, win_len, seg_len,
            low_dim=8, n_attn_head=8, attn_dropout=0.1):
        """
        Globally Attentive Locally Recurrent (GALR) Networks
            (https://arxiv.org/abs/2101.05014)

        Parameters:
            n_block (int): Number of GALR blocks (e.g. choice in paper: ``6``).
            enc_dim (int): Dimension of each frame (e.g. choice in paper: ``128``).
            hid_dim (int): Number of hidden nodes used in the Bi-LSTM.
            win_len (int): Window length for processing raw signal samples (e.g. a
                common choice: ``8``). By default, the windows are half-overlapping.
            seg_len (int): Segment length for processing frame sequence (e.g. a
                ``64`` when win_len is ``8``). By default, the segments are half-overlapping.
            low_dim (int): Lower dimension for speeding up GALR. (Sec. 2.2.3)
                (e.g. ``8``  when seg_len is ``64``).
            n_attn_head (int): Number of heads for the multi-head attention. (e.g.
                choice in paper: ``8``)
            attn_dropout (flaot): Dropout rate for multi-head attention (e.g. choice in
                paper: ``0.1``).
        """
        super().__init__()
        self.win_len = win_len
        self.seg_len = seg_len
        self.encoder = Encoder(enc_dim, win_len)
        self.bottleneck = nn.Conv1d(enc_dim, enc_dim, 1, bias=False)
        # GALR blocks
        self.blocks = nn.ModuleList([
            GALRBlock(enc_dim, hid_dim, seg_len, low_dim, n_attn_head, attn_dropout)
                for i in range(n_block)])
        # Many-to-one gated layer applied to GALR's last block
        self.block_gate = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(enc_dim, 1, 1)
        )

    def forward(self, noisy_signals):
        """
        Process through a GALR network for noise estimation

        Parameters:
        	noisy_signals (tensor): A batch of 1D signals in shape `[B, T]`, where
                `B` is the batch size, `T` is the maximum length of the `B` signals.
        Returns:
        	est_ratios (tensor): A batch of scalar noise scale ratios in `[B, 1]` shape.
        """
        noisy_signals_padded, _ = self.pad_zeros(noisy_signals)
        mix_frames = self.encoder(noisy_signals_padded)
        mix_frames = self.bottleneck(mix_frames)
        block_feature, _ = self.split_feature(mix_frames)
        for block in self.blocks:
            block_feature = block(block_feature)
        est_segments = self.block_gate(block_feature)
        est_ratios = torch.sigmoid(est_segments.mean([2, 3]))
        est_ratios = est_ratios.mean(1, keepdim=True)
        return est_ratios

    def pad_zeros(self, signals):
        """
        Pad a batch of signals with zeros before encoding.

        Parameters:
        	signals (tensor): A batch of 1D signals in shape `[B, T]`, where `B` is
                the batch size, `T` is the maximum length of the `B` signals.
        Returns:
        	signals (tensor): A batch of padded signals and the length of zeros used for
                padding. (in shape `[B, T]`)
            rest (int): the redundant spaces created in padding
        """
        batch_size, sig_len = signals.shape
        self.hop_size = self.win_len // 2
        rest = self.win_len - (self.hop_size + sig_len % self.win_len) % self.win_len
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(signals.type())
            signals = torch.cat([signals, pad], 1)
        pad_aux = torch.zeros(batch_size, self.hop_size).type(signals.type())
        signals = torch.cat([pad_aux, signals, pad_aux], 1)
        return signals, rest

    def pad_segment(self, frames):
        """
        Pad a batch of frames with zeros before segmentation.

        Parameters:
        	frames (tensor): A batch of 2D frames in shape `[B, D, L]`, where `B` is
                the batch size, `D` is the encoded feature dimension (enc_dim), and
                `L` is the length of this feature sequence.
        Returns:
        	frames (tensor): A batch of padded frames and the length of zeros used for
                padding. (in shape `[B, D, L]`)
            rest (int): the redundant spaces created in padding
        """
        batch_size, feat_dim, seq_len = frames.shape
        stride = self.seg_len // 2
        rest = self.seg_len - (stride + seq_len % self.seg_len) % self.seg_len
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, feat_dim, rest)).type(frames.type())
            frames = torch.cat([frames, pad], 2)
        pad_aux = Variable(torch.zeros(batch_size, feat_dim, stride)).type(frames.type())
        frames = torch.cat([pad_aux, frames, pad_aux], 2)
        return frames, rest

    def split_feature(self, frames):
        """
        Perform segmentation by dividing every K (seg_len) consecutive frames
            into S segments.

        Parameters:
        	frames (tensor): A batch of 2D frames in shape `[B, D, L]`, where `B` is
                the batch size, `D` is the encoded feature Dension (enc_D), and
                `L` is the length of this feature sequence.
        Returns:
        	segments (tensor): A batch of 3D segments and the length of zeros used for
                padding. (in shape `[B, D, K, S]`)
            rest (int): the redundant spaces created in padding
        """
        frames, rest = self.pad_segment(frames)
        batch_size, feat_dim, _ = frames.shape
        stride = self.seg_len // 2
        lsegs = frames[:, :, :-stride].contiguous().view(batch_size, feat_dim, -1, self.seg_len)
        rsegs = frames[:, :, stride:].contiguous().view(batch_size, feat_dim, -1, self.seg_len)
        segments = torch.cat([lsegs, rsegs], -1).view(batch_size, feat_dim, -1, self.seg_len)
        segments = segments.transpose(2, 3).contiguous()
        return segments, rest


class GALR(nn.Module):

    def __init__(self, blocks=2, input_dim=128, hidden_dim=128, window_length=8, segment_size=64):
        """
        GALR Schedule Network

        Parameters:
            blocks (int): Number of GALR blocks (e.g. choice in paper: ``6``).
            input_dim (int): Dimension of each frame (e.g. choice in paper: ``128``).
            hidden_dim (int): Number of hidden nodes used in the Bi-LSTM.
            window_length (int): Window length for processing raw signal samples (e.g. a
                common choice: ``8``). By default, the windows are half-overlapping.
            segment_size (int): Segment length for processing frame sequence (e.g. a
                ``64`` when win_len is ``8``). By default, the segments are half-overlapping.
        """
        super().__init__()
        self.ratio_nn = _GALR(blocks, input_dim, hidden_dim, window_length, segment_size)

    def forward(self, audio, scales):
        """
        Estimate the next beta scale using the GALR schedule network

        Parameters:
        	audio (tensor): A batch of 1D signals in shape `[B, T]`, where
                `B` is the batch size, `T` is the maximum length of the `B` signals.
            scales (list): [beta_nxt, delta] for computing the upper bound
        Returns:
        	beta (tensor): A batch of scalar noise scale ratios in `[B, 1, 1]` shape.
        """
        beta_nxt, delta = scales
        bounds = torch.cat([beta_nxt, delta], 1)
        mu, _ = torch.min(bounds, 1)
        mu = mu[:, None]
        ratio = self.ratio_nn(audio)
        beta = mu * ratio
        return beta.view(audio.size(0), 1, 1)
