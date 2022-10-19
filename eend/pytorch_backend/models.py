# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved

import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
from itertools import permutations
from eend.pytorch_backend.transformer import TransformerEncoder
import padertorch as pt

"""
P: number of permutation
T: number of frames
C: number of speakers (classes)
B: mini-batch size
"""

def batch_pit_loss(outputs, labels, ilens=None):
    """ calculate the batch pit loss parallelly
    Args:
        outputs (torch.Tensor): B x T x C
        labels (torch.Tensor): B x T x C
        ilens (torch.Tensor): B
    Returns:
        perm (torch.Tensor): permutation for outputs (Batch, num_spk)
        loss
    """

    if ilens is None:
        mask, scale = 1.0, outputs.shape[1]
    else:
        scale = torch.unsqueeze(torch.LongTensor(ilens), 1).to(outputs.device)
        mask = outputs.new_zeros(outputs.size()[:-1])
        for i, chunk_len in enumerate(ilens):
            mask[i, :chunk_len] += 1.0
    mask /= scale

    def loss_func(output, label):
        return torch.sum(F.binary_cross_entropy_with_logits(output, label, reduction='none') * mask, dim=-1)

    def pair_loss(outputs, labels, permutation):
        return sum([loss_func(outputs[:,:,s], labels[:,:,t]) for s, t in enumerate(permutation)]) / len(permutation)

    device = outputs.device
    num_spk = outputs.shape[-1]
    all_permutations = list(permutations(range(num_spk)))
    losses = torch.stack([pair_loss(outputs, labels, p) for p in all_permutations], dim=1)
    loss, perm = torch.min(losses, dim=1)
    perm = torch.index_select(torch.tensor(all_permutations, device=device, dtype=torch.long), 0, perm)
    return torch.mean(loss), perm


def fix_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            # remove 'module.' of DataParallel
            k = k[7:]
        if k.startswith('net.'):
            # remove 'net.' of PadertorchModel
            k = k[4:]
        new_state_dict[k] = v
    return new_state_dict


class PadertorchModel(pt.base.Model):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs):
        xs = inputs[0]
        ys, spksvecs = self.net(xs)

        return dict(prediction=ys, spksvecs=spksvecs)

    def review(self, inputs, outputs):
        ys = outputs["prediction"]
        spksvecs = outputs["spksvecs"]
        ts = inputs[1]
        ss = inputs[2]
        ns = inputs[3]
        ilens = inputs[4]
        ilens = [ilen.item() for ilen in ilens]

        pit_loss, sigmas = batch_pit_loss(ys, ts, ilens)
        if pit_loss.requires_grad:
            spk_loss = self.batch_spk_loss(
                    spksvecs, ys, ts, ss, sigmas, ns, ilens)
        else:
            spk_loss = torch.tensor(0.0).to(pit_loss.device)

        alpha = torch.clamp(self.net.alpha, min=sys.float_info.epsilon)

        return pt.summary.review_dict(
                losses={'pit_loss': pit_loss, 'spk_loss': spk_loss},
                scalars={'alpha': alpha})

    def batch_spk_loss(self, spksvecs, ys, ts, ss, sigmas, ns, ilens):
        '''
        spksvecs (List[torch.Tensor, ...]): [B x T x emb_dim, ...]
        ys (torch.Tensor): B x T x 3
        ts (torch.Tensor): B x T x 3
        ss (torch.Tensor): B x 3
        sigmas (torch.Tensor): B x 3
        ns (torch.Tensor): B x total_spk_num x 1
        ilens (List): B
        '''
        chunk_spk_num = len(spksvecs)  # 3

        len_mask = ys.new_zeros((ys.size()[:-1]))  # B x T
        for i, len_val in enumerate(ilens):
            len_mask[i,:len_val] += 1.0
        ts = ts * len_mask.unsqueeze(-1)
        len_mask = len_mask.repeat((chunk_spk_num, 1))  # B*3 x T

        spk_vecs = torch.cat(spksvecs, dim=0)  # B*3 x T x emb_dim
        # Normalize speaker vectors before weighted average
        spk_vecs = F.normalize(spk_vecs, dim=-1)

        ys = torch.permute(torch.sigmoid(ys), dims=(2, 0, 1))  # 3 x B x T
        ys = ys.reshape(-1, ys.shape[-1]).unsqueeze(-1)  # B*3 x T x 1

        weight_spk_vec = ys * spk_vecs  # B*3 x T x emb_dim
        weight_spk_vec *= len_mask.unsqueeze(-1)
        sum_spk_vec = torch.sum(weight_spk_vec, dim=1)  # B*3 x emb_dim
        norm_spk_vec = F.normalize(sum_spk_vec, dim=1)

        embeds = F.normalize(self.net.embed(ns[0]).squeeze(), dim=1)  # total_spk_num x emb_dim
        dist = torch.cdist(norm_spk_vec, embeds)  # B*3 x total_spk_num
        logits = -1.0 * torch.add(torch.clamp(self.net.alpha, min=sys.float_info.epsilon) * torch.pow(dist, 2), self.net.beta)
        label = torch.gather(ss, 1, sigmas).transpose(0, 1).reshape(-1, 1).squeeze()  # B*3
        label[label==-1] = 0
        valid_spk_mask = torch.gather(torch.sum(ts, dim=1), 1, sigmas).transpose(0, 1)  # 3 x B
        valid_spk_mask = (torch.flatten(valid_spk_mask) > 0).float()  # B*3

        valid_spk_loss_num = torch.sum(valid_spk_mask).item()
        if valid_spk_loss_num > 0:
            # uncomment the line below, nly divide the number of samples contain speakers
            # loss = F.cross_entropy(logits, label, reduction='none') * valid_spk_mask / valid_spk_loss_num
            # uncomment the line below, the loss result is same as batch_spk_loss
            loss = F.cross_entropy(logits, label, reduction='none') * valid_spk_mask / valid_spk_mask.shape[0]
            return torch.sum(loss)
        else:
            return torch.tensor(0.0).to(ys.device)


class TransformerDiarization(nn.Module):
    def __init__(self,
                 n_speakers,
                 in_size,
                 n_units,
                 n_heads,
                 n_layers,
                 dropout_rate,
                 all_n_speakers,
                 d
                 ):
        super(TransformerDiarization, self).__init__()
        self.enc = TransformerEncoder(
            in_size, n_layers, n_units, h=n_heads, dropout_rate=dropout_rate)
        self.linear = nn.Linear(n_units, n_speakers)

        for i in range(n_speakers):
            setattr(self, '{}{:d}'.format("linear", i), nn.Linear(n_units, d))

        self.n_speakers = n_speakers
        self.embed = nn.Embedding(all_n_speakers, d)
        self.alpha = nn.Parameter(torch.rand(1)[0] + torch.Tensor([0.5])[0])
        self.beta = nn.Parameter(torch.rand(1)[0] + torch.Tensor([0.5])[0])

    def modfy_emb(self, weight):
        self.embed = nn.Embedding.from_pretrained(weight)

    def forward(self, xs):
        # Since xs is pre-padded, the following code is extra,
        # but necessary for reproducibility
        xs = nn.utils.rnn.pad_sequence(xs, padding_value=-1, batch_first=True)
        pad_shape = xs.shape
        emb = self.enc(xs)
        ys = self.linear(emb)
        ys = ys.reshape(pad_shape[0], pad_shape[1], -1)

        spksvecs = []
        for i in range(self.n_speakers):
            spkivecs = getattr(self, '{}{:d}'.format("linear", i))(emb)
            spkivecs = spkivecs.reshape(pad_shape[0], pad_shape[1], -1)
            spksvecs.append(spkivecs)

        return ys, spksvecs

    def batch_estimate(self, xs):
        out = self(xs)
        ys = out[0]
        spksvecs = out[1]
        spksvecs = list(zip(*spksvecs))
        outputs = [
                self.estimate(spksvec, y)
                for (spksvec, y) in zip(spksvecs, ys)]
        outputs = list(zip(*outputs))

        return outputs

    def batch_estimate_with_perm(self, xs, ts, ilens=None):
        out = self(xs)
        ys = out[0]
        if ts[0].shape[1] > ys[0].shape[1]:
            # e.g. the case of training 3-spk model with 4-spk data
            add_dim = ts[0].shape[1] - ys[0].shape[1]
            y_device = ys[0].device
            zeros = [torch.zeros(ts[0].shape).to(y_device)
                     for i in range(len(ts))]
            _ys = []
            for zero, y in zip(zeros, ys):
                _zero = zero
                _zero[:, :-add_dim] = y
                _ys.append(_zero)
            _, sigmas = batch_pit_loss(_ys, ts, ilens)
        else:
            _, sigmas = batch_pit_loss(ys, ts, ilens)
        spksvecs = out[1]
        spksvecs = list(zip(*spksvecs))
        outputs = [self.estimate(spksvec, y)
                   for (spksvec, y) in zip(spksvecs, ys)]
        outputs = list(zip(*outputs))
        zs = outputs[0]

        if ts[0].shape[1] > ys[0].shape[1]:
            # e.g. the case of training 3-spk model with 4-spk data
            add_dim = ts[0].shape[1] - ys[0].shape[1]
            z_device = zs[0].device
            zeros = [torch.zeros(ts[0].shape).to(z_device)
                     for i in range(len(ts))]
            _zs = []
            for zero, z in zip(zeros, zs):
                _zero = zero
                _zero[:, :-add_dim] = z
                _zs.append(_zero)
            zs = _zs
            outputs[0] = zs
        outputs.append(sigmas)

        # outputs: [zs, nmz_wavg_spk0vecs, nmz_wavg_spk1vecs, ..., sigmas]
        return outputs

    def estimate(self, spksvec, y):
        outputs = []
        z = torch.sigmoid(y.transpose(1, 0))

        outputs.append(z.transpose(1, 0))
        for spkid, spkvec in enumerate(spksvec):
            norm_spkvec_inv = 1.0 / torch.norm(spkvec, dim=1)
            # Normalize speaker vectors before weighted average
            spkvec = torch.mul(
                    spkvec.transpose(1, 0), norm_spkvec_inv
                    ).transpose(1, 0)
            wavg_spkvec = torch.mul(
                    spkvec.transpose(1, 0), z[spkid]
                    ).transpose(1, 0)
            sum_wavg_spkvec = torch.sum(wavg_spkvec, dim=0)
            nmz_wavg_spkvec = sum_wavg_spkvec / torch.norm(sum_wavg_spkvec)
            outputs.append(nmz_wavg_spkvec)

        # outputs: [z, nmz_wavg_spk0vec, nmz_wavg_spk1vec, ...]
        return outputs
