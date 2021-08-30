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


def pit_loss(pred, label):
    """
    Permutation-invariant training (PIT) cross entropy loss function.

    Args:
      pred:  (T,C)-shaped pre-activation values
      label: (T,C)-shaped labels in {0,1}

    Returns:
      min_loss: (1,)-shape mean cross entropy
      label_perms[min_index]: permutated labels
      sigma: permutation
    """

    device = pred.device
    T = len(label)
    C = label.shape[-1]
    label_perms_indices = [
            list(p) for p in permutations(range(C))]
    P = len(label_perms_indices)
    perm_mat = torch.zeros(P, T, C, C).to(device)

    for i, p in enumerate(label_perms_indices):
        perm_mat[i, :, torch.arange(label.shape[-1]), p] = 1

    x = torch.unsqueeze(torch.unsqueeze(label, 0), -1).to(device)
    y = torch.arange(P * T * C).view(P, T, C, 1).to(device)

    broadcast_label = torch.broadcast_tensors(x, y)[0]
    allperm_label = torch.matmul(
            perm_mat, broadcast_label
            ).squeeze(-1)

    x = torch.unsqueeze(pred, 0)
    y = torch.arange(P * T).view(P, T, 1)
    broadcast_pred = torch.broadcast_tensors(x, y)[0]

    # broadcast_pred: (P, T, C)
    # allperm_label: (P, T, C)
    losses = F.binary_cross_entropy_with_logits(
               broadcast_pred,
               allperm_label,
               reduction='none')
    mean_losses = torch.mean(torch.mean(losses, dim=1), dim=1)
    min_loss = torch.min(mean_losses) * len(label)
    min_index = torch.argmin(mean_losses)
    sigma = list(permutations(range(label.shape[-1])))[min_index]

    return min_loss, allperm_label[min_index], sigma


def batch_pit_loss(ys, ts, ilens=None):
    """
    PIT loss over mini-batch.

    Args:
      ys: B-length list of predictions
      ts: B-length list of labels

    Returns:
      loss: (1,)-shape mean cross entropy over mini-batch
      sigmas: B-length list of permutation
    """
    if ilens is None:
        ilens = [t.shape[0] for t in ts]

    loss_w_labels_w_sigmas = [pit_loss(y[:ilen, :], t[:ilen, :])
                              for (y, t, ilen) in zip(ys, ts, ilens)]
    losses, _, sigmas = zip(*loss_w_labels_w_sigmas)
    loss = torch.sum(torch.stack(losses))
    n_frames = np.sum([ilen for ilen in ilens])
    loss = loss / n_frames

    return loss, sigmas


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
        spksvecs = list(zip(*spksvecs))
        ts = inputs[1]
        ss = inputs[2]
        ns = inputs[3]
        ilens = inputs[4]
        ilens = [ilen.item() for ilen in ilens]

        pit_loss, sigmas = batch_pit_loss(ys, ts, ilens)
        ss = [[i.item() for i in s] for s in ss]
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
        spksvecs = [[spkvec[:ilen] for spkvec in spksvec]
                    for spksvec, ilen in zip(spksvecs, ilens)]
        loss = torch.stack(
                [self.spk_loss(spksvec, y[:ilen], t[:ilen], s, sigma, n)
                    for(spksvec, y,  t,  s,  sigma,  n,  ilen)
                    in zip(spksvecs, ys, ts, ss, sigmas, ns, ilens)])
        loss = torch.mean(loss)

        return loss

    def spk_loss(self, spksvec, y, t, s, sigma, n):
        embeds = self.net.embed(n).squeeze()
        z = torch.sigmoid(y.transpose(1, 0))

        losses = []
        for spkid, spkvec in enumerate(spksvec):
            norm_spkvec_inv = 1.0 / torch.norm(spkvec, dim=1)
            # Normalize speaker vectors before weighted average
            spkvec = torch.mul(
                    spkvec.transpose(1, 0), norm_spkvec_inv).transpose(1, 0)
            wavg_spkvec = torch.mul(
                    spkvec.transpose(1, 0), z[spkid]).transpose(1, 0)
            sum_wavg_spkvec = torch.sum(wavg_spkvec, dim=0)
            nmz_wavg_spkvec = sum_wavg_spkvec / torch.norm(sum_wavg_spkvec)
            nmz_wavg_spkvec = torch.unsqueeze(nmz_wavg_spkvec, 0)
            norm_embeds_inv = 1.0 / torch.norm(embeds, dim=1)
            embeds = torch.mul(
                    embeds.transpose(1, 0), norm_embeds_inv).transpose(1, 0)
            dist = torch.cdist(nmz_wavg_spkvec, embeds)[0]
            d = torch.add(
                    torch.clamp(
                        self.net.alpha,
                        min=sys.float_info.epsilon) * torch.pow(dist, 2),
                    self.net.beta)

            round_t = torch.round(t.transpose(1, 0)[sigma[spkid]])
            if torch.sum(round_t) > 0:
                loss = -F.log_softmax(-d, 0)[s[sigma[spkid]]]
            else:
                loss = torch.tensor(0.0).to(y.device)
            losses.append(loss)

        return torch.mean(torch.stack(losses))


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
