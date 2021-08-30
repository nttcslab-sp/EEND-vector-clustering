# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved

import torch
import numpy as np
from eend import kaldi_data
from eend import feature


def _count_frames(data_len, size, step):
    return int((data_len - size + step) / step)


def _gen_frame_indices(data_length, size=2000, step=2000):
    i = -1
    for i in range(_count_frames(data_length, size, step)):
        yield i * step, i * step + size

    if i * step + size < data_length:
        if data_length - (i + 1) * step > 0:
            if i == -1:
                yield (i + 1) * step, data_length
            else:
                yield data_length - size, data_length


class DiarizationDatasetFromWave(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir,
            dtype=np.float32,
            chunk_size=2000,
            context_size=0,
            frame_size=1024,
            frame_shift=256,
            subsampling=1,
            rate=16000,
            input_transform=None,
            n_speakers=None,
            ):
        self.data_dir = data_dir
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.context_size = context_size
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.subsampling = subsampling
        self.rate = rate
        self.input_transform = input_transform
        self.n_speakers = n_speakers

        self.chunk_indices = []
        self.data = kaldi_data.KaldiData(self.data_dir)
        self.all_speakers = sorted(self.data.spk2utt.keys())
        self.all_n_speakers = len(self.all_speakers)
        self.all_n_speakers_arr =\
            np.arange(self.all_n_speakers,
                      dtype=np.int64).reshape(self.all_n_speakers, 1)

        # Make chunk indices: filepath, start_frame, end_frame
        for rec in self.data.wavs:
            data_len = int(self.data.reco2dur[rec] * self.rate / frame_shift)
            data_len = int(data_len / self.subsampling)
            for st, ed in _gen_frame_indices(data_len, chunk_size, chunk_size):
                self.chunk_indices.append(
                    (rec, st * self.subsampling, ed * self.subsampling))
        print(len(self.chunk_indices), " chunks")

    def __len__(self):
        return len(self.chunk_indices)

    def __getitem__(self, i):
        rec, st, ed = self.chunk_indices[i]
        filtered_segments = self.data.segments[rec]
        # speakers: the value given from data
        speakers = np.unique(
            [self.data.utt2spk[seg['utt']] for seg in filtered_segments]
            ).tolist()
        n_speakers = self.n_speakers
        if self.n_speakers < len(speakers):
            n_speakers = len(speakers)

        Y, T = feature.get_labeledSTFT(
            self.data,
            rec,
            st,
            ed,
            self.frame_size,
            self.frame_shift,
            n_speakers,
            )
        T = T.astype(np.float32)

        S_arr = -1 * np.ones(n_speakers).astype(np.int64)
        for seg in filtered_segments:
            speaker_index = speakers.index(self.data.utt2spk[seg['utt']])
            all_speaker_index = self.all_speakers.index(
                self.data.utt2spk[seg['utt']])
            S_arr[speaker_index] = all_speaker_index

        # If T[:, n_speakers - 1] == 0.0, then S_arr[n_speakers - 1] == -1,
        # so S_arr[n_speakers - 1] is not used for training,
        # e.g., in the case of training 3-spk model with 2-spk data

        Y = feature.transform(Y, self.input_transform)
        Y_spliced = feature.splice(Y, self.context_size)
        Y_ss, T_ss = feature.subsample(Y_spliced, T, self.subsampling)
        ilen = np.array(Y_ss.shape[0], dtype=np.int64)

        return Y_ss, T_ss, S_arr, self.all_n_speakers_arr, ilen

    def get_allnspk(self):
        return self.all_n_speakers


class DiarizationDatasetFromFeat(torch.utils.data.Dataset):
    def __init__(
            self,
            featlab_chunk_indices_path,
            featdim,
            ):
        self.featlab_chunk_indices_path = featlab_chunk_indices_path
        self.featdim = featdim

        self.chunk_indices = [
            (line.strip().split()[0], line.strip().split()[1])
            for line in open(featlab_chunk_indices_path)]
        print(len(self.chunk_indices), " chunks")

        # define self.all_n_speakers
        featlab_path, chunk_idx = self.chunk_indices[0]
        chunks = np.load(featlab_path, mmap_mode='r')
        chunk = chunks[int(chunk_idx)]
        chunk = np.array(chunk)
        labs_data = chunk[:, self.featdim:]
        self.all_n_speakers = np.round(labs_data[0, -2]).astype(np.int64)

    def __len__(self):
        return len(self.chunk_indices)

    def __getitem__(self, i):
        featlab_path, chunk_idx = self.chunk_indices[i]
        chunks = np.load(featlab_path, mmap_mode='r')
        # mmap_mode='r' is for accessing small fragments specified by
        # chunk_index of the file without reading the entire file into memory
        #
        # data structure info of chunks (=featlab_XXXXXXXX.npy):
        # (assuming that batch_size == 1024, chunk_size == 150
        # featdim == 345, and num_speakers == 3)
        #
        # - chunks.shape: (1024, 150, 353)
        # - chunk.shape == chunks[int(chunk_idx)].shape: (150, 353)
        # - 1) chunk[:, :345]    : feature data from audio file
        # - 2) chunk[:, 345:348] : reference speech activities of 3-speakers
        # - 3) chunk[:, 348:351] : reference speaker IDs of 3-speakers
        #                          (speaker order is same as 2))
        # - 4) chunk[:, 351]     : reference number of all speakers
        # - 5) chunk[:, 352]     : real chunk size

        chunk = chunks[int(chunk_idx)]
        chunk = np.array(chunk)
        feat_data = chunk[:, :self.featdim]
        labs_data = chunk[:, self.featdim:]
        num_speakers = (labs_data.shape[1] - 2) // 2
        y = feat_data
        t = labs_data[:, :num_speakers]
        s = np.round(labs_data[0, num_speakers:-2]).astype(np.int64)
        n = np.round(labs_data[0, -2]).astype(np.int64)
        n = np.arange(n, dtype=np.int64).reshape(n, 1)
        ilen = np.round(labs_data[0, -1]).astype(np.int64)
        return y, t, s, n, ilen

    def get_allnspk(self):
        return self.all_n_speakers
